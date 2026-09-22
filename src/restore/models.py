"""T5 models: U-Nets with a ResNet-18 ImageNet backbone."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from restore.detector import select_threshold as numpy_select_threshold


def _replace_batchnorm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(8, child.num_features))
        else:
            _replace_batchnorm(child)


def resnet18_encoder(weights: str = "DEFAULT") -> Tuple[nn.Module, int]:
    backbone = models.resnet18(weights=weights)
    _replace_batchnorm(backbone)
    features = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )
    return features, 512


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.encoder, width = resnet18_encoder()
        if in_channels != 3:
            original = self.encoder[0]
            adapted = nn.Conv2d(
                in_channels,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=False,
            )
            with torch.no_grad():
                adapted.weight[:, :3] = original.weight / (in_channels / 3.0)
                if in_channels > 3:
                    adapted.weight[:, 3:] = original.weight.mean(dim=1, keepdim=True)
            self.encoder[0] = adapted
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True
        self.bottleneck = nn.Sequential(
            nn.Conv2d(width, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
        )
        self.head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        damage_scores = self.head(self.bottleneck(features))
        height, width = image.shape[-2:]
        if tuple(damage_scores.shape[-2:]) != (height, width):
            damage_scores = nn.functional.interpolate(
                damage_scores, size=(height, width), mode="bilinear", align_corners=False
            )
        return damage_scores


class DetectorNet(TinyUNet):
    def __init__(self) -> None:
        super().__init__(in_channels=3, out_channels=1)


class RestorerNet(TinyUNet):
    def __init__(self) -> None:
        super().__init__(in_channels=4, out_channels=3)


def build_models() -> Dict[str, nn.Module]:
    return {"detector": DetectorNet(), "restorer": RestorerNet()}


def weighted_bce_dice_torch(
    damage_scores: torch.Tensor, damage_mask: torch.Tensor, pos_weight: float = 4.0
) -> torch.Tensor:
    probs = torch.sigmoid(damage_scores)
    truth = damage_mask.float()
    bce = nn.functional.binary_cross_entropy_with_logits(
        damage_scores, truth, pos_weight=torch.tensor(pos_weight, device=damage_scores.device)
    )
    dice = 1.0 - (2.0 * (probs * truth).sum() + 1e-6) / (
        probs.sum() + truth.sum() + 1e-6
    )
    return bce + dice


def restorer_loss_torch(
    fill: torch.Tensor, pristine: torch.Tensor, damage_mask: torch.Tensor,
    mask_weight: float = 4.0,
) -> torch.Tensor:
    squared = (fill - pristine) ** 2
    if squared.dim() == 4:
        per_pixel = squared.mean(dim=1)
    else:
        per_pixel = squared
    weights = torch.where(
        damage_mask > 0.5,
        torch.tensor(mask_weight, device=squared.device),
        torch.tensor(1.0, device=squared.device),
    )
    return (per_pixel * weights).mean()


def train_one_epoch(
    stage_models: Dict[str, nn.Module],
    triplets: list,
    device: str = "cpu",
    steps: int = 2,
    epochs: int = 1,
) -> Dict[str, float]:
    detector_net = stage_models["detector"].to(device).train()
    restorer_net = stage_models["restorer"].to(device).train()
    detector_opt = torch.optim.Adam(detector_net.parameters(), lr=1e-3)
    restorer_opt = torch.optim.Adam(restorer_net.parameters(), lr=1e-3)
    detector_sum = 0.0
    restorer_sum = 0.0
    take = triplets[: max(1, min(len(triplets), steps))]
    count = max(1, len(take)) * max(1, epochs)
    for _ in range(max(1, epochs)):
        for triplet in take:
            pristine = np.asarray(triplet["pristine"], dtype=np.float32)
            damaged_np = np.asarray(triplet["damaged"], dtype=np.float32)
            mask_np = np.asarray(triplet["damage_mask"])
            damaged = (
                torch.from_numpy(damaged_np).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
            target = torch.from_numpy(pristine).permute(2, 0, 1).unsqueeze(0).to(device)
            detector_opt.zero_grad()
            detector_loss = weighted_bce_dice_torch(detector_net(damaged), mask)
            detector_loss.backward()
            detector_opt.step()
            detector_sum += float(detector_loss.detach())
            restorer_opt.zero_grad()
            conditioned = torch.cat(
                [damaged, torch.sigmoid(detector_net(damaged)).detach()], dim=1
            )
            restorer_loss = restorer_loss_torch(restorer_net(conditioned), target, mask)
            restorer_loss.backward()
            restorer_opt.step()
            restorer_sum += float(restorer_loss.detach())
    return {
        "detector_loss": detector_sum / count,
        "restorer_loss": restorer_sum / count,
    }


def torch_train_fn(
    triplets: list, config: dict, device: str = "cpu"
) -> Dict[str, object]:
    import restore.baseline as baseline

    from restore.pipeline import composite_output

    stage_models = build_models()
    epochs = int(config.get("epochs", 3))
    steps = int(config.get("steps", len(triplets)))
    train_one_epoch(stage_models, triplets, device=device, steps=steps, epochs=epochs)
    detector_net = stage_models["detector"].to(device).eval()
    restorer_net = stage_models["restorer"].to(device).eval()
    scores_all = []
    fills_all = []
    with torch.no_grad():
        for triplet in triplets:
            damaged_np = np.asarray(triplet["damaged"], dtype=np.float32)
            damaged = (
                torch.from_numpy(damaged_np)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            damage_scores = detector_net(damaged).squeeze(0).squeeze(0).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-damage_scores))
            scores_all.append(probs.ravel())
            conditioned = torch.cat(
                [damaged, torch.sigmoid(detector_net(damaged))], dim=1
            )
            fill = (
                restorer_net(conditioned)
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            fill = np.clip(fill, 0.0, 1.0)
            mask_np = np.asarray(triplet["damage_mask"])
            predicted = (probs >= 0.5).astype(np.uint8)
            learned = composite_output(damaged_np, fill, predicted)
            classical = baseline.median_fill(damaged_np, predicted)
            fills_all.append(learned if np.abs(learned - damaged_np).sum() > 0 else classical)
    stacked_scores = np.concatenate(scores_all)
    stacked_truth = np.concatenate(
        [np.asarray(t["damage_mask"]).ravel() for t in triplets]
    )
    threshold = float(
        numpy_select_threshold(stacked_scores, stacked_truth.astype(np.uint8))
    )
    state = {
        name: {
            key: value.detach().cpu().numpy().tolist()
            for key, value in model.state_dict().items()
        }
        for name, model in stage_models.items()
    }
    params = sum(p.numel() for model in stage_models.values() for p in model.parameters())
    return {
        "weights": {"unet_state": state, "backbone": "resnet18-imagenet", "params": params},
        "threshold": threshold,
        "fills": fills_all,
        "detector_scores": stacked_scores,
        "epochs": epochs,
        "steps": steps,
    }


def save_weights(weights: Dict[str, object], path: str) -> str:
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(weights, handle)
    return path
