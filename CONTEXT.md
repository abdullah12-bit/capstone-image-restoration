# Image Restoration Capstone

Deep-learning system that restores and enhances degraded historical visual content, built as the ZAKA capstone project.

## Language

**Pristine image**:
The clean reference photo; the restoration target.
_Avoid_: ground truth, original, clean image

**Damaged image**:
The degraded input photo; the restoration source.
_Avoid_: dirty image, input, degraded image

**Restoration**:
Repairing damaged areas, correcting fading, and enhancing overall quality of a damaged image.
_Avoid_: enhancement, fix, reconstruction

**Pair**:
One damaged image and its matching pristine image used for supervised training.
_Avoid_: sample, example, couple

**Smoke run**:
A short training run on ~200 pairs that validates the pipeline before full training.
_Avoid_: trial run, test run, dry run

**Classical baseline**:
A non-learned restoration technique (e.g. Telea inpainting, median filter) our model must beat.
_Avoid_: baseline, traditional method, classical method

**Backbone**:
A network pretrained on a different task (e.g. ImageNet, generic denoising) that our restoration training starts from.
_Avoid_: pretrained model, foundation model, encoder

**Composite output**:
Final image keeping original pixels everywhere except detected-damage regions, which come from the restorer.
_Avoid_: blended output, merged image

**Damage mask**:
Pixel map marking which regions are damaged and must be restored.
_Avoid_: mask, segmentation, detection map
