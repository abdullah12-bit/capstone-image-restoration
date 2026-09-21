# Shared Goal — Capstone: Image Restoration (ZAKA)

Locked with the user across intent-grilling Rounds 1–4 (Lavish pages `grill-round-1` … `grill-round-4`).
This phase (intent + how-we-work) is DONE. A separate building-grilling phase follows, driven by skills the user will invoke by hand.

## The goal in one line

Finish the whole capstone end to end — proposal → data pipeline → trained model → eval → demo → blog/report — with the agent building everything autonomously from the terminal and the user stepping in only for taste and meaningful decisions.

## How we work (locked)

- **Finish line (R1-Q1: A-full-capstone):** proposal + data pipeline + trained model + eval + demo + report/blog. Nothing less.
- **Team (R1-Q3: A-solo-builder):** user builds everything via the agent; teammates exist on paper for the proposal only and do nothing.
- **Autonomy (R1-Q4: A-full-control):** agent installs, creates, runs; user is pinged only for taste + key decisions. Never spends money without asking.
- **Collaboration surface:** Lavish HTML pages, one fresh page per grilling round; plain chat otherwise. User types skill invocations by hand.
- **Proposal logistics (R3-Q15: A-tbd-placeholders):** names/dates stay TBD placeholders until the user fills them pre-submission.

## Technical decisions (locked)

- **Dataset (R2-Q7: A-lock-5k):** `joshuachin/openphoto-restore-dataset` — 5k pairs (4.5k train / 500 test), damaged→pristine, synthetic degradations, CC-BY-4.0. Public, no auth needed.
- **Demo (R2-Q8: A-notebook-plus-gradio):** Kaggle training notebook + exported weights + hosted Gradio demo (upload damaged → see restored).
- **Metrics (R2-Q9: A-full-stack):** PSNR + SSIM + LPIPS + side-by-side visuals + a classical baseline (inpainting/median) to beat.
- **Modeling (R3-Q12 → R4-Q18: A-finetune-different-task):** backbone pretrained on a *different* task (e.g. ImageNet / generic denoising/SR) + our restoration training on the 5k pairs + classical baseline. NOT a model already trained for old-photo restoration. Yes — this is the same as the original Round 1 option A, clarified.
- **Contribution (R4-Q19: A-cheap-practical):** "Cheap + practical: SOTA-informed fine-tune on paired synthetic data with a reproducible smoke→full pipeline, classical baseline, and a live Gradio demo."
- **Lit review cast (R3-Q13: A-full-cast):** Bringing Old Photos Back to Life + GFP-GAN/CodeFormer + Real-ESRGAN + classical inpainting baseline.
- **Training budget (R3-Q14: A-smoke-then-full):** smoke run (~200 pairs) to green the pipeline, then full 4.5k train.
- **Kaggle (R4-Q16: A-proceed):** token verified working (`kernels list --mine` returns user's kernels), stored in `~/.kaggle/access_token` + `KAGGLE_API_TOKEN`. Agent drives all GPU work from this terminal. User disables the token after the capstone.
- **HuggingFace (R4-Q17: A-defer):** token 401s but nothing needs it (public dataset, Kaggle-hosted training). Deferred; revisit Hub push at the end. All pasted tokens treated as burned.

## Build phase — Round B1 locks (zero-base)

- **Pipeline (B1-BQ1: A-detect-then-restore):** detector → masked restorer → composite output. Decision delegated to agent; verified against grading rubric (see below) — A wins on theoretical/technical mastery story, diagnosable failures, demo-able mask, and blog-ready results discussion.
- **Masks (B1-BQ2: A-synthetic-masks):** our own damage simulator gives exact masks by construction.
- **Code shape (B1-BQ3: A-src-package):** shared `src/restore/` package + thin notebook (clone + import) + local CPU pytest smoke.
- **Tickets (B1-BQ4: A-github-issues):** GitHub repo + `gh` Issues. Human step: user creates the repo + runs `gh auth login` once.
- **Build order (B1-BQ5: A-data-first):** loader + manifest + eval harness + classical baseline green on CPU before any network trains.

### Why A grades best (re-read docs per BQ1 note)

Boarding Kit grades: weekly progress (amount/quality/initiative), two live presentations (theory + technique mastery, handling questions), written blog (problem → prior work → approach with technical details → setup → results/discussion → industry implications → future work). Detect-then-restore serves every line: two trainable components = visible weekly progress; decomposition = theory story; mask vs fill separation = handles hard questions; mask figure + composite = demo and blog visuals; classical baseline comparison = results/discussion. Single end-to-end (B) is simpler but fuses failures into one undiagnosable box and weakens the theory/demo/blog lines.

## Build phase — Round B2 locks (data + training)

- **Data source (B2-BQ6: A-stream-hf):** stream from HuggingFace inside the kernel; mirror to Kaggle only if downloads slow a run.
- **Splits (B2-BQ7: A-fixed-plus-val):** keep shipped 4.5k/500; carve validation (~300) out of the 4.5k by fixed content-hash rule, recorded in a manifest.
- **Simulator (B2-BQ8: A-tiered):** core tier first (scratches, dust/specks, fading, sepia/cast, blur, noise, JPEG); extended tier (tears, creases, vignette) only if smoke stays green.
- **Scale (B2-BQ9: A-crops-plus-tiling):** train on 256px crops; serve full images via tiled inference with overlap blending.
- **Gates (B2-BQ10: C-both-gates):** pipeline green first (load → train → score → figures → weights download), then metric bar (beat identity AND classical baseline on PSNR/SSIM over val slice), then full training.

Spikes: S2 — no official Kaggle dataset mirror (only a third-party subset), so HF streaming is the path. S3 — this machine has zero ML libs; local CPU smoke needs installs first.

## Build phase — Round B3 locks (eval + demo + docs)

- **Scoreboards (B3-BQ11: A-three-boards):** (1) val during training for selection; (2) held-out test vs identity + classical rows; (3) ablations — true-mask vs detected-mask, core vs extended sim tier.
- **LPIPS (B3-BQ12: B-reporting-only):** PSNR/SSIM drive selection and gates; LPIPS computed once on final test outputs for blog/demo. Keeps the R2-Q9 lock without taxing training loops.
- **Demo (B3-BQ13: A-upload-plus-examples):** upload → damaged / damage-mask / restored side-by-side + note; 4 built-in example pairs; hosted on HuggingFace Spaces (free).
- **Real photos (B3-BQ14: A-qualitative-set):** ~10 public-domain archival scans (e.g. Library of Congress), eyes-only before/after in demo + blog, labeled "no reference — qualitative". Never training data.
- **Docs (B3-BQ15: A-proposal-now):** proposal draft now from locked decisions; presentations/blog after results exist.

## Build phase — Round B4 locks (final frontier)

- **Backbone (B4-BQ16: A-resnet18-imagenet):** ResNet-18 ImageNet encoder inside our U-Net, fine-tuned end-to-end (torchvision verified, spike S6).
- **Detector loss (B4-BQ17: A-weighted-bce-dice):** weighted BCE + Dice; select on val F1/IoU, recall-favored.
- **Failure policy (B4-BQ18: A-two-fix-loops):** two fix loops max (diagnose → adjust sim/hparams → re-smoke), then escalate with evidence before full-train spend.
- **Repo (B4-BQ19: agent-creates-private):** agent installs `gh`, asks user to auth if needed, creates a private repo, files all tickets as GitHub Issues.

## Open items (owned by user, needed before submission)

- Teammate names + real proposal/final deadlines (fill TBD placeholders).
- Fresh HF token only if we decide on a Hub push at the end.
- Disable all pasted tokens after the capstone.
