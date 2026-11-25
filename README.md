# T-SAE Follow-Up: Spatial SAE Study

Research code for studying spatially regularized sparse autoencoders (SAEs) on vision-model patch representations, with DINOv2 as the main experimental backbone. The study compares Spatial and reconstruction-only baseline SAEs through downstream tasks, direct spatial-smoothness measurements, reconstruction diagnostics, and controlled synthetic experiments.

This repository contains the contents of [`nhp_sae_analysis/spatial_sae_study`](https://github.com/Semi-entirety/nhp_sae_analysis/tree/dfff5deb2a9902df7948ec978124db6b60dbe138/spatial_sae_study), promoted to the repository root. The source snapshot is commit `dfff5deb2a9902df7948ec978124db6b60dbe138`. All 60 non-README files are preserved unchanged, including the figures and ZIP archive. This README replaces the source directory's README; the previous contents of T-SAE-Follow-Up remain available in Git history.

> **Reproducibility status:** this is a research-code snapshot, not a self-contained installable package. Several scripts depend on the original `T-SAE-Spatial` / `temporal-saes` codebase, trained checkpoints, datasets, and hard-coded cluster paths. Prepare those dependencies and adapt paths before running the examples below.

## What is studied?

- **Downstream behavior:** object-instance alignment, unsupervised patch segmentation, label-efficient propagation, and cross-view patch correspondence.
- **Spatial smoothness:** same-image code maps versus matched across-image mosaics, with cosine, Fourier, Haar-wavelet, multiscale, and normalized local-variation metrics.
- **Mechanisms and controls:** channel energy, PCA structure, smooth-channel allocation, reconstruction quality, whitening, and batch-negative controls.
- **Synthetic recovery:** whether a grouped SAE recovers known smooth and rough factors and allocates them to different latent groups.

`Full` refers to all latent dimensions; `G0` and `G1` refer to checkpoint-defined latent groups. Group sizes come from the checkpoint, and should not be assumed to be identical across experiments. The multiblock launchers commonly cover DINOv2 blocks **0, 2, 4, 6, 8, 10, and 11**. Check the extractor used by each training/evaluation script before transferring block indices between workflows.

## Repository map

| Location | Contents |
| --- | --- |
| [`training/`](training/) | Spatial SAE training entry points, an independent-group variant, and a grouped TopK trainer implementation. |
| [`downstream_tasks/`](downstream_tasks/) | Segmentation/alignment, label propagation, cross-view correspondence, shared backbone evaluation, and direct smoothness/contrastive controls. |
| [`mechanistic_analysis/`](mechanistic_analysis/) | Channel smoothness, energy shortcuts, PCA, FVU, whitening, image bootstrap, and multiblock analyses. |
| [`synthetic/`](synthetic/) | Smooth/rough factor generation, SAE training, factor-recovery analysis, aggregation, and toy shortcut controls. |
| [`utils/`](utils/) | Checkpoint-specific feature preprocessing and GPU/reproducibility checks. |
| [`figures/`](figures/) | Plotting scripts and **10 committed PNG figures** under `batch_random_effect/`, `contrastive_direct/`, and `smoothness_effects/`. |
| [`slurm/`](slurm/) | Ten original SLURM job scripts for cluster experiments. |
| `Spatial_SAEs (4).zip` | Archive copied unchanged from the source directory; not required by the command examples below. |

Useful entry points:

| Goal | Script |
| --- | --- |
| Train a DINOv2 patch-pair SAE | [`training/new_spatial_train.py`](training/new_spatial_train.py) |
| Train the multiscale/independent-group variant | [`training/train_multiscale_spatial_independent_subsae.py`](training/train_multiscale_spatial_independent_subsae.py) |
| Evaluate instance alignment | [`downstream_tasks/segment_alignment.py`](downstream_tasks/segment_alignment.py) |
| Compare segmentation and few-label propagation | [`downstream_tasks/spatial_downstream.py`](downstream_tasks/spatial_downstream.py) |
| Compare Full/G0/G1 downstream performance | [`downstream_tasks/spatial_downstream_groups.py`](downstream_tasks/spatial_downstream_groups.py) |
| Evaluate cross-view localization | [`downstream_tasks/two_view_correspondence.py`](downstream_tasks/two_view_correspondence.py) |
| Measure within-image smoothness against a mosaic null | [`downstream_tasks/code_smoothness_same_vs_across.py`](downstream_tasks/code_smoothness_same_vs_across.py) |
| Compare training-aligned positive/negative similarities | [`downstream_tasks/code_contrastive_direct_eval.py`](downstream_tasks/code_contrastive_direct_eval.py) |
| Test a batch-random negative control | [`downstream_tasks/code_smoothness_batch_random.py`](downstream_tasks/code_smoothness_batch_random.py) |
| Evaluate reconstruction | [`mechanistic_analysis/fvu_compare.py`](mechanistic_analysis/fvu_compare.py), [`mechanistic_analysis/eval_fullcode_fvu_multiblock.py`](mechanistic_analysis/eval_fullcode_fvu_multiblock.py) |
| Train and analyze synthetic recovery | [`synthetic/train_synth_spatial.py`](synthetic/train_synth_spatial.py), [`synthetic/analyze_synth_recovery.py`](synthetic/analyze_synth_recovery.py) |

## Environment and external dependencies

The source README identifies the original environment as `torch2`; this snapshot does not include a pinned environment or dependency lockfile. Principal third-party packages used by the scripts include PyTorch, torchvision, NumPy, pandas, SciPy, scikit-learn, Pillow, matplotlib, Hugging Face `datasets`, PyArrow, joblib, and einops. Backbone-specific dependencies such as timm may also be needed by the external feature extractors. Use a PyTorch/torchvision build compatible with your GPU environment.

The following project modules are imported but are **not supplied as a complete dependency tree here**:

- `dictionary_learning`, including `trainers.temporal_sequence_top_k` and, for synthetic training, `trainers.temporal_sequence_top_k_weight`;
- `new_vision_patch_pairs`;
- `new_spatial_train_triplet`;
- `channel_transform` and `prenorm_scale_check`;
- `synth_topk_gen` for the toy shortcut experiments.

Use the matching versions from the original experiment environment. The file `training/temporal_sequence_top_k.py` is a preserved trainer source file with package-relative imports; its presence does **not** install the `dictionary_learning` package or make it a standalone script.

After cloning, expose the reorganized local helpers and the external project modules on `PYTHONPATH`. The following is a Bash example; replace the external path with your own:

```bash
git clone https://github.com/Semi-entirety/T-SAE-Follow-Up.git
cd T-SAE-Follow-Up

STUDY_ROOT="$PWD"
EXTERNAL_ROOT=/absolute/path/to/T-SAE-Spatial
export PYTHONPATH="$STUDY_ROOT/downstream_tasks:$STUDY_ROOT/mechanistic_analysis:$STUDY_ROOT/utils:$STUDY_ROOT/synthetic:$EXTERNAL_ROOT/scripts:$EXTERNAL_ROOT/temporal-saes/dictionary_learning:${PYTHONPATH:-}"
```

This addresses cross-directory imports; it does not replace missing dependencies or fix hard-coded paths. Search the scripts for `/n/home12/`, `/n/holylfs06/`, and `/home/ubuntu/`, then update the relevant cache, data, checkpoint, and output locations. Some modules directly assign `TORCH_HOME` / `HF_HOME` and set the torch.hub directory at import time, so exporting environment variables alone is insufficient. Several scripts also prepend original project paths to `sys.path`; check those to avoid accidentally importing an older helper.

## Data and checkpoint preparation

Checkpoints and datasets are not included as standalone resources in this snapshot. Depending on the experiment, provide:

- DINOv2's local torch.hub source/cache and model weights;
- held-out image Parquet files, typically with an `image` column containing image bytes;
- ADE20K instance annotations for segmentation and propagation (`1aurent/ADE20K` is used by these scripts);
- matching Spatial/baseline checkpoints, commonly named `ae_latest.pt`;
- each checkpoint's adjacent `run_args.json` when using `utils/arm_pipeline.py`, together with any channel-transform statistics referenced by that metadata;
- `ground_truth.npz` and run metadata for synthetic-recovery analysis, produced by synthetic training.

Use held-out evaluation images and match the backbone, block, normalization, input scale, and image size to the checkpoint's training convention. Some evaluators use fixed post-LayerNorm DINOv2 features; `arm_pipeline.py` instead reconstructs a pipeline from checkpoint metadata. These paths are not interchangeable for arbitrary checkpoints.

## Example workflows

These examples show the checked-in command-line interfaces. They require the environment, external modules, and path changes described above; they were not executed end to end during the repository migration.

### Direct smoothness: Spatial versus baseline

```bash
python downstream_tasks/code_smoothness_same_vs_across.py \
  --ckpt spatial:/absolute/path/to/spatial/ae_latest.pt \
  --ckpt baseline:/absolute/path/to/baseline/ae_latest.pt \
  --dino_block 0 \
  --parquet /absolute/path/to/heldout_images.parquet \
  --n_images 200 --null_reps 10 --n_boot 5000 --n_perm 100000 \
  --device cuda \
  --outdir results/code_smoothness_b0
```

Main outputs:

| File | Meaning |
| --- | --- |
| `smoothness_same_vs_across_raw.csv` | Image-level same-image, across-image, and effect measurements. |
| `smoothness_same_vs_across_summary.csv` | Bootstrap summaries of the measurements. |
| `smoothness_spatial_vs_baseline_paired.csv` | Image-paired Spatial-minus-baseline comparisons. |
| `run_config.json` | Evaluation arguments. |

The mosaic null preserves patch position and latent marginal distributions while disrupting coherence within a single image. For cosine, Fourier, wavelet, and multiscale metrics, the effect is `same_image - across_image`; for Lipschitz/local variation it is reversed. Positive effects therefore indicate greater within-image smoothness under the chosen metric.

### Segmentation and label-efficient propagation

```bash
python downstream_tasks/spatial_downstream.py \
  --ckpt spatial:/absolute/path/to/spatial/ae_latest.pt \
  --ckpt baseline:/absolute/path/to/baseline/ae_latest.pt \
  --dino_block 0 --max_images 300 --n_pcs 32 \
  --seeds_grid 1 2 4 8 --device cuda \
  --outdir results/spatial_downstream_b0
```

This entry point evaluates G0 representations, reports clustering adjusted Rand index and nearest-neighbor propagation accuracy, and includes a matched-number-of-PCs control. The label budget is **per class**, not per image. Outputs are `spatial_downstream_raw.csv` and `spatial_downstream_summary.csv`. Use `spatial_downstream_groups.py` for Full/G0/G1 comparisons.

### Synthetic training and recovery

```bash
python synthetic/train_synth_spatial.py \
  --save_dir results/synth_spatial_s0 --seed 0 --data_seed 0 --device cuda

python synthetic/analyze_synth_recovery.py \
  --ckpts spatial=results/synth_spatial_s0/ae_latest.pt \
  --outdir results/synth_recovery --device cuda
```

Training saves the generated ground-truth dictionary beside the checkpoint. Recovery analysis writes `synth_recovery_summary.csv`, `synth_recovery_per_gt.csv`, `synth_recovery_per_latent.csv`, and a figure. `aggregate_synth_recovery.py` accepts summary and per-ground-truth CSVs for aggregation across runs. Keep data-generation settings matched when comparing Spatial and baseline arms.

## SLURM and figures

The launchers preserve their original paths and job settings. Before submitting any job, edit its account/partition, log destinations, Python executable, data/checkpoint/output paths, and script invocation paths. Variables include `HERE`, `ROOT`, `REPO`, `HOME_REPO`, `STORE`, `RES`, `OUT`, and `PY`, depending on the launcher. **Do not merely point `HERE` at this repository root:** old launchers invoke scripts from a flat directory, whereas this snapshot places them in `downstream_tasks/`, `mechanistic_analysis/`, or `synthetic/`. The whitening-training launcher additionally invokes an external training script.

For example, after adapting its paths, submit:

```bash
sbatch slurm/run_code_smoothness_multiblock.sbatch
```

Figure scripts read analysis CSVs and often contain their own fixed `ROOT`/output paths. Update those before regeneration. The committed PNGs are copied outputs, not evidence that the analysis was rerun in this repository; the associated raw result CSVs are not present as standalone files in the migrated directory.

## Interpretation and validation

- Pair Spatial/baseline measurements on matched images and seeds wherever the experiment supports it.
- Image-bootstrap confidence intervals quantify variation across images, not across independently trained SAEs. Label single-checkpoint multiblock results accordingly.
- Report same-image and null measurements alongside their contrast, and distinguish latent-code smoothness from decoded-group smoothness.
- FVU implementations differ across scripts (squared-error versus residual-variance conventions); compare values only under the same definition and preprocessing.
- The migration preserves experiment code rather than changing its algorithms or claiming new results. Python syntax and SLURM shell syntax were checked, and copied file hashes were compared with the source snapshot. Full training/evaluation requires the external resources described above.
