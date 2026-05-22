#!/bin/bash
set -e

CKPT="results/checkpoints_spatial_hl/ae_final.pt"
VALID_PARQUET="data/imagenet_data/valid-00000-of-00001-70d52db3c749a935.parquet"
DEVICE="cuda"

echo "=============================="
echo " Starting all analysis scripts"
echo " Checkpoint: $CKPT"
echo "=============================="

echo "[1/5] analyze_spatial_smoothness_lipschitz.py..."
python scripts/analyze_spatial_smoothness_lipschitz.py \
    --parquet $VALID_PARQUET \
    --ckpt $CKPT \
    --n_images 500 \
    --device $DEVICE \
    --outdir results/smoothness_lipschitz_hl

echo "[2/5] analyze_latent_statistics_v3.py..."
python scripts/analyze_latent_statistics_v3.py \
    --parquet $VALID_PARQUET \
    --ckpt $CKPT \
    --n_images 2000 \
    --n_latents 5 \
    --n_images_per_latent 10 \
    --device $DEVICE \
    --outdir results/latent_statistics_hl

echo "[3/5] analyze_discriminative_concepts.py..."
python scripts/analyze_discriminative_concepts.py \
    --parquet $VALID_PARQUET \
    --ckpt $CKPT \
    --n_classes 20 \
    --top_k 50 \
    --n_images_per_class 50 \
    --device $DEVICE \
    --outdir results/discriminative_hl

echo "[4/5] analyze_highlevel_vs_lowlevel.py..."
python scripts/analyze_highlevel_vs_lowlevel.py \
    --parquet $VALID_PARQUET \
    --ckpt $CKPT \
    --n_classes 20 \
    --n_images_per_class 50 \
    --device $DEVICE \
    --outdir results/highlevel_vs_lowlevel_hl

echo "[5/5] analyze_shared_concepts.py..."
python scripts/analyze_shared_concepts.py \
    --json results/discriminative_hl/discriminative_concepts.json \
    --parquet $VALID_PARQUET \
    --ckpt $CKPT \
    --min_classes 2 \
    --top_n_concepts 20 \
    --n_images_per_class 3 \
    --device $DEVICE \
    --outdir results/shared_concepts_hl

echo "=============================="
echo " All analysis complete!"
echo " Results saved to results/*_hl/"
echo "=============================="