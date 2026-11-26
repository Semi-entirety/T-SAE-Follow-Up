#!/usr/bin/env python
"""Groupwise segmentation evaluation: Full/G0/G1 for one block/seed pair."""
import argparse, math, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path("/n/home12/binxuwang/T-SAE-Spatial")
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "temporal-saes" / "dictionary_learning")]
import arm_pipeline
arm_pipeline.DINO_REPO = os.environ.get("DINO_REPO", "/n/holylabs/kempner_alvarez_lab/Lab/binxuwang/Caches/torch/hub/facebookresearch_dinov2_main")
torch.hub.set_dir(os.environ.get("TORCH_HOME", "/n/holylabs/kempner_alvarez_lab/Lab/binxuwang/Caches/torch") + "/hub")
from arm_pipeline import pipeline_from_ckpt, assert_same_pipeline
from dictionary_learning.trainers.temporal_sequence_top_k import TemporalMatryoshkaBatchTopKSAE
from segment_alignment import build_instance_id_map, feature_metrics, label_map_to_patch_labels


def load_sae(ckpt, device):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "ae_state_dict" in sd: sd = sd["ae_state_dict"]
    ad, ds = sd["W_enc"].shape
    gs = [int(x) for x in sd["group_sizes"].tolist()]
    ae = TemporalMatryoshkaBatchTopKSAE(activation_dim=ad, dict_size=ds,
        k=int(sd["k"].item()), group_sizes=gs, temporal=False)
    ae.load_state_dict(sd, strict=False)
    return ae.eval().to(device), gs


def run_dirs(root, block, seed):
    if block == 2:
        return (root / f"dino_b2_postLN_spatial_s{seed}",
                root / f"dino_b2_postLN_baseline_s{seed}")
    if seed == 0:
        return root / f"spatial_block{block}", root / f"spatial_block{block}_baseline"
    return root / f"block{block}_spatial_s{seed}", root / f"block{block}_baseline_s{seed}"


def semantic_patch_labels(ex, image_size, side, purity=.6):
    objs = [(j, o["name"]) for j, o in enumerate(ex.get("objects", []))
            if o.get("parts", {}).get("part_level", 0) == 0]
    if not objs: return None
    names = sorted({n for _, n in objs}); ni = {n:i for i,n in enumerate(names)}
    ps = image_size // side
    cov = np.zeros((len(names), side * side), dtype=np.float32)
    from PIL import Image
    for j, name in objs:
        if j >= len(ex.get("instances", [])): continue
        m = np.array(ex["instances"][j].resize((image_size, image_size), Image.Resampling.NEAREST)) > 0
        cov[ni[name]] += m.reshape(side, ps, side, ps).mean((1,3)).reshape(-1)
    best = cov.argmax(0); val = cov.max(0)
    return np.where(val >= purity, best, -1)


def propagate(z, labels, n_per_class, seed):
    valid = np.where(labels >= 0)[0]; classes = np.unique(labels[valid])
    if len(classes) < 2: return np.nan
    rng = np.random.default_rng(seed); chosen = []
    for c in classes:
        pool = valid[labels[valid] == c]
        take = min(n_per_class, max(1, len(pool)-1))
        chosen.extend(rng.choice(pool, take, replace=False).tolist())
    chosen = np.asarray(chosen); rest = np.setdiff1d(valid, chosen)
    if not len(rest): return np.nan
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    pred = labels[chosen][np.argmax(zn[rest] @ zn[chosen].T, axis=1)]
    return float(np.mean(pred == labels[rest]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_seg", type=int, default=500); ap.add_argument("--max_prop", type=int, default=300)
    ap.add_argument("--store", default=os.environ.get("SAE_STORE", "/n/holylabs/kempner_alvarez_lab/Lab/binxuwang/Projects/T-SAE-Spatial"))
    ap.add_argument("--dataset_cache", default=os.environ.get("HF_DATASETS_CACHE", "/n/holylabs/kempner_alvarez_lab/Lab/binxuwang/Caches/huggingface/datasets"))
    ap.add_argument("--outdir", required=True); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    root = Path(a.store)
    sd, bd = run_dirs(root, a.block, a.seed)
    pipe = pipeline_from_ckpt(sd, a.device); bp = pipeline_from_ckpt(bd, a.device)
    assert_same_pipeline(pipe, bp)
    sa, gs = load_sae(sd/"ae_latest.pt", a.device); ba, bgs = load_sae(bd/"ae_latest.pt", a.device)
    assert gs == bgs and sum(gs) == 16384
    g0 = gs[0]; groups = {"Full": slice(None), "G0": slice(0,g0), "G1": slice(g0,None)}

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    ds = load_dataset("1aurent/ADE20K", split="validation", cache_dir=a.dataset_cache)
    seg_rows=[]; prop_rows=[]; nseg=nprop=0
    with torch.no_grad():
        for i, ex in enumerate(ds):
            if nseg >= a.max_seg and nprop >= a.max_prop: break
            try: im = ex["image"].convert("RGB"); tok = pipe.tokens(im)
            except Exception: continue
            side = int(round(math.sqrt(tok.shape[0]))); codes={"Spatial":sa.encode(tok,use_threshold=False), "Baseline":ba.encode(tok,use_threshold=False)}

            if nseg < a.max_seg:
                inst = build_instance_id_map(ex.get("instances",[]), ex.get("objects",[]), image_size=pipe.image_size, mask_rule="positive")
                if inst is not None:
                    labs, valid, _ = label_map_to_patch_labels(inst, image_size=pipe.image_size,
                        patch_size=pipe.image_size//side, purity_threshold=.6)
                    trial=[]; ok=True
                    for arm,z in codes.items():
                        for group,sl in groups.items():
                            m=feature_metrics(z[:,sl],labs,valid,precision_k=5)
                            if m is None: ok=False; break
                            trial.append(dict(block=a.block,seed=a.seed,image=i,arm=arm,group=group,
                                gap=m["gap"],same_sim=m["same_sim"],diff_sim=m["diff_sim"],prec5=m["precision_at_5"]))
                    if ok: seg_rows.extend(trial); nseg += 1

            if nprop < a.max_prop:
                labs = semantic_patch_labels(ex, pipe.image_size, side, .6)
                if labs is not None and (labs>=0).sum()>=16 and len(np.unique(labs[labs>=0]))>=2:
                    trial=[]
                    for arm,z in codes.items():
                        zn=z.float().cpu().numpy()
                        for group,sl in groups.items():
                            rec=dict(block=a.block,seed=a.seed,image=i,arm=arm,group=group)
                            for N in (1,2,4,8): rec[f"prop_N{N}"]=propagate(zn[:,sl],labs,N,1000+i)
                            trial.append(rec)
                    prop_rows.extend(trial); nprop += 1
            if (nseg+nprop)%100==0: print(f"seg={nseg}/{a.max_seg} prop={nprop}/{a.max_prop}",flush=True)

    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    sg=pd.DataFrame(seg_rows); pr=pd.DataFrame(prop_rows)
    sg.to_csv(out/"segment_raw.csv",index=False); pr.to_csv(out/"propagation_raw.csv",index=False)
    sg.groupby(["block","seed","arm","group"])[["gap","prec5","same_sim","diff_sim"]].mean().reset_index().to_csv(out/"segment_summary.csv",index=False)
    pr.groupby(["block","seed","arm","group"])[[f"prop_N{x}" for x in (1,2,4,8)]].mean().reset_index().to_csv(out/"propagation_summary.csv",index=False)
    print(f"DONE block={a.block} seed={a.seed} seg={nseg} prop={nprop} out={out}",flush=True)

if __name__ == "__main__": main()
