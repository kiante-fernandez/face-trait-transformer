"""Partial end-to-end fine-tune of a DINOv2 backbone on the OMI 34-attribute task.

Unfreezes the last N transformer blocks + final norm and trains them jointly with
an MLP head. Intended for CPU-only clusters: batch size 16, gradient accumulation
optional, heavy use of torch.set_num_threads. Slow, run overnight.

Example:
    python -m scripts.finetune \
        --images-dir $SCRATCH/omi/images \
        --labels $SCRATCH/omi/attribute_means.csv \
        --splits $SCRATCH/omi/splits.json \
        --model dinov2_vitl14 --unfreeze-blocks 1 \
        --hidden 1024 --dropout 0.2 --lr 1e-4 --wd 1e-4 \
        --epochs 30 --batch-size 16 --patience 6 \
        --out $SCRATCH/omi/checkpoints/finetune_vitl_lb1.pt
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from face_trait_transformer.data import load_labels, load_splits
from face_trait_transformer.features import build_transform, pick_device
from face_trait_transformer.metrics import per_attribute_pearson
from face_trait_transformer.model import TraitHead


class LabeledImageSet(Dataset):
    def __init__(self, paths: list[Path], labels: np.ndarray, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), torch.from_numpy(self.labels[i]).float()


def _parse_id(p: Path) -> int:
    m = re.fullmatch(r"(\d+)", p.stem)
    if m is None:
        raise ValueError(f"unexpected filename: {p}")
    return int(m.group(1))


def _freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze_last_blocks(backbone: nn.Module, n_blocks: int) -> list[nn.Parameter]:
    """Unfreeze the last n_blocks of the ViT + its final norm. Return trainable params."""
    if not hasattr(backbone, "blocks"):
        raise AttributeError("backbone has no .blocks; unsupported model type")
    trainable: list[nn.Parameter] = []
    for blk in backbone.blocks[-n_blocks:]:
        for p in blk.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


class BackbonePlusHead(nn.Module):
    def __init__(self, backbone: nn.Module, head: TraitHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--model", default="dinov2_vitl14")
    ap.add_argument("--unfreeze-blocks", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=None, help="separate LR for head (default = --lr * 10)")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")
    if device.type == "cpu":
        # Tune threading for the requested cores (set OMP_NUM_THREADS in the job script)
        print(f"torch threads: {torch.get_num_threads()}")

    # Data
    ids, Y, attr_names = load_labels(args.labels)
    splits = load_splits(args.splits)
    id_to_row = {int(i): k for k, i in enumerate(ids)}

    def paths_and_labels(split_ids):
        paths = []
        labels = []
        for sid in split_ids:
            p = args.images_dir / f"{int(sid)}.jpg"
            if not p.exists():
                raise FileNotFoundError(p)
            paths.append(p)
            labels.append(Y[id_to_row[int(sid)]])
        return paths, np.stack(labels)

    train_paths, Y_train = paths_and_labels(splits["train"])
    val_paths, Y_val = paths_and_labels(splits["val"])

    transform = build_transform()
    train_ds = LabeledImageSet(train_paths, Y_train, transform)
    val_ds = LabeledImageSet(val_paths, Y_val, transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    # Model
    print(f"Loading {args.model}...")
    backbone = torch.hub.load("facebookresearch/dinov2", args.model)
    _freeze_all(backbone)
    bb_trainable = _unfreeze_last_blocks(backbone, args.unfreeze_blocks)
    in_dim = backbone.norm.weight.shape[0]
    head = TraitHead(in_dim=in_dim, out_dim=Y.shape[1], hidden=args.hidden, dropout=args.dropout)
    model = BackbonePlusHead(backbone, head).to(device)

    n_bb = sum(p.numel() for p in bb_trainable)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"Trainable params: backbone={n_bb:,}  head={n_head:,}  (unfreeze={args.unfreeze_blocks} blocks)")

    head_lr = args.head_lr if args.head_lr is not None else args.lr * 10.0
    optimizer = torch.optim.AdamW(
        [
            {"params": bb_trainable, "lr": args.lr},
            {"params": head.parameters(), "lr": head_lr},
        ],
        weight_decay=args.wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    loss_fn = nn.MSELoss()

    best_val_r = -math.inf
    best_state = None
    epochs_no_improve = 0
    log_lines: list[str] = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tl = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tl.append(loss.item())
        train_loss = float(np.mean(tl))

        # Validate
        model.eval()
        vl = []
        preds, targs = [], []
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vl.append(loss_fn(pred, yb).item())
                preds.append(pred.cpu().numpy())
                targs.append(yb.cpu().numpy())
        val_loss = float(np.mean(vl))
        preds = np.concatenate(preds)
        targs = np.concatenate(targs)
        val_rs, _ = per_attribute_pearson(targs, preds)
        val_mean_r = float(np.mean(val_rs))
        scheduler.step()

        line = (
            f"epoch {epoch:3d}/{args.epochs} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_mean_r={val_mean_r:.4f}"
        )
        print(line, flush=True)
        log_lines.append(line)

        if val_mean_r > best_val_r + 1e-6:
            best_val_r = val_mean_r
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stop at epoch {epoch} (best val_mean_r={best_val_r:.4f})")
                break

    assert best_state is not None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "config": {
            "backbone": args.model,
            "unfreeze_blocks": args.unfreeze_blocks,
            "in_dim": int(in_dim),
            "out_dim": int(Y.shape[1]),
            "hidden": args.hidden,
            "dropout": args.dropout,
            "head": "mlp",
            "attr_names": attr_names,
            "seed": args.seed,
            "best_val_mean_r": best_val_r,
            "finetune": True,
        },
    }
    torch.save(ckpt, args.out)
    args.out.with_suffix(".log").write_text("\n".join(log_lines) + "\n")
    args.out.with_suffix(".meta.json").write_text(json.dumps(ckpt["config"], indent=2))
    print(f"Saved checkpoint -> {args.out}  (best val mean r = {best_val_r:.4f})")


if __name__ == "__main__":
    main()
