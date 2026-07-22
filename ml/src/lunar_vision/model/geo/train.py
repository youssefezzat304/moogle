
import argparse
import csv
import os
import time
from pathlib import Path
 
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Normalize

 
from lunarGeoData import LunarGeoData
from model import Geo2Geo
 
 
# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Geomap Reconstruction Autoencoder")
 
    # Data
    p.add_argument("--root",         type=str, default="/home/pg2026/data",
                   help="Root dir passed to LunarGeoData (contains UnifiedGeoMap/ and wac/)")
    p.add_argument("--patch_size",   type=int, default=256,
                   help="Patch size used by LunarGeoData (default 256)")
    p.add_argument("--stride",       type=int, default=128,
                   help="Stride used by LunarGeoData (default 128, gives overlapping patches)")
 
    # Model
    p.add_argument("--enc_patch",    type=int, default=16,
                   help="ViT token size inside each 256-px tile (default 16)")
    p.add_argument("--hidden_dim",   type=int, default=512)
    p.add_argument("--nheads",       type=int, default=8)
    p.add_argument("--num_layers",   type=int, default=6)
 
    # Training
    p.add_argument("--epochs",       type=int,   default=75)
    p.add_argument("--batch_size",   type=int,   default=24)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs",type=int,   default=5)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--train_frac",   type=float, default=0.70)
    p.add_argument("--val_frac",     type=float, default=0.15)
    p.add_argument("--seed",         type=int,   default=42)
 
    # Saving
    p.add_argument("--out_dir",      type=str, default="./runs/lunar_ae")
    p.add_argument("--save_every",   type=int, default=10,
                   help="Save a checkpoint every N epochs")
    p.add_argument("--vis_every",    type=int, default=5,
                   help="Save a reconstruction grid every N epochs")
    p.add_argument("--resume",       type=str, default=None,
                   help="Path to a checkpoint to resume from")
 
    return p.parse_args()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# Maps uint8 RGB [0, 255]  ←→  float32 [-1, 1]
# ──────────────────────────────────────────────────────────────────────────────
def to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 (3, H, W) → float32 (3, H, W) in [-1, 1]."""
    return x.float() / 127.5 - 1.0
 
 
def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """float32 [-1,1] → uint8 [0,255]  (for saving images)."""
    return ((x.clamp(-1, 1) + 1.0) * 127.5).byte()
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Collate: pull geomap["original"] and normalise in one place
# ──────────────────────────────────────────────────────────────────────────────
def geomap_collate(batch):
    """
    Each item in `batch` is a dict from LunarGeoData.__getitem__:
        item["geomap"]["original"]  →  (3, H, W)  uint8 tensor
 
    Returns:
        x  (B, 3, H, W) float32 in [-1, 1]  – both input AND target
    """
    patches = torch.stack([item["geomap"]["original"] for item in batch])  # (B,3,H,W) uint8
    x = to_float(patches)
    return x   # input == target for an autoencoder
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────
class ReconstructionLoss(nn.Module):
    """
    0.7 × MSE  +  0.3 × L1
 
    MSE drives the coarse structure; L1 preserves sharper colour boundaries
    which matter a lot for discrete geological class colours.
    """
    def __init__(self, w_mse: float = 0.7, w_l1: float = 0.3):
        super().__init__()
        self.mse  = nn.MSELoss()
        self.l1   = nn.L1Loss()
        self.w_mse, self.w_l1 = w_mse, w_l1
 
    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w_mse * self.mse(recon, target) + self.w_l1 * self.l1(recon, target)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# LR schedule: linear warm-up → cosine decay
# ──────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# One epoch (train or eval)
# ──────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
 
    with ctx:
        for x in loader:
            # x is the geomap float tensor (B, 3, H, W) from our collate_fn
            x = x.to(device, non_blocking=True)
 
            with autocast(enabled=device.type == "cuda"):
                recon = model(x)           # (B, 3, H, W)
                loss  = criterion(recon, x)   # target == input
 
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
 
            total_loss += loss.item()
 
    return total_loss / max(len(loader), 1)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Visualisation: side-by-side original | reconstruction grid
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def save_grid(model, loader, device, path: str, n: int = 8):
    model.eval()
    x = next(iter(loader))[:n].to(device)   # (n, 3, H, W)
    recon = model(x)
 
    # Denormalise [-1,1] → [0,1] for torchvision
    x_vis     = (x.cpu()     * 0.5 + 0.5).clamp(0, 1)
    recon_vis = (recon.cpu() * 0.5 + 0.5).clamp(0, 1)
 
    # Interleave: orig, recon, orig, recon, …
    pairs = torch.stack([x_vis, recon_vis], dim=1).reshape(-1, *x_vis.shape[1:])
    vutils.save_image(pairs, path, nrow=4, padding=2)
    print(f"     Reconstruction grid → {path}")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
 
    out = Path(args.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "visuals").mkdir(parents=True, exist_ok=True)
 
    # ── Device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"{'='*55}\n")
 
    # ── Dataset ───────────────────────────────────────────────────────
    # LunarGeoData loads the full mosaic once; patches are sliced on the fly.
    # We do NOT pass a transform here because normalisation happens in the
    # collate_fn so it applies to geomap["original"] (uint8 RGB).
    t = {'wac': Normalize(mean=[0.2523528337], std=[0.1592932492])}
    print("Loading LunarGeoData …")
    dataset = LunarGeoData(
        root=args.root,
        patch_size=args.patch_size,
        stride=args.stride,
        transform=t
    )
    print(f"  Total patches  : {len(dataset)}")
    print(f"  Geological classes: {dataset.num_classes}\n")
 
    # ── Train / val / test split ──────────────────────────────────────
    n       = len(dataset)
    n_train = int(n * args.train_frac)
    n_val   = int(n * args.val_frac)
    n_test  = n - n_train - n_val
 
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Split → train={n_train}, val={n_val}, test={n_test}\n")
 
    pin = device.type == "cuda"
    loader_kw = dict(
        collate_fn=geomap_collate,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
 
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False, **loader_kw)
 
    # ── Model ─────────────────────────────────────────────────────────
    model =     Geo2Geo(
        patch_size=args.enc_patch,    # ViT token size (16 px inside the 256-px tile)
        image_size=args.patch_size,   # each geomap tile is 256×256
        img_channels=3,               # RGB geomap
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_layers=args.num_layers,
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params   : {n_params:,}\n")
 
    # ── Optimiser / schedule / loss ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    criterion = ReconstructionLoss().to(device)
    scaler    = GradScaler(enabled=device.type == "cuda")
 
    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
 
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"  Resumed from epoch {ckpt['epoch']}  (best_val={best_val:.6f})\n")
 
    # ── CSV log ───────────────────────────────────────────────────────
    log_path = out / "training_log.csv"
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if start_epoch == 0:
        log_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "time_s"])
 
    # ──────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────
    print(f"Starting training  [{start_epoch+1} → {args.epochs}]\n")
 
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
 
        train_loss = run_epoch(model, train_loader, criterion,
                               optimizer, scaler, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion,
                               None,      scaler, device, train=False)
 
        scheduler.step()
        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
 
        log_writer.writerow([epoch+1, f"{train_loss:.6f}",
                              f"{val_loss:.6f}", f"{lr:.2e}", f"{elapsed:.1f}"])
        log_file.flush()
 
        print(f"  [{epoch+1:4d}/{args.epochs}]  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={lr:.2e}  {elapsed:.1f}s")
 
        # ── Best checkpoint ───────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_val":   best_val,
                "args":       vars(args),
            }, out / "checkpoints" / "best.pt")
            print(f"    ✔  New best model  (val={best_val:.6f})")
 
        # ── Periodic checkpoint ───────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val":  best_val,
                "args":      vars(args),
            }, out / "checkpoints" / f"epoch_{epoch+1:04d}.pt")
 
        # ── Reconstruction grid ───────────────────────────────────────
        if (epoch + 1) % args.vis_every == 0:
            save_grid(model, val_loader, device,
                      str(out / "visuals" / f"recon_epoch_{epoch+1:04d}.png"))
 
    log_file.close()
 
    # ── Final test evaluation ─────────────────────────────────────────
    print("\nEvaluating best model on test set …")
    ckpt = torch.load(out / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
 
    test_loss = run_epoch(model, test_loader, criterion,
                          None, scaler, device, train=False)
    print(f"  Test loss : {test_loss:.6f}")
 
    save_grid(model, test_loader, device,
              str(out / "visuals" / "recon_final_test.png"))
 
    print(f"\nAll outputs saved to: {out.resolve()}\n")
 
 
if __name__ == "__main__":
    main()
 
