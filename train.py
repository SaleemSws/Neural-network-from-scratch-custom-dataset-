import argparse
import csv
import platform
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from common import ROOT, DEFAULT_TRAIN_SPLIT, DEFAULT_VAL_SPLIT, save_json, seed_all, device_for
from dataset import LaneDataset, class_names, manifest_hash
from model import CustomUNet


def loss_parts(logits, truth):
    bce = F.binary_cross_entropy_with_logits(logits, truth)
    prob = logits.sigmoid()
    axes = (1, 2, 3)
    # Smooth=1 keeps loss finite for an empty ground-truth mask.
    dice = 1-((2*(prob*truth).sum(axes)+1)/(prob.sum(axes)+truth.sum(axes)+1)).mean()
    return .5*bce+.5*dice, bce, dice


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    totals = torch.zeros(4, device=device)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, truth in loader:
            images, truth = images.to(device), truth.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss, bce, dice = loss_parts(logits, truth)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            p, t = logits.detach().sigmoid()>.5, truth.bool()
            inter = (p&t).sum((1,2,3))
            union = (p|t).sum((1,2,3))
            iou = torch.where(union>0, inter/union.clamp_min(1), 1.).mean()
            n = images.shape[0]
            totals += torch.stack([loss.detach(), bce.detach(), dice.detach(), iou])*n
            count += n
    return dict(zip(['loss', 'bce', 'dice_loss', 'iou'], (totals/count).cpu().tolist()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data/dataset_seg')
    p.add_argument('--output', type=Path, default=ROOT/'results/training')
    p.add_argument('--train-split', default=DEFAULT_TRAIN_SPLIT)
    p.add_argument('--val-split', default=DEFAULT_VAL_SPLIT)
    p.add_argument('--height', type=int, default=36)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--base-channels', type=int, default=8)
    p.add_argument('--batch-size', type=int, choices=[2,3,4], default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if args.epochs < 30 or min(args.height, args.width)<8:
        p.error('Assignment requires >=30 epochs and dimensions >=8')
    if (args.output/'history.csv').exists():
        p.error('Output already contains training history; choose a new --output')
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update({'lane_class':3, 'class_names':class_names(args.data),
                   'train_manifest_sha256':manifest_hash(args.data, args.train_split),
                   'val_manifest_sha256':manifest_hash(args.data, args.val_split),
                   'loss':'0.5 BCEWithLogits + 0.5 soft Dice (smooth=1)',
                   'optimizer':'Adam', 'scheduler':'CosineAnnealingLR eta_min=1e-5',
                   'precision':'float32', 'pretrained':False})
    save_json(args.output/'config.json', config)
    start = time.perf_counter()
    print('Preparing resized in-memory dataset...', flush=True)
    train = LaneDataset(args.data, args.train_split, args.height, args.width, augment=True)
    val = LaneDataset(args.data, args.val_split, args.height, args.width)
    if set(train.paths)&set(val.paths):
        raise ValueError('Train/validation paths overlap')
    loaders = [DataLoader(x, batch_size=args.batch_size, shuffle=i==0, num_workers=0)
               for i,x in enumerate([train,val])]
    model = CustomUNet(args.base_channels).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    writer = SummaryWriter(str(args.output/'tensorboard'))
    writer.add_text('config', str(config))
    history, best, best_epoch = [], -1., 0
    training_start = time.perf_counter()
    print(f'Device={device}, parameters={parameters}, train={len(train)}, val={len(val)}', flush=True)
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            tr = run_epoch(model, loaders[0], device, optimizer)
            va = run_epoch(model, loaders[1], device)
            row = {'epoch':epoch, 'lr':lr, 'seconds':time.perf_counter()-epoch_start,
                   **{f'train_{k}':v for k,v in tr.items()}, **{f'val_{k}':v for k,v in va.items()}}
            history.append(row)
            for tag, value in row.items():
                if tag != 'epoch':
                    writer.add_scalar(tag, value, epoch)
            writer.flush()
            with (args.output/'history.csv').open('w', newline='', encoding='utf-8') as f:
                csvwriter = csv.DictWriter(f, fieldnames=list(row))
                csvwriter.writeheader()
                csvwriter.writerows(history)
            state = {'model':model.state_dict(), 'config':config, 'epoch':epoch,
                     'val_iou':va['iou'], 'parameter_count':parameters}
            if va['iou']>best:
                best, best_epoch = va['iou'], epoch
                torch.save(state, args.output/'best.pt')
            if epoch==args.epochs:
                torch.save(state, args.output/'last.pt')
            scheduler.step()
            print(f'Epoch {epoch:02}/{args.epochs}: loss {tr["loss"]:.4f}/{va["loss"]:.4f}; val IoU {va["iou"]:.4f}; {row["seconds"]:.1f}s', flush=True)
    finally:
        writer.close()
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Train (augmented)')
    ax.plot([r['epoch'] for r in history], [r['val_loss'] for r in history], label='Validation')
    ax.set(xlabel='Epoch', ylabel='0.5 BCE + 0.5 Dice loss', title='Custom U-Net trained from scratch')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.output/'loss.png', dpi=160); plt.close(fig)
    summary = {'completed_epochs':len(history), 'best_epoch':best_epoch, 'best_low_resolution_val_iou':best,
               'train_images':len(train), 'validation_images':len(val), 'parameter_count':parameters,
               'training_seconds':time.perf_counter()-training_start,
               'total_seconds_including_preparation':time.perf_counter()-start,
               'device':str(device), 'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
               'torch_version':str(torch.__version__), 'python':platform.python_version(),
               'first_epoch':history[0], 'last_epoch':history[-1]}
    save_json(args.output/'summary.json', summary)
    print(summary, flush=True)


if __name__=='__main__':
    main()
