"""Read-only audit of source annotations; optional independent manifest output."""
import argparse
from collections import Counter
from pathlib import Path
import re
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from common import ROOT, DEFAULT_TRAIN_SPLIT, DEFAULT_VAL_SPLIT, save_json
from dataset import class_names, manifest, label_path, polygons, lane_mask, manifest_hash


def audit(root, train_split=DEFAULT_TRAIN_SPLIT, val_split=DEFAULT_VAL_SPLIT):
    names = class_names(root)
    train, val = manifest(root, train_split), manifest(root, val_split)
    if set(train) & set(val):
        raise ValueError('Train/validation share image paths')
    counts, sizes = Counter(), Counter()
    empty, issues = [], []
    for image in train+val:
        rows = polygons(label_path(root, image), names)
        for cls, pts in rows:
            counts[cls] += 1
            poly = Polygon(pts)
            if not poly.is_valid:
                issues.append({'image': image.name, 'class': cls,
                               'reason': explain_validity(poly)})
        with Image.open(image) as im:
            sizes[f'{im.width}x{im.height}'] += 1
            if not np.array(lane_mask(root, image, im.size, names=names)).any():
                empty.append(image.name)
    frame = lambda p: int(re.search(r'frame_(\d+)', p.name).group(1))
    tr_frames = np.array([frame(p) for p in train])
    distances = [int(np.abs(tr_frames-frame(p)).min()) for p in val]
    return {'class_names': names, 'lane_class': 3, 'train_images': len(train),
            'validation_images': len(val), 'image_sizes': dict(sizes),
            'polygon_counts': dict(counts), 'empty_lane_images': empty,
            'geometrically_invalid_polygons': issues,
            'invalid_polygon_policy': 'Keep original vertices; Pillow polygon fill. Source unchanged. Report annotation defects.',
            'missing_or_structurally_malformed_annotations': 0,
            'train_split': str(train_split), 'validation_split': str(val_split),
            'train_manifest_sha256': manifest_hash(root, train_split),
            'validation_manifest_sha256': manifest_hash(root, val_split),
            'nearest_training_frame_gap': {'min': min(distances), 'median': float(np.median(distances)),
                                           'max': max(distances), 'within_1_frame': sum(x<=1 for x in distances)},
            'limitation': 'Frames from the same video with nearby train/validation frames; results are not independent test generalization.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data/dataset_seg')
    p.add_argument('--output', type=Path, default=ROOT/'results/dataset_audit.json')
    p.add_argument('--split-65-35', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-split', default=DEFAULT_TRAIN_SPLIT)
    p.add_argument('--val-split', default=DEFAULT_VAL_SPLIT)
    args = p.parse_args()
    if args.split_65_35:
        paths = sorted(manifest(args.data, 'train')+manifest(args.data, 'val'))
        indices = np.random.default_rng(args.seed).permutation(len(paths))
        boundary = int(.65*len(paths))
        dest = ROOT/'splits/65-35'
        dest.mkdir(parents=True, exist_ok=True)
        for name, selection in [('train', indices[:boundary]), ('val', indices[boundary:])]:
            # Paths stay relative to the dataset root. Source manifests are untouched.
            (dest/f'{name}.txt').write_text('\n'.join(paths[i].relative_to(args.data.resolve()).as_posix()
                                                    for i in selection)+'\n', encoding='utf-8')
        save_json(dest/'split.json', {'seed': args.seed, 'train': boundary, 'val': len(paths)-boundary})
    report = audit(args.data, args.train_split, args.val_split)
    save_json(args.output, report)
    print({k:v for k,v in report.items() if k not in ('geometrically_invalid_polygons', 'empty_lane_images')})
    print('Invalid geometries:', len(report['geometrically_invalid_polygons']))


if __name__ == '__main__':
    main()
