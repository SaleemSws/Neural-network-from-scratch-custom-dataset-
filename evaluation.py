"""One union lane mask per image; image/mask AP at a stated IoU threshold."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, DEFAULT_VAL_SPLIT, save_json, iou_arrays
from dataset import manifest, class_names, lane_mask, manifest_hash


def mask_average_precision(rows, threshold=.5, inclusive=False):
    """All-point precision-envelope AP with one candidate per nonempty prediction.

    One GT object is the union lane mask in each GT-positive image. A candidate
    can match only the GT in its own image. Empty predictions create no candidate.
    Equal confidence candidates are evaluated together to avoid tie-order bias.
    """
    positives = sum(r['ground_truth_foreground'] for r in rows)
    candidates = sorted([r for r in rows if r['predicted_foreground']],
                        key=lambda r:r['confidence'], reverse=True)
    if positives==0:
        return {'ap':None, 'precision':[], 'recall':[], 'ground_truth_positives':0,
                'predictions':len(candidates), 'reason':'AP undefined: no positive ground truth'}
    precisions, recalls, tp, fp = [], [], 0, 0
    for i, row in enumerate(candidates):
        match = row['iou']>=threshold if inclusive else row['iou']>threshold
        is_tp = row['ground_truth_foreground'] and match
        tp += int(is_tp); fp += int(not is_tp)
        if i+1==len(candidates) or candidates[i+1]['confidence']!=row['confidence']:
            precisions.append(tp/(tp+fp)); recalls.append(tp/positives)
    r = np.array([0.]+recalls+[1.])
    p = np.array([0.]+precisions+[0.])
    p = np.maximum.accumulate(p[::-1])[::-1]
    ap = float(np.sum(np.diff(r)*p[1:]))
    return {'ap':ap, 'precision':precisions, 'recall':recalls,
            'ground_truth_positives':positives, 'predictions':len(candidates)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/dataset_seg')
    parser.add_argument('--split', default=DEFAULT_VAL_SPLIT)
    parser.add_argument('--predictions', type=Path, default=ROOT/'run/validation')
    parser.add_argument('--output', type=Path, default=ROOT/'results/evaluation')
    parser.add_argument('--iou-threshold', type=float, default=.5)
    args = parser.parse_args()
    if not 0<=args.iou_threshold<=1:
        parser.error('IoU threshold must be in [0,1]')
    metadata = json.loads((args.predictions/'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('manifest_sha256') != manifest_hash(args.data, args.split):
        raise ValueError('Predictions and evaluation manifest do not match')
    records = {r['stem']:r for r in metadata['records']}
    paths = manifest(args.data, args.split)
    if len({p.stem for p in paths})!=len(paths) or set(records)!={p.stem for p in paths}:
        raise ValueError('Prediction image set must exactly match evaluation split')
    names = class_names(args.data)
    rows, total_inter, total_union = [], 0, 0
    for image in paths:
        with Image.open(image) as im:
            truth = np.array(lane_mask(args.data, image, im.size, names=names))>0
        with Image.open(args.predictions/'masks'/f'{image.stem}.png') as im:
            values = np.array(im)
        if values.shape!=truth.shape or not np.isin(values,[0,255]).all():
            raise ValueError(f'{image.name}: binary mask shape/encoding mismatch')
        pred = values>0
        record = records[image.stem]
        # Check stored confidence against retained low-resolution probabilities.
        probability = np.load(args.predictions/'probabilities'/f'{image.stem}.npy')
        if probability.shape!=(metadata['input_height'],metadata['input_width']) or not np.isfinite(probability).all():
            raise ValueError('Invalid saved probabilities')
        low_fg = probability>=metadata['probability_threshold']
        conf = float(probability[low_fg].mean()) if low_fg.any() else 0.
        restored = np.array(Image.fromarray(low_fg.astype(np.uint8)*255).resize(
            (truth.shape[1],truth.shape[0]),Image.Resampling.NEAREST))>0
        if not np.array_equal(restored,pred) or abs(conf-record['confidence'])>1e-6:
            raise ValueError('Prediction mask/confidence and retained probabilities disagree')
        iou = iou_arrays(pred,truth)
        rows.append({'image':image.name, 'iou':iou, 'detected':iou>args.iou_threshold,
                     'ground_truth_foreground':bool(truth.any()), 'predicted_foreground':bool(pred.any()),
                     'confidence':conf})
        total_inter += int((pred&truth).sum()); total_union += int((pred|truth).sum())
    detected = [r['iou'] for r in rows if r['detected']]
    ap = mask_average_precision(rows,args.iou_threshold)
    variants = {}
    for label, threshold, inclusive in [('gt_0.5',.5,False),('gt_0.6',.6,False),('ge_0.6',.6,True)]:
        ds = [r['iou'] for r in rows if (r['iou']>=threshold if inclusive else r['iou']>threshold)]
        variants[label] = {'detection_rate':len(ds)/len(rows), 'detected_images':len(ds),
                           'mean_detected_iou':float(np.mean(ds)) if ds else None,
                           'mask_ap':mask_average_precision(rows,threshold,inclusive)['ap']}
    result = {'evaluated_split':args.split, 'images':len(rows), 'evaluation_resolution':'original image resolution',
              'mean_image_lane_iou':float(np.mean([r['iou'] for r in rows])),
              'aggregate_pixel_lane_iou':total_inter/total_union if total_union else 1.,
              'iou_threshold':args.iou_threshold, 'positive_rule':'strictly greater than threshold',
              'detection_rate':len(detected)/len(rows), 'detected_images':len(detected),
              'mean_iou_detected':float(np.mean(detected)) if detected else None,
              'mask_ap':ap['ap'], 'threshold_variants':variants,
              'empty_mask_policy':'IoU=1 if both empty, IoU=0 if exactly one empty. Empty prediction creates no AP candidate; empty GT creates no AP positive.',
              'ap_definition':'Image/mask-level single-class all-point precision-envelope AP. One union lane GT per positive image; one candidate per nonempty prediction. Confidence = mean predicted-foreground probability at model resolution. Candidates only match GT in the same image. Ties grouped. IoU rule as specified.',
              'pixel_ap_reported':False,
              'checkpoint_epoch':metadata['epoch'], 'probability_threshold':metadata['probability_threshold'],
              'manifest_sha256':metadata['manifest_sha256']}
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output/'metrics.json',result)
    save_json(args.output/'precision_recall.json',ap)
    with (args.output/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig,ax = plt.subplots(figsize=(6,4.5))
    ax.step([0.]+ap['recall'], [1.]+ap['precision'],where='post',label=f'Mask AP={ap["ap"]:.4f}' if ap['ap'] is not None else 'AP undefined')
    ax.set(xlabel='Recall (GT-positive images)',ylabel='Precision (mask candidates)',
           title=f'Single-class mask PR, IoU > {args.iou_threshold}',xlim=(0,1.01),ylim=(0,1.01))
    ax.grid(alpha=.25); ax.legend();fig.tight_layout()
    fig.savefig(args.output/'precision_recall.png',dpi=160);plt.close(fig)
    print(result,flush=True)


if __name__=='__main__':
    main()
