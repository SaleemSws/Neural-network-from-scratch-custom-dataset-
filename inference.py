import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import ROOT, DEFAULT_VAL_SPLIT, device_for, load_model, image_tensor, save_json
from dataset import manifest, manifest_hash


@torch.inference_mode()
def predict(model, image, config, device, threshold=.5):
    tensor = image_tensor(image, config['height'], config['width'])[None].to(device)
    prob = model(tensor).sigmoid()[0,0].cpu().numpy()
    low = Image.fromarray((prob>=threshold).astype(np.uint8)*255)
    mask = low.resize(image.size, Image.Resampling.NEAREST)
    foreground = prob>=threshold
    confidence = float(prob[foreground].mean()) if foreground.any() else 0.
    return mask, prob, confidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=ROOT/'results/training/best.pt')
    p.add_argument('--data', type=Path, default=ROOT/'data/dataset_seg')
    p.add_argument('--split', default=DEFAULT_VAL_SPLIT)
    p.add_argument('--input', type=Path, help='Image or directory; annotations not required')
    p.add_argument('--output', type=Path, default=ROOT/'run/validation')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = p.parse_args()
    if not 0<args.threshold<1:
        p.error('threshold must be between 0 and 1')
    torch.set_num_threads(2)
    device = device_for(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    if args.input:
        paths = sorted(x for x in args.input.rglob('*') if x.suffix.lower() in ('.jpg','.jpeg','.png')) if args.input.is_dir() else [args.input]
    else:
        paths = manifest(args.data, args.split)
    if not paths:
        p.error('No images found')
    if len({x.stem for x in paths}) != len(paths):
        p.error('Duplicate stems would overwrite outputs')
    if (args.output/'metadata.json').exists():
        p.error('Output already contains predictions; choose a new --output')
    for sub in ['masks','probabilities']:
        (args.output/sub).mkdir(parents=True, exist_ok=True)
    records = []
    for i,path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert('RGB')
            mask, prob, confidence = predict(model, image, ckpt['config'], device, args.threshold)
            mask.save(args.output/'masks'/f'{path.stem}.png')
            np.save(args.output/'probabilities'/f'{path.stem}.npy', prob)
            records.append({'image':path.name, 'stem':path.stem, 'original_size':list(image.size),
                            'confidence':confidence, 'predicted_foreground':bool(np.array(mask).any())})
        if (i+1)%50==0:
            print(f'Predicted {i+1}/{len(paths)}', flush=True)
    save_json(args.output/'metadata.json', {'checkpoint':args.checkpoint.name, 'epoch':ckpt['epoch'], 'device':str(device),
        'input_height':ckpt['config']['height'], 'input_width':ckpt['config']['width'],
        'probability_threshold':args.threshold, 'binary_mask_encoding':'0/255 PNG',
        'restoration':'nearest-neighbor binary mask resize to original resolution',
        'confidence_definition':'Mean foreground probability at model input resolution; zero for empty prediction',
        'split':None if args.input else args.split,
        'manifest_sha256':None if args.input else manifest_hash(args.data,args.split), 'records':records})
    print(f'Saved {len(records)} masks to {args.output}', flush=True)


if __name__=='__main__':
    main()
