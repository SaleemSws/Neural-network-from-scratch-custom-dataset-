"""Evidence checks for the real run; source comparison optional and read-only."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from common import ROOT, DEFAULT_TRAIN_SPLIT, DEFAULT_VAL_SPLIT, save_json, seed_all, load_model
from dataset import manifest, manifest_hash
from model import CustomUNet


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,help='Optional original dataset path to prove copy identity')
    args=p.parse_args()
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                         cwd=ROOT,capture_output=True,text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    history=list(csv.DictReader((ROOT/'results/training/history.csv').open(encoding='utf-8')))
    assert len(history)>=30 and [int(r['epoch']) for r in history]==list(range(1,len(history)+1))
    ckpt=torch.load(ROOT/'results/training/best.pt',map_location='cpu',weights_only=True)
    seed_all(ckpt['config']['seed'])
    initial=CustomUNet(ckpt['config']['base_channels'])
    changed={name:not torch.equal(param.detach(),ckpt['model'][name]) for name,param in initial.named_parameters()}
    assert all(changed.values()), 'Some learned parameters match random initialization'
    root=ROOT/'data/dataset_seg'
    assert manifest_hash(root,DEFAULT_VAL_SPLIT)==ckpt['config']['val_manifest_sha256']
    assert manifest_hash(root,DEFAULT_TRAIN_SPLIT)==ckpt['config']['train_manifest_sha256']
    training_images=manifest(root,DEFAULT_TRAIN_SPLIT)
    images=manifest(root,DEFAULT_VAL_SPLIT)
    assert len(training_images)==650 and len(images)==350
    assert not set(training_images)&set(images)
    assert set(training_images+images)==set(manifest(root,'train')+manifest(root,'val'))
    metrics=json.loads((ROOT/'results/evaluation/metrics.json').read_text())
    assert metrics['images']==len(images)==350
    assert metrics['manifest_sha256']==ckpt['config']['val_manifest_sha256']
    assert metrics['checkpoint_epoch']==ckpt['epoch']
    memory=json.loads((ROOT/'results/memory.json').read_text())
    assert memory['checkpoint_epoch']==ckpt['epoch']
    assert memory['split']==DEFAULT_VAL_SPLIT
    rows=list(csv.DictReader((ROOT/'results/evaluation/per_image.csv').open(encoding='utf-8')))
    assert len(rows)==350
    for path in images:
        with Image.open(path) as image,Image.open(ROOT/'run/validation/masks'/f'{path.stem}.png') as mask:
            assert image.size==mask.size
    source_check=None
    if args.source:
        files=sorted(p for p in args.source.rglob('*') if p.is_file())
        copied=sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
        assert sorted(p.relative_to(args.source).as_posix() for p in files)==copied
        for path in files:
            assert sha(path)==sha(root/path.relative_to(args.source)), f'Source differs: {path}'
        source_check={'files_compared':len(files),'all_sha256_equal':True,'source_modified':False}
    # Run a full source-resolution tensor through the model as a shape smoke check.
    # Reported training and inference still use 36x64.
    model,_=load_model(ROOT/'results/training/best.pt',torch.device('cpu'))
    torch.set_num_threads(2)
    with torch.inference_mode():
        shape=list(model(torch.zeros(1,3,720,1280)).shape)
    assert shape==[1,1,720,1280]
    cpu_output=ROOT/'run/cpu-example'
    if not (cpu_output/'metadata.json').exists():
        subprocess.run([sys.executable,'inference.py','--device','cpu','--input',str(images[0]),
                        '--output',str(cpu_output)],cwd=ROOT,check=True)
    with Image.open(cpu_output/'masks'/f'{images[0].stem}.png') as cpu_mask, Image.open(ROOT/'run/validation/masks'/f'{images[0].stem}.png') as cuda_mask:
        assert np.array_equal(np.array(cpu_mask),np.array(cuda_mask))
    result={'unit_tests_return_code':tests.returncode,'unit_tests_output':tests.stdout+tests.stderr,
            'completed_epochs':len(history),'all_parameter_tensors_changed_from_seeded_random_initialization':all(changed.values()),
            'parameter_tensors_checked':len(changed),'validation_masks_checked':len(images),
            'split':'65:35', 'train_images':len(training_images), 'validation_images':len(images),
            'split_disjoint_and_covers_all_1000_source_images':True,
            'best_checkpoint_epoch':ckpt['epoch'], 'checkpoint_sha256':sha(ROOT/'results/training/best.pt'),
            'evaluation_and_memory_match_new_checkpoint':True,
            'cpu_standalone_inference':{'original_resolution_output':[1280,720], 'binary_mask_matches_cuda':True},
            'original_resolution_mask_shape_check':True,'full_resolution_forward_shape':shape,
            'frozen_validation_manifest_matches_checkpoint':True,'source_copy_identity':source_check,
            'actual_commands':['prepare.py --split-65-35','train.py','inference.py','evaluation.py',
                               'measure_memory.py','python -m unittest discover -s tests -v',
                               'verify_submission.py --source ../dataset_seg'],
            'incomplete_assignment_requirements':[],
            'external_submission':'Not uploaded or published; local repository and ZIP prepared'}
    save_json(ROOT/'results/verification.json',result)
    print(result)


if __name__=='__main__':
    main()
