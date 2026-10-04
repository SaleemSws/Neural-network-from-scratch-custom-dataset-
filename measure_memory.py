"""Fresh-process RAM sampling and CUDA allocator peak during full inference."""
import argparse
import gc
import platform
import threading
import time
from pathlib import Path
import psutil
import torch
from PIL import Image
from common import ROOT, DEFAULT_VAL_SPLIT, save_json, device_for, load_model
from dataset import manifest
from inference import predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=ROOT/'results/training/best.pt')
    p.add_argument('--data',type=Path,default=ROOT/'data/dataset_seg')
    p.add_argument('--split',default=DEFAULT_VAL_SPLIT)
    p.add_argument('--output',type=Path,default=ROOT/'results/memory.json')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--iterations',type=int,default=30)
    args = p.parse_args()
    if args.iterations<1:
        p.error('iterations must be positive')
    torch.set_num_threads(2)
    device = device_for(args.device)
    proc = psutil.Process()
    baseline = proc.memory_info().rss
    samples = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(.002):
            samples.append(proc.memory_info().rss)
    sampler = threading.Thread(target=sample,daemon=True);sampler.start()
    try:
        model,ckpt = load_model(args.checkpoint,device)
        paths = manifest(args.data,args.split)
        with Image.open(paths[0]) as im:
            original_size = list(im.size)
            for _ in range(5):
                predict(model,im,ckpt['config'],device)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        gc.collect()
        warm_baseline = proc.memory_info().rss
        start = time.perf_counter()
        for i in range(args.iterations):
            with Image.open(paths[i%len(paths)]) as im:
                mask,prob,confidence = predict(model,im,ckpt['config'],device)
            del mask,prob
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-start
    finally:
        samples.append(proc.memory_info().rss)
        stop.set();sampler.join()
    mib = 1024**2
    result = {'device':str(device), 'split':args.split, 'checkpoint_epoch':ckpt['epoch'], 'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
              'input_hwc':[ckpt['config']['height'],ckpt['config']['width'],3],
              'source_wh':original_size,'batch_size':1,'precision':'float32',
              'warmup_iterations':5,'measured_iterations':args.iterations,
              'ram_before_model_mib':baseline/mib,'ram_after_warmup_mib':warm_baseline/mib,
              'ram_sampled_peak_mib':max(samples)/mib,'ram_increase_over_baseline_mib':(max(samples)-baseline)/mib,
              'ram_sampling_interval_ms':2,'ram_sample_count':len(samples),
              'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/mib if device.type=='cuda' else None,
              'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved(device)/mib if device.type=='cuda' else None,
              'cuda_peaks_scope':'after warm-up reset, live model included, PyTorch allocator only',
              'scope':'RAM sampling covers model load, warm-up and inference. Timed inference includes image disk decode, resize/tensor conversion, network, D2H transfer, binary mask restoration and confidence. Excludes saving outputs.',
              'limitations':'Sampled RSS may miss brief peaks; includes Python/import overhead. CUDA allocator excludes driver/context and other apps. RAM baseline is after imports and before model load.',
              'mean_end_to_end_ms':elapsed/args.iterations*1000,
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'checkpoint_bytes':args.checkpoint.stat().st_size,
              'system_ram_mib':psutil.virtual_memory().total/mib}
    save_json(args.output,result);print(result,flush=True)


if __name__=='__main__':
    main()
