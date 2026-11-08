"""Bounded fresh-context diagnostic; no model inputs or production mutation."""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import time
import torch
from exllamav3.modules.quant.exl3_lib.quantize import g_scale_gss, set_worker_stream

OUT = Path('/output/model-job-c03/repair-015-quant-stall')
assert os.environ.get('EXL3_QT_KERNEL') is None, 'Use unchanged production default'
assert os.environ.get('EXL3_QT_OPTIMIZED') is None, 'Use unchanged production default'
torch.set_grad_enabled(False)
torch.manual_seed(47)
print('BEGIN_GPU_HEALTH', flush=True)
x = torch.arange(256, device='cuda', dtype=torch.float32)
assert torch.equal((x + 1).cpu(), torch.arange(256, dtype=torch.float32) + 1)
print('GPU_HEALTH_PASS', flush=True)
weights = [torch.randn(2560, 640, device='cuda', dtype=torch.float32) for _ in range(2)]
torch.cuda.synchronize()
qa = dict(seed=47, K=4, devices=[0], apply_out_scales=True, mul1=True)
results = []

def one(index):
    with torch.inference_mode():
        stream = torch.cuda.Stream(device=0)
        set_worker_stream(stream)
        try:
            with torch.cuda.stream(stream):
                scale, mse = g_scale_gss(weights[index], False, qa.copy())
                val = float(mse.item())
                stream.synchronize()
            assert math.isfinite(scale) and scale > 0 and math.isfinite(val) and val >= 0
            return [scale, val]
        finally:
            set_worker_stream(None)

baseline = None
for workers in (1, 2, 2):
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(one, range(2)))
    torch.cuda.synchronize()
    if baseline is None:
        baseline = values
    assert values == baseline, (values, baseline)
    results.append(dict(workers=workers, values=values, seconds=time.monotonic()-start))
    print(json.dumps(results[-1]), flush=True)
(OUT/'PROBE.json').write_text(json.dumps(dict(status='passed', scope='synthetic K4 scale-search liveness/finite/repeat only; not root-cause proof or full layer correctness', results=results), indent=2)+'\n')
