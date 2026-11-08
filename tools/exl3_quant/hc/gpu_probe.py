"""GPU qualification against saved CPU HC candidates; no baseline regeneration."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file, save_file
from . import quant as q
from .gpu_quant import feedback_library, gptq, measure, refine_up
from .refine import refine
from .weights import load_weight


def probe(library, pilot, source_model, target_model, output):
    library,pilot,source_model,target_model,output=map(Path,(library,pilot,source_model,target_model,output))
    output.mkdir(parents=True,exist_ok=False)
    lib=feedback_library(library)
    report={'status':'running','torch':torch.__version__,'torch_hip':torch.version.hip,
            'library_sha256':q.sha(library),'device':torch.cuda.get_device_name(),'cases':{}}
    def save(): (output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        rng=np.random.default_rng(1741);w=rng.normal(size=(5,64)).astype(np.float32)
        x=rng.normal(size=(128,64));h=x.T@x/len(x)
        expected,_,damping=q.gptq(w,h)
        packed,meta=gptq(w,torch.tensor(h,device='cuda'),lib,damping)
        if not np.array_equal(packed,expected): raise ValueError('synthetic sequential feedback mismatch')
        report['cases']['synthetic']={'exact':True,**meta};save()
        w=rng.normal(size=(5,320)).astype(np.float32)
        x=rng.normal(size=(640,320));h=x.T@x/len(x)
        initial,_,damping=q.gptq(w,h)
        for scale in (False,True):
            expected,_=refine(w,initial,h,damping,scales=scale)
            actual,meta=refine_up(w,initial,torch.tensor(h,device='cuda'),damping,scales=scale)
            if not np.array_equal(actual,expected):raise ValueError('synthetic refinement differs from CPU oracle')
            report['cases']['refinement_'+str(scale)]={'exact':True,'meta':meta};save()
        for p in ('up','down'):
            root=pilot/('gptq-'+p);old=json.loads((root/'RESULT.json').read_text());key=old['tensor']
            w,identity=load_weight(source_model,key,'BF16');_,target=load_weight(target_model,key,'F16')
            if identity!=old['source_identity'] or target!=old['target_identity']:raise ValueError('changed source')
            stats=pilot/('stats-'+p)/'stats.npz'
            if q.sha(stats)!=old['receipt']['stats_sha256']:raise ValueError('changed stats')
            if q.sha(root/'candidate.safetensors')!=old['candidate_sha256']:raise ValueError('changed candidate')
            with np.load(stats,allow_pickle=False) as f:
                h=torch.tensor(f['H'],device='cuda');held=torch.tensor(f['X_holdout'],device='cuda')
            packed,meta=gptq(w,h,lib,old['damping'])
            metrics=measure(w,packed,h,held)
            expected=load_file(str(root/'candidate.safetensors'))[key]
            ratio=metrics['holdout_output_sse']/old['candidate']['holdout_output_sse']
            report['cases'][p]={'gptq':meta,'metrics':metrics,'differing_bytes':int(np.count_nonzero(packed!=expected)),
                                 'holdout_sse_ratio_to_saved_CPU':ratio}
            save_file({key:packed},str(output/(p+'-gptq.safetensors')))
            save()
            if ratio>1.005:raise ValueError('GPU feedback holdout regression vs saved CPU')
            if p=='up':
                packed,meta=refine_up(w,packed,h,old['damping'])
                metrics=measure(w,packed,h,held)
                ref=json.loads((pilot/'refined-up/RESULT.json').read_text())
                if ref['baseline_result_sha256']!=q.sha(root/'RESULT.json'):raise ValueError('refinement baseline changed')
                ratio=metrics['holdout_output_sse']/ref['metrics']['gptq_codes_scales']['holdout_output_sse']
                report['cases'][p]['refinement']={'meta':meta,'metrics':metrics,'holdout_sse_ratio_to_saved_CPU':ratio}
                save_file({key:packed},str(output/'up-refined.safetensors'));save()
                if ratio>1.005:raise ValueError('GPU refinement holdout regression vs saved CPU')
            del h,held;torch.cuda.empty_cache()
        report['status']='passed_pending_independent_review';save()
    except BaseException as e:
        report.update(status='failed',error=repr(e));save();raise
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','pilot','source-model','target-model','output'):p.add_argument('--'+name,required=True)
    print(json.dumps(probe(**vars(p.parse_args())),indent=2))
