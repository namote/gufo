import copy
import json
from unittest.mock import patch
import unittest
from conversion_admission import clients, desktop_decision, wait_conversion_admission

def fixture():
    proc=dict(PID=12,comm='chrome',starttime_ticks=99,cgroup='user',PGID=12,UIDs=[1000]*4,exe='/opt/google/chrome/chrome',exe_stat=[1,2,3])
    row=dict(proc,verified_proc=proc.copy(),drm_client_id='7',device='/dev/dri/renderD128',fdinfo='drm-driver: amdgpu\ndrm-pdev: 0000:c6:00.0\ndrm-engine-gfx: 100000000 ns\ndrm-engine-compute: 0 ns\n')
    a=dict(boot_id='boot',KFD_contexts=[],errors=[],coherence=dict(stable=True,boot_before='boot',boot_after='boot',begin=9.95,end=10),telemetry=dict(gpu_busy_percent=10),GPU_FDs=[row])
    b=copy.deepcopy(a);b['coherence'].update(begin=11.95,end=12);b['GPU_FDs'][0]['fdinfo']='drm-driver: amdgpu\ndrm-pdev: 0000:c6:00.0\ndrm-engine-gfx: 300000000 ns\ndrm-engine-compute: 0 ns\n'
    m=dict(purpose='conversion_desktop_graphics_only',pdev='0000:c6:00.0',boot_id='boot',clients=[dict(v,allow_gfx=True) for v in clients(a).values()])
    return a,b,m

class Admission(unittest.TestCase):
    def test_wrapper_receipt_serializable(self):
        a,b,m=fixture();samples=iter([a,b])
        with patch('conversion_admission.Path.read_text',return_value=json.dumps(m)):
            h,receipt=wait_conversion_admission(lambda:next(samples),lambda h:False,'fixture',pause=lambda _:None,attempts=2)
        self.assertTrue(receipt['admitted']);json.dumps(receipt)
    def test_original_quiet(self):
        a,b,m=fixture()
        with patch('conversion_admission.Path.read_text',return_value=json.dumps(m)):
            _,r=wait_conversion_admission(lambda:a,lambda h:True,'fixture',pause=lambda _:None)
        self.assertEqual(r['mode'],'original_quiet')
    def test_desktop(self):
        a,b,m=fixture();self.assertTrue(desktop_decision(a,b,m)['admitted'])
    def test_deny(self):
        for kind in ['kfd','busy100','noactivity','compute','encoder','identity','newclient','boot','incoherent','error','resetcounter','unknownowner','excess','unexplained','duration']:
            with self.subTest(kind=kind):
                a,b,m=fixture()
                if kind=='kfd':b['KFD_contexts']=[{'PID':42}]
                elif kind=='busy100':b['telemetry']['gpu_busy_percent']=100
                elif kind=='noactivity':b['GPU_FDs'][0]['fdinfo']=a['GPU_FDs'][0]['fdinfo']
                elif kind=='compute':b['GPU_FDs'][0]['fdinfo']=b['GPU_FDs'][0]['fdinfo'].replace('compute: 0','compute: 1')
                elif kind=='encoder':b['GPU_FDs'][0]['fdinfo']+='drm-engine-enc: 1 ns\n'
                elif kind=='identity':b['GPU_FDs'][0]['verified_proc']['exe']='/tmp/other'
                elif kind=='newclient':b['GPU_FDs'][0]['drm_client_id']='8'
                elif kind=='boot':b['boot_id']='new'
                elif kind=='incoherent':b['coherence']['stable']=False
                elif kind=='error':b['errors']=['unknown']
                elif kind=='resetcounter':b['GPU_FDs'][0]['fdinfo']='drm-driver: amdgpu\ndrm-pdev: 0000:c6:00.0\ndrm-engine-gfx: 0 ns\ndrm-engine-compute: 0 ns\n'
                elif kind=='unknownowner':m['clients'][0]['allow_gfx']=False
                elif kind=='excess':a['telemetry']['gpu_busy_percent']=21
                elif kind=='unexplained':b['telemetry']['gpu_busy_percent']=19
                elif kind=='duration':b['coherence']['end']=10
                self.assertFalse(desktop_decision(a,b,m)['admitted'])
    def test_malformed_counters(self):
        for line in ['drm-engine-compute: 99 us', 'drm-engine-compute: -1 ns', 'drm-engine-compute: 0 ns garbage', 'drm-engine-gfx: 0 ns']:
            a,b,m=fixture()
            for h in (a,b):h['GPU_FDs'][0]['fdinfo'] += line+'\n'
            self.assertFalse(desktop_decision(a,b,m)['admitted'],line)
    def test_device_and_scan(self):
        for kind in ['pdev','driver','overlap','longscan']:
            a,b,m=fixture()
            if kind=='pdev':b['GPU_FDs'][0]['fdinfo']=b['GPU_FDs'][0]['fdinfo'].replace('c6:00.0','c7:00.0')
            elif kind=='driver':b['GPU_FDs'][0]['fdinfo']=b['GPU_FDs'][0]['fdinfo'].replace('amdgpu','unknown')
            elif kind=='overlap':b['coherence'].update(begin=9.9,end=12)
            else:b['coherence']['begin']=11
            self.assertFalse(desktop_decision(a,b,m)['admitted'],kind)
    def test_duplicate_client_not_double_counted(self):
        a,b,m=fixture();a['GPU_FDs'].append(copy.deepcopy(a['GPU_FDs'][0]));b['GPU_FDs'].append(copy.deepcopy(b['GPU_FDs'][0]));self.assertEqual(desktop_decision(a,b,m)['gfx_percent'],10)

if __name__=='__main__':unittest.main()
