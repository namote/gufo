"""In-place pinned upstream rebuild; no GPU devices, exclusive owner."""
import fcntl,json,subprocess,time,shutil
from pathlib import Path
from test.experiments.gufo_upstream_port_20261023 import window

def main():
 out=window.ROOT/'tmp/test-results/kyojin-update-20261103'
 name='kyojin-update-186-build'
 with window.LOCK.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  h=window.host()
  assert not h['KFD_contexts'] and h['available_bytes']>=80*2**30
  assert not window.npu()
  assert shutil.disk_usage(window.ROOT).free>=150*2**30
  cmd=['docker','run','--rm','--name',name,'--network','none','--user','1000:1000','--memory','20g','--memory-swap','20g','--tmpfs','/tmp:rw,size=2g','--entrypoint','/bin/bash','-v',str(window.ROOT)+':/workspace','-v',str(out)+':/receipt','ai-tools/gufo-eval:f3-native-b4ebe27c-rocm101','/workspace/test/experiments/gufo_upstream_port_20261023/kyojin_runtime/build.sh']
  report={'command':cmd,'started':time.time(),'GPU_devices':False,'max_seconds':1200}
  (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n')
  with (out/'build.log').open('w') as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   try:
    while p.poll() is None:
     assert time.time()-report['started']<1200,'Build deadline'
     assert shutil.disk_usage(window.ROOT).free>=150*2**30,'Disk floor'
     time.sleep(2)
   finally:
    if p.poll() is None:
     subprocess.run(['docker','stop','-t','5',name],capture_output=True,timeout=15)
     p.wait(timeout=15)
  report.update(returncode=p.returncode,seconds=time.time()-report['started'])
  (out/'BUILD.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)
  assert p.returncode==0
if __name__=='__main__':main()
