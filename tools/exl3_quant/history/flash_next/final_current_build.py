"""CPU-only coherent current-source build in the retained Nix10.1 closure.

No model mount, GPU devices, old objects, source edits or network. Build outputs
remain incremental. Parent schedules GPU qualification after independent review.
"""
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
ENV=Path('/nix/store/65ksrp4rfgpz1wfyxdqfs3syl2m13p80-gufo-0-unstable-dirty-env')
LD='/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib'
GLIBC='/nix/store/pqyiidz5mi73wv5f91d48s9plcqyj5bz-glibc-2.44-57'
GCC_PREFIX='/nix/store/l3msfz6rylwsyhnjzsbr8ylfshh1xw5f-gcc-prefix'
TARGETS=['gufo','qwen38_flash_next_rq_logits','qwen38_flash_next_exl3_loader_test','quant_exl3_format_test']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def environment():
    data=json.loads(ENV.read_text())['variables']
    env={k:v['value'] for k,v in data.items() if v['type']=='exported' and isinstance(v['value'],str)
         and k not in {'HOME','PWD','OLDPWD','TMPDIR','TMP','TEMP','NIX_BUILD_TOP'}}
    env.update(NIX_BUILD_TOP='/build',LD_LIBRARY_PATH=LD,TMPDIR='/build/tmp',TMP='/build/tmp',TEMP='/build/tmp',
               PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME='/build/cache')
    env['PATH']+=':/opt/venv/bin:/usr/bin:/bin'
    return {**os.environ,**env},data['cmakeFlags']['value']

def configure_args(flags,revision):
    return ['cmake','-S','/build/source','-B','/build/obj','-G','Ninja',*shlex.split(flags),
        '-DCMAKE_BUILD_TYPE=RelWithDebInfo','-DCMAKE_HIP_FLAGS=--gcc-toolchain='+GCC_PREFIX+' -B'+GLIBC+'/lib',
        '-DCMAKE_EXE_LINKER_FLAGS=-L'+GLIBC+'/lib -Wl,--dynamic-linker,'+GLIBC+'/lib/ld-linux-x86-64.so.2 -Wl,-rpath,'+GLIBC+'/lib',
        '-DCMAKE_C_COMPILER=gcc','-DCMAKE_CXX_COMPILER=g++','-DBUILD_TESTING=ON',
        '-DGUFO_BUILD_TOOLS=OFF','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_HIP_ARCHITECTURES=gfx1151',
        '-DGUFO_REVISION='+revision,'-DGUFO_RELEASE_VERSION=development',
        '-DGUFO_ROCR_LIBRARY_DIR='+LD,'-DCMAKE_INSTALL_PREFIX=/build/install',
        '-DPython3_EXECUTABLE=/opt/venv/bin/python3','-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -gz=zlib -DNDEBUG',
        '-DCMAKE_HIP_FLAGS_RELWITHDEBINFO=-O2 -g -gz=zlib -DNDEBUG']

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('configure','build','cpu'),required=True);a=p.parse_args()
    plan=json.loads(Path('/plan/BUILD-PLAN.json').read_text());source=json.loads(Path('/plan/SOURCE.json').read_text())
    assert sha(__file__)==plan['driver_sha256'] and sha(ENV)==plan['environment_sha256']
    for path,digest in source['files_sha256'].items():assert sha(Path('/build/source')/path)==digest,path
    for path,target in source['symlinks'].items():assert str((Path('/build/source')/path).readlink())==target
    assert not Path('/dev/kfd').exists() and not Path('/dev/dri').exists()
    env,flags=environment();out=Path('/build');(out/'tmp').mkdir(exist_ok=True);(out/'cache').mkdir(exist_ok=True)
    for x in ('gcc','g++','cmake','ninja'):
        from shutil import which
        assert str(which(x,path=env['PATH'])).startswith('/nix/store/'),x
    assert Path(LD).is_dir()
    if a.phase=='configure':commands=[configure_args(flags,source['revision'])]
    elif a.phase=='build':commands=[['cmake','--build','/build/obj','--parallel','8','--target',*TARGETS]]
    else:commands=[['/build/obj/gufo','--version'],['/build/obj/quant_exl3_format_test'],
                   ['/build/obj/tests/models/qwen38_flash_next/qwen38_flash_next_exl3_loader_test']]
    state=dict(status='running',phase=a.phase,started=time.time(),commands=commands,gpu_used=False,
               source_revision=source['revision'],environment_sha256=sha(ENV))
    receipt=Path('/receipt')/(a.phase.upper()+'.json');assert not receipt.exists(),'Preserve earlier phase evidence; use explicit reviewed recovery'
    try:
        for i,cmd in enumerate(commands):
            with (Path('/receipt')/(a.phase+f'-{i}.log')).open('x') as log:
                subprocess.run(cmd,cwd='/build/source',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['status']='passed_pending_review'
        if a.phase=='cpu':
            state['binaries']={x:{'sha256':sha(out/'obj'/x),'bytes':(out/'obj'/x).stat().st_size} for x in
                ['gufo','tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits']}
    except BaseException as e:state.update(status='failed',error=repr(e));raise
    finally:state['finished']=time.time();save(receipt,state)

if __name__=='__main__':main()
