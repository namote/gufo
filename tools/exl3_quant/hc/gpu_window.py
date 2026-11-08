"""Owned GPU window with per-stage evidence and exact server restoration."""
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import time
import uuid
from .capture_window import LIVE, LLM, available, load_ops, run, same_config, save, sha


class Window:
    def __init__(self, root, model):
        self.root=Path(root).resolve();self.model=Path(model).resolve()
        self.state={'status':'preparing','cases':{},'cleanup_errors':[],'service_restored':False}
        self.owned=None;self.lock=None;self.stopped=False;self.kernel=None
    def update(self): save(self.root/'STATE.json',self.state)
    def __enter__(self):
        self.root.mkdir(parents=True,exist_ok=False)
        self.installed=json.loads((LIVE/'INSTALLED.json').read_text())
        for rel,digest in self.installed['files'].items():
            if sha(LIVE/rel)!=digest:raise ValueError('installed ops changed: '+rel)
        self.before=json.loads(run([LLM,'status']).stdout)
        if not self.before.get('ready') or Path(self.before['model']).resolve()!=self.model:
            raise ValueError('expected owned ready model')
        self.command=json.loads(run([LLM,'config']).stdout)
        save(self.root/'BEFORE.json',self.before);save(self.root/'COMMAND-BEFORE.json',self.command)
        save(self.root/'INSTALLED.json',self.installed)
        self.ops,journal=load_ops();self.kernel=journal.KernelDelta(self.root,run)
        def interrupt(*_):raise InterruptedError('HC window interrupted')
        self.handlers={s:signal.signal(s,interrupt) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            self.kernel.start();self.stopped=True;self.update();run([LLM,'stop'],timeout=90)
            self.lock=self.ops.LOCK.open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if self.ops.npu():raise RuntimeError('other NPU owner')
            for _ in range(10):
                h=self.ops.host();save(self.root/'GPU-PRE.json',h)
                if self.ops.quiet(h):break
                time.sleep(2)
            else:raise RuntimeError('GPU not quiet')
            if available()<80*2**30:raise RuntimeError('insufficient available memory')
            self.state['status']='running';self.update();return self
        except BaseException as e:
            self.__exit__(type(e),e,e.__traceback__);raise
    def execute(self, label, argv, timeout):
        directory=self.root/label;directory.mkdir(exist_ok=False)
        name='gufo-hc-all-'+uuid.uuid4().hex;argv=list(argv);argv[2:2]=['--name',name]
        self.owned=name;save(directory/'ARGV.json',argv)
        self.state['active']=label;self.update();start=time.monotonic();minimum=available()
        with (directory/'run.log').open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    minimum=min(minimum,available())
                    if minimum<4*2**30 or time.monotonic()-start>timeout:raise RuntimeError('GPU stage memory/time bound')
                    time.sleep(1)
                if child.returncode:raise RuntimeError(label+' failed, see saved log')
            finally:
                c=subprocess.run(['docker','rm','-f',name],capture_output=True,text=True,timeout=30)
                (directory/'cleanup.log').write_text(c.stdout+c.stderr)
                if c.returncode and 'No such container' not in c.stderr:raise RuntimeError('owned cleanup failed')
                if child.poll() is None:child.wait(timeout=30)
                self.owned=None
        post=self.ops.host();save(directory/'GPU-POST.json',post)
        if post['KFD_contexts']:raise RuntimeError('KFD owner remains')
        self.kernel.collect()
        self.state['cases'][label]={'status':'completed','seconds':time.monotonic()-start,'min_available_bytes':minimum}
        self.update()
    def __exit__(self, kind, error, trace):
        if error is not None:self.state.update(status='failed',error=repr(error))
        else:self.state['status']='completed_pending_review'
        if self.owned:
            try:
                removed=run(['docker','rm','-f',self.owned],timeout=30)
                self.owned=None
            except Exception as e:
                self.state['cleanup_errors'].append(repr(e))
                try:
                    check=subprocess.run(['docker','inspect',self.owned],capture_output=True,text=True,timeout=30)
                    if check.returncode and 'No such object' in check.stderr:self.owned=None
                except Exception as check_error:self.state['cleanup_errors'].append('inspect: '+repr(check_error))
        if self.kernel and self.kernel.cursor:
            try:
                self.state['kernel_final']=self.kernel.collect(raise_error=False)
                if self.state['kernel_final']['new_GPU_error']:self.state.update(status='failed',error='new kernel GPU error')
            except Exception as e:self.state['cleanup_errors'].append(repr(e))
        if self.stopped:
            try:
                if self.owned:raise RuntimeError('owned container removal not confirmed; refusing server restore')
                for _ in range(10):
                    host=self.ops.host()
                    save(self.root/'GPU-FINAL.json',host)
                    if self.ops.quiet(host):break
                    time.sleep(2)
                else:raise RuntimeError('GPU remains occupied; refusing overlapping server restore')
                for rel,digest in self.installed['files'].items():
                    if sha(LIVE/rel)!=digest:raise RuntimeError('restore controller changed')
                if self.lock:self.lock.close();self.lock=None
                run([LLM,'start'],timeout=120)
                for _ in range(60):
                    after=json.loads(run([LLM,'status']).stdout)
                    if after.get('ready'):break
                    if after.get('error'):raise RuntimeError('restore failed: '+str(after['error']))
                    time.sleep(2)
                command=json.loads(run([LLM,'config']).stdout)
                save(self.root/'AFTER.json',after);save(self.root/'COMMAND-AFTER.json',command)
                if not after.get('ready') or not same_config(self.before,after) or self.command!=command:
                    raise RuntimeError('restore configuration/readiness mismatch')
                self.state['service_restored']=True
            except Exception as e:self.state['cleanup_errors'].append('restore: '+repr(e))
        if self.lock:self.lock.close()
        for s,h in getattr(self,'handlers',{}).items():signal.signal(s,h)
        self.update()
        if error is None and (self.state['status']=='failed' or self.state['cleanup_errors'] or not self.state['service_restored']):
            raise RuntimeError('HC window cleanup/restore needs diagnosis')
