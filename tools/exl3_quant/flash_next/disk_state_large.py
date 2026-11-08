"""User-selected 800-row / 10 GiB prefetch policy; qualified I/O mechanics reused."""
from pathlib import Path
from disk_state_async import DiskRows as AsyncRows


class DiskRows(AsyncRows):
    def __init__(self, source, scratch_parent, *, read_bytes=10*2**30,
                 write_bytes=2*2**30, ahead=800):
        super().__init__(source, scratch_parent, read_bytes=read_bytes,
                         write_bytes=write_bytes, ahead=ahead)


def install(cm, scratch_parent):
    old_load, old_save = cm.load_tensor, cm.save_tensor
    stores = []
    def load(filename, args):
        if filename != 'ckpt/state.safetensors':
            return old_load(filename, args)
        store = DiskRows(Path(args['work_dir'])/filename, scratch_parent)
        if len(store) != args['cal_rows']:
            store.close()
            raise ValueError('Checkpoint/calibration row count mismatch')
        stores.append(store)
        print(f'GUFO_STATE_IO_ACTIVE rows={len(store)} read=10GiB write=2GiB ahead=800 CPU_workers=2', flush=True)
        return store
    def save(value, filename, args):
        if isinstance(value, DiskRows):
            value.save_checkpoint(Path(args['work_dir'])/filename)
        else:
            old_save(value, filename, args)
    cm.load_tensor, cm.save_tensor = load, save
    return stores
