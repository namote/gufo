"""Idempotent owned-container cleanup; auto-remove races are verified, not hidden."""
import json
import subprocess
import time


def cleanup_owned(name, label, child, host, run=subprocess.run):
    result = dict(cleanup_errors=[])

    def inspect(identifier):
        response = run(['docker','inspect',identifier],capture_output=True,text=True,timeout=15)
        if response.returncode:
            diagnostic = response.stderr.casefold()
            if 'no such object' in diagnostic or 'no such container' in diagnostic:
                return None
            raise RuntimeError(f'Docker inspect failed: {response.stderr}')
        item = json.loads(response.stdout)[0]
        if item['Config']['Labels'].get('gufo.quant.run') != label:
            raise RuntimeError('Container ownership mismatch')
        return item

    try:
        item = inspect(name)
        if item is not None and item['State']['Running']:
            identifier = item['Id']
            try:
                run(['docker','stop','--time','10',identifier],capture_output=True,timeout=20)
            except subprocess.TimeoutExpired:
                pass
            # A timed-out stop can already have removed the --rm container.
            item = inspect(identifier)
            if item is not None and item['State']['Running']:
                run(['docker','kill',identifier],capture_output=True,timeout=15)
                for _ in range(10):
                    item = inspect(identifier)
                    if item is None or not item['State']['Running']:
                        break
                    time.sleep(0.2)
                else:
                    raise RuntimeError('Owned container still running after kill')
    except BaseException as error:
        result['cleanup_errors'].append(repr(error))
    try:
        child.wait(timeout=30)
    except BaseException as error:
        result['cleanup_errors'].append(repr(error))
    try:
        result['returned_KFD_contexts'] = host()['KFD_contexts']
        if result['returned_KFD_contexts']:
            result['cleanup_errors'].append('KFD contexts remain after cleanup')
    except BaseException as error:
        result['cleanup_errors'].append(repr(error))
    return result
