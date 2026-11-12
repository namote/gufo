"""Run one answer's hidden tests. Runs inside the sandbox container.

Standard library only; keep it compatible with Python 3.8+ (the container
image's interpreter, not the host's).

The judge (this process, started without --child) never runs answer code.
Each step runs in a fresh `python3 -I harness.py --child` process that gets
only the code, the entry point and the test's argument expression, and reports
a type-tagged plain-data encoding of what happened (the returned value and the
arguments after the call, or the raised exception's class names). The judge
decodes that and evaluates `expect`/`raises`/`check` itself, so an answer that
forges its child's output still has to produce the right value. The judge
marks itself non-dumpable, so other processes of the same user cannot open its
/proc entries, and it signs its report with a nonce only the host knows.

stdin:  one JSON job
        {"code": str, "entry_point": str, "tests": [test, ...], "nonce": str,
         "timeout": seconds per test, "memory_mb": address-space cap per test,
         "run_main": bool}
        test = {"args": "<python tuple expression>",
                "expect": "<python expression>"          # result == expect
                | "raises": "ValueError|TypeError"       # builtin exception names
                | "check": "<expression over result, args>",
                "approx": false, "no_mutate": false, "scramble_args": false}
        check sees the arguments as they were after the call; scramble_args
        changes every list/dict/set in the arguments after the call, before
        the result is encoded, so a result that aliases them fails.
stdout: one JSON object {"nonce", "load": {...}, "main": {...} | null, "tests": [...]}
        each entry {"status": "pass|fail|error|timeout|crash", "detail": str}
"""
import builtins
import collections
import datetime
import decimal
import fractions
import json
import math
import os
import signal
import subprocess
import sys

try:
    import resource
except ImportError:  # pragma: no cover - not POSIX
    resource = None

# Names available to test argument and expectation expressions. The answer's
# own namespace is never visible to them.
EXPR_NAMES = {
    '__builtins__': builtins,
    'math': math,
    'datetime': datetime,
    'date': datetime.date,
    'time': datetime.time,
    'timedelta': datetime.timedelta,
    'timezone': datetime.timezone,
    'Decimal': decimal.Decimal,
    'Fraction': fractions.Fraction,
    'OrderedDict': collections.OrderedDict,
    'Counter': collections.Counter,
    'defaultdict': collections.defaultdict,
    'inf': math.inf,
    'nan': math.nan,
}

DETAIL_LIMIT = 300
# Child start-up, argument building and result encoding are outside the
# per-test limit, which times only the answer's own code; the judge's wall
# clock allows this much on top.
WALL_SLACK = 8.0
PAYLOAD_LIMIT = 64 << 20
ENCODE_DEPTH = 200


def short(value):
    try:
        text = value if isinstance(value, str) else repr(value)
    except BaseException as error:  # noqa: BLE001 - a hostile __repr__
        text = '<unrepresentable %s>' % type(error).__name__
    return text if len(text) <= DETAIL_LIMIT else text[:DETAIL_LIMIT] + '...'


# Plain-data encoding ------------------------------------------------------------
#
# Only exact standard types are encoded; anything else (a subclass with its own
# __eq__, a custom class, a cycle, deep nesting) becomes an opaque marker that
# equals nothing.

_type = type
SEQUENCES = {list: 'list', tuple: 'tuple', set: 'set', frozenset: 'frozenset',
             collections.deque: 'deque'}
MAPPINGS = {dict: 'dict', collections.OrderedDict: 'OrderedDict',
            collections.defaultdict: 'defaultdict', collections.Counter: 'Counter'}


def encode(value, depth=0, active=None):
    active = set() if active is None else active
    kind = _type(value)
    if depth > ENCODE_DEPTH:
        return ['opaque', 'nesting deeper than %d' % ENCODE_DEPTH]
    if value is None:
        return ['none']
    if kind is bool:
        return ['bool', value]
    if kind is int:  # hex: no int-to-str digit limit, and exact in JSON
        return ['int', value] if abs(value) < 1 << 53 else ['int', hex(value)]
    if kind is float:
        return ['float', repr(value)]
    if kind is complex:
        return ['complex', repr(value.real), repr(value.imag)]
    if kind is str:
        return ['str', value]
    if kind is bytes:
        return ['bytes', value.hex()]
    if kind is range:
        return ['range', value.start, value.stop, value.step]
    if kind in SEQUENCES or kind in MAPPINGS:
        if id(value) in active:
            return ['opaque', 'a container that holds itself']
        active.add(id(value))
        try:
            if kind in SEQUENCES:
                return [SEQUENCES[kind], [encode(v, depth + 1, active) for v in value]]
            return [MAPPINGS[kind], [[encode(k, depth + 1, active), encode(v, depth + 1, active)]
                                     for k, v in dict.items(value)]]
        finally:
            active.discard(id(value))
    if kind is datetime.timezone:
        offset = value.utcoffset(None)
        return ['timezone', offset.days, offset.seconds, offset.microseconds,
                datetime.timezone.tzname(value, None)]
    if kind is datetime.timedelta:
        return ['timedelta', value.days, value.seconds, value.microseconds]
    if kind in (datetime.date, datetime.datetime, datetime.time):
        tz = getattr(value, 'tzinfo', None)
        if tz is not None and _type(tz) is not datetime.timezone:
            return ['opaque', 'tzinfo %s' % _type(tz).__qualname__]
        naive = value.replace(tzinfo=None) if tz is not None else value
        return [kind.__name__, naive.isoformat(), None if tz is None else encode(tz)]
    if kind is decimal.Decimal:
        return ['Decimal', str(value)]
    if kind is fractions.Fraction:
        return ['Fraction', str(value.numerator), str(value.denominator)]
    return ['opaque', '%s.%s object' % (getattr(kind, '__module__', '?'),
                                        getattr(kind, '__qualname__', '?'))]


class Opaque(object):
    """A value the judge does not compare; equal to nothing."""

    def __init__(self, what):
        self.what = what

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    __hash__ = object.__hash__

    def __repr__(self):
        return '<%s>' % self.what


def decode(node):
    tag = node[0]
    if tag == 'none':
        return None
    if tag == 'bool':
        return bool(node[1])
    if tag == 'int':
        return node[1] if _type(node[1]) is int else int(node[1], 16)
    if tag == 'float':
        return float(node[1])
    if tag == 'complex':
        return complex(float(node[1]), float(node[2]))
    if tag == 'str':
        return node[1]
    if tag == 'bytes':
        return bytes.fromhex(node[1])
    if tag == 'range':
        return range(node[1], node[2], node[3])
    if tag in ('list', 'tuple', 'set', 'frozenset', 'deque'):
        items = [decode(v) for v in node[1]]
        return {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset,
                'deque': collections.deque}[tag](items)
    if tag in ('dict', 'OrderedDict', 'defaultdict', 'Counter'):
        pairs = [(decode(k), decode(v)) for k, v in node[1]]
        if tag == 'defaultdict':
            return collections.defaultdict(None, pairs)
        return {'dict': dict, 'OrderedDict': collections.OrderedDict,
                'Counter': lambda p: collections.Counter(dict(p))}[tag](pairs)
    if tag == 'timezone':
        offset = datetime.timedelta(node[1], node[2], node[3])
        return datetime.timezone(offset, node[4])
    if tag == 'timedelta':
        return datetime.timedelta(node[1], node[2], node[3])
    if tag in ('date', 'datetime', 'time'):
        cls = getattr(datetime, tag)
        value = cls.fromisoformat(node[1])
        return value if node[2] is None else value.replace(tzinfo=decode(node[2]))
    if tag == 'Decimal':
        return decimal.Decimal(node[1])
    if tag == 'Fraction':
        return fractions.Fraction(int(node[1]), int(node[2]))
    return Opaque(node[1] if len(node) > 1 else 'opaque')


def contains_opaque(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Opaque):
            return item
        if isinstance(item, (list, tuple, set, frozenset, collections.deque)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
    return None


# Comparison ----------------------------------------------------------------------

def approx_equal(got, want, rel=1e-9, abs_tol=1e-9):
    if isinstance(want, float) or isinstance(got, float):
        if isinstance(got, bool) or isinstance(want, bool):
            return got == want
        try:
            got_f, want_f = float(got), float(want)
        except (TypeError, ValueError):
            return False
        if math.isnan(want_f):
            return math.isnan(got_f)
        return math.isclose(got_f, want_f, rel_tol=rel, abs_tol=abs_tol)
    if isinstance(want, (list, tuple)):
        return (type(got) is type(want) and len(got) == len(want)
                and all(approx_equal(g, w, rel, abs_tol) for g, w in zip(got, want)))
    if isinstance(want, dict):
        return (isinstance(got, dict) and set(got) == set(want)
                and all(approx_equal(got[k], want[k], rel, abs_tol) for k in want))
    return got == want


def exact_equal(got, want):
    # Containers and strings must also match in kind: a tuple is not the list
    # the contract asked for (the dict variants are all dicts).
    kind = next((k for k in (dict, list, tuple, set, frozenset, str)
                 if isinstance(want, k)), None)
    return (kind is None or isinstance(got, kind)) and got == want


def judge(test, outcome):
    """Score one child outcome against the test; the child never sees `test`."""
    status = outcome.get('kind')
    if status == 'raised':
        names = outcome.get('names', [])
        detail = '%s: %s' % (names[0].split('.')[-1] if names else '?',
                             short(outcome.get('detail', '')))
        if 'raises' in test:
            # Only the most derived builtin class counts, checked against the
            # judge's own classes: a class deriving from several builtin
            # exceptions (or a forged name list) matches only through that one.
            first = next((n[len('builtins.'):] for n in names
                          if n.startswith('builtins.')), None)
            found = getattr(builtins, first, None) if first else None
            if isinstance(found, type) and issubclass(found, BaseException):
                for name in test['raises'].split('|'):
                    wanted = getattr(builtins, name.strip(), None)
                    if isinstance(wanted, type) and issubclass(found, wanted):
                        return {'status': 'pass', 'detail': ''}
            return {'status': 'fail', 'detail': detail}
        return {'status': 'error', 'detail': detail}
    if status != 'returned':
        return {'status': 'error', 'detail': short(outcome.get('detail', 'no result'))}
    result = decode(outcome['value'])
    if 'raises' in test:
        return {'status': 'fail',
                'detail': 'expected %s, returned %s' % (test['raises'], short(result))}
    odd = contains_opaque(result)
    if odd is not None:
        return {'status': 'fail', 'detail': 'returned %r' % odd}
    after = decode(outcome['args'])  # the arguments as they were after the call
    if test.get('no_mutate'):
        before = eval(test['args'], dict(EXPR_NAMES))  # noqa: S307 - our fixture
        if not isinstance(before, tuple):
            before = (before,)
        if contains_opaque(after) is not None or not exact_equal(after, before):
            return {'status': 'fail', 'detail': 'mutated its input'}
    if 'check' in test:
        if contains_opaque(after) is not None:
            return {'status': 'fail', 'detail': 'arguments hold %r' % contains_opaque(after)}
        names = dict(EXPR_NAMES, result=result, args=after)
        try:
            ok = bool(eval(test['check'], names))  # noqa: S307 - our fixture
        except Exception as error:  # noqa: BLE001 - a malformed result
            return {'status': 'fail', 'detail': 'check raised %s on %s'
                    % (type(error).__name__, short(result))}
        return {'status': 'pass' if ok else 'fail',
                'detail': '' if ok else 'check failed on %s' % short(result)}
    want = eval(test['expect'], dict(EXPR_NAMES))  # noqa: S307 - our fixture
    ok = approx_equal(result, want) if test.get('approx') else exact_equal(result, want)
    return {'status': 'pass' if ok else 'fail',
            'detail': '' if ok else 'got %s, want %s' % (short(result), short(want))}


# Child -------------------------------------------------------------------------

def child_main():
    request = json.loads(sys.stdin.read())
    out = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    if resource is not None:
        seconds = int(math.ceil(request['timeout'] + WALL_SLACK))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
        if request.get('memory_mb'):
            limit = int(request['memory_mb']) << 20
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 3000))
    try:
        outcome = child_step(request)
    except BaseException as error:  # noqa: BLE001
        outcome = {'kind': 'error', 'detail': '%s: %s' % (type(error).__name__,
                                                          short(str(error)))}
    try:
        payload = json.dumps(outcome).encode()
    except BaseException as error:  # noqa: BLE001
        payload = json.dumps({'kind': 'error', 'detail': 'unencodable result: %s'
                              % type(error).__name__}).encode()
    if len(payload) > PAYLOAD_LIMIT:
        payload = json.dumps({'kind': 'error', 'detail': 'result over 64 MiB'}).encode()
    view = memoryview(payload)
    while view:
        view = view[os.write(out, view):]
    os._exit(0)


class Overtime(BaseException):
    pass


def on_alarm(signum, frame):
    raise Overtime()


def timed(request, action):
    """Run action() under the per-test real-time limit."""
    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, request['timeout'])
    try:
        return action()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def child_step(request):
    mode = request['mode']
    namespace = {'__name__': '__main__' if mode == 'main' else 'solution',
                 '__builtins__': builtins}
    code = compile(request['code'], '<answer>', 'exec')
    try:
        timed(request, lambda: exec(code, namespace))  # noqa: S102 - sandboxed
    except Overtime:
        return {'kind': 'timeout'}
    except BaseException as error:  # noqa: BLE001
        return {'kind': 'error', 'detail': '%s: %s' % (type(error).__name__,
                                                       short(str(error)))}
    if mode in ('load', 'main'):
        return {'kind': 'ok'}
    function = namespace.get(request['entry_point'])
    if not callable(function):
        return {'kind': 'error', 'detail': 'no callable %s' % request['entry_point']}
    args = eval(request['args'], dict(EXPR_NAMES))  # noqa: S307 - our fixture
    if not isinstance(args, tuple):
        args = (args,)
    try:
        result = timed(request, lambda: function(*args))
    except Overtime:
        return {'kind': 'timeout'}
    except BaseException as error:  # noqa: BLE001
        names = ['%s.%s' % (c.__module__, c.__qualname__) for c in type(error).__mro__]
        return {'kind': 'raised', 'names': names, 'detail': short(str(error))}
    encoded_args = encode(args)
    if request.get('scramble_args'):
        scramble(args, set())  # a result that shares objects with them changes too
    return {'kind': 'returned', 'value': encode(result), 'args': encoded_args}


SCRAMBLED = '<scrambled>'


def scramble(value, seen):
    """Change every list, dict and set inside value in place."""
    kind = _type(value)
    if kind not in (list, tuple, dict, set) or id(value) in seen:
        return
    seen.add(id(value))
    children = list(value.values()) if kind is dict else list(value)
    for child in children:
        scramble(child, seen)
    if kind is list:
        value.append(SCRAMBLED)
    elif kind is dict:
        value[SCRAMBLED] = SCRAMBLED
    elif kind is set:
        value.add(SCRAMBLED)


# Judge -------------------------------------------------------------------------

def make_undumpable():
    """Keep same-user processes (the answers) out of this process's /proc fds."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 4
    except Exception:  # noqa: BLE001 - best effort outside Linux
        pass


def run_child(job, mode, args=None, scramble_args=False):
    request = dict(code=job['code'], entry_point=job['entry_point'], mode=mode,
                   args=args, scramble_args=scramble_args,
                   timeout=float(job.get('timeout', 2.0)),
                   memory_mb=job.get('memory_mb'))
    process = subprocess.Popen([sys.executable, '-I', os.path.abspath(__file__), '--child'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        stdout, _ = process.communicate(json.dumps(request).encode(),
                                        timeout=request['timeout'] + WALL_SLACK)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        try:  # the child and anything it left behind
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if timed_out:
        process.communicate()
        return {'kind': 'timeout'}
    try:
        return json.loads(stdout.decode())
    except ValueError:
        code = process.returncode
        return {'kind': 'crash',
                'detail': 'signal %d' % -code if code and code < 0 else 'exit %s' % code}


def step_status(outcome, timeout):
    kind = outcome.get('kind')
    if kind == 'ok':
        return {'status': 'pass', 'detail': ''}
    if kind == 'timeout':
        return {'status': 'timeout', 'detail': 'over %.1fs' % timeout}
    if kind == 'crash':
        return {'status': 'crash', 'detail': outcome.get('detail', '')}
    return {'status': 'error', 'detail': short(outcome.get('detail', ''))}


def reap():
    """As PID 1 without an init, collect orphans the children left."""
    try:
        while os.waitpid(-1, os.WNOHANG)[0] > 0:
            pass
    except ChildProcessError:
        pass


def main():
    job = json.loads(sys.stdin.read())
    sys.stdin.close()
    make_undumpable()
    timeout = float(job.get('timeout', 2.0))
    report = {'nonce': job.get('nonce'), 'load': None, 'main': None, 'tests': []}
    try:
        compile(job['code'], '<answer>', 'exec')
    except (SyntaxError, ValueError) as error:
        report['load'] = {'status': 'error', 'detail': 'SyntaxError: %s' % short(str(error))}
    if report['load'] is None:
        report['load'] = step_status(run_child(job, 'load'), timeout)
    if report['load']['status'] == 'pass':
        if job.get('run_main'):
            report['main'] = step_status(run_child(job, 'main'), timeout)
        for test in job['tests']:
            outcome = run_child(job, 'call', test['args'], bool(test.get('scramble_args')))
            kind = outcome.get('kind')
            if kind in ('returned', 'raised'):
                report['tests'].append(judge(test, outcome))
            elif kind in ('timeout', 'crash', 'error'):  # child-reported timeouts too
                report['tests'].append(step_status(outcome, timeout))
            else:  # 'ok' or anything else is not an answer to a call
                report['tests'].append({'status': 'error',
                                        'detail': 'no result (%s)' % short(kind)})
            reap()
    else:
        report['tests'] = [{'status': 'error', 'detail': 'load failed'} for _ in job['tests']]
    sys.stdout.write(json.dumps(report) + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    if sys.argv[1:] == ['--child']:
        child_main()
    else:
        main()
