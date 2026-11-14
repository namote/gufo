"""A small family of directions around a base direction, for SPSA (R11).

Standard library only (spsa.py, the steer tool); the basis itself is built
offline with NumPy by direction_basis.py.

A basis file (`gufo-direction-basis-v1`) holds the base direction d0 (the
base profile's row at the direction layer, as float32 values) and k
orthonormal vectors u_1..u_k orthogonal to d0. A coefficient vector alpha
names the direction

    d(alpha) = normalize(d0 + sum_k alpha_k u_k),

whose angle to d0 is atan(|alpha|) (d0 is a unit vector and the u_k are
orthonormal and orthogonal to it), so the cap `max_angle` bounds |alpha| by
tan(max_angle). alpha = 0 is d0 itself: the profile then keeps the base
profile's rows bit for bit (the re-serialized upload differs in bytes, so its
cache identity is not the base profile's; scores are the same).

`profile_with_direction` puts d(alpha) in a base profile's row at the
direction layer: `projected_directions[L]` (base64 little-endian float32, as
prepare.encode_rows writes it) and `directions[L]` (float32 values as
prepare.save_profile writes them) for the projected kind, `directions[L]`
alone for mean_diff. The base must be a one-direction suppression profile
(rank 1), so no other row has to stay orthogonal to the new one.
"""
import base64
import hashlib
import json
import math
from pathlib import Path
import struct

SCHEMA = 'gufo-direction-basis-v1'


def float32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def float32_text(x):
    """prepare.float32_text: the float32 nearest x in at most 9 digits."""
    return float('%.9g' % x)


def encode_row(values):
    """One row as prepare.encode_rows encodes it: base64 of little-endian
    float32 (each value rounded to the nearest float32)."""
    return base64.b64encode(struct.pack(f'<{len(values)}f', *values)).decode()


def decode_row(text, hidden):
    raw = base64.b64decode(text, validate=True)
    if len(raw) != 4 * hidden:
        raise ValueError(f'a row of {hidden} float32 values expected')
    return list(struct.unpack(f'<{hidden}f', raw))


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def load_basis(path):
    basis = json.loads(Path(path).read_text())
    if basis.get('schema') != SCHEMA:
        raise ValueError(f'{path}: not a {SCHEMA} file')
    hidden = basis['hidden']
    if len(basis['d0']) != hidden or not basis['basis'] or any(len(u) != hidden for u in basis['basis']):
        raise ValueError(f'{path}: d0 and every basis vector need {hidden} values')
    return basis


def max_norm(max_angle):
    """The largest |alpha| whose direction lies within max_angle degrees of d0."""
    if not 0 < max_angle < 90:
        raise ValueError('the angle cap must lie in (0, 90) degrees')
    return math.tan(math.radians(max_angle))


def cap(alpha, max_angle):
    """alpha scaled back onto the cap ball when it lies outside."""
    limit = max_norm(max_angle)
    size = norm(alpha)
    if size <= limit:
        return list(alpha)
    return [a * limit / size for a in alpha]


def direction(basis, alpha):
    """d(alpha) as float64 values (None for alpha = 0: the base row itself)."""
    if len(alpha) != len(basis['basis']):
        raise ValueError(f'alpha needs {len(basis["basis"])} values')
    if not any(alpha):
        return None
    d0 = basis['d0']
    scale = norm(d0)
    v = [x / scale for x in d0]
    for a, u in zip(alpha, basis['basis']):
        if a:
            v = [x + a * y for x, y in zip(v, u)]
    size = norm(v)
    return [x / size for x in v]


def angle(basis, alpha):
    """The angle (degrees) between d(alpha) and d0, measured on the vectors."""
    v = direction(basis, alpha)
    if v is None:
        return 0.0
    d0 = basis['d0']
    c = dot(v, d0) / norm(d0)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def active_row(profile, layer, kind):
    """The base profile's direction row at `layer` for `kind` (float values)."""
    if kind == 'projected':
        return decode_row(profile['projected_directions'][layer], profile['hidden'])
    return [float(x) for x in profile['directions'][layer]]


def check_base(profile, basis, layer, kind):
    """The base must be a rank-1 suppression profile whose row at `layer` is
    the basis' d0."""
    if profile.get('mode') == 'add':
        raise ValueError('the direction family needs a suppression profile, not an additive one')
    if profile.get('rank', 1) != 1 or profile.get('subspace_directions') is not None:
        raise ValueError('the direction family needs a rank-1 base profile (one direction)')
    if kind not in ('projected', 'mean_diff'):
        raise ValueError('direction kind must be projected or mean_diff')
    if kind == 'projected' and not profile.get('projected_directions'):
        raise ValueError('the base profile has no projected directions')
    if profile.get('hidden') != basis['hidden']:
        raise ValueError('the basis and the base profile differ in hidden size')
    if not 0 <= layer < profile['layers']:
        raise ValueError('direction layer out of range')
    row = active_row(profile, layer, kind)
    d0 = basis['d0']
    if not norm(row) > 0 or not norm(d0) > 0:
        raise ValueError('a zero direction row')
    c = dot(row, d0) / (norm(row) * norm(d0))
    if c < 1 - 1e-6:
        raise ValueError(f'the basis d0 is not the base profile\'s {kind} row at layer {layer} '
                         f'(cosine {c:.6f}): build the basis from this profile')


def profile_with_direction(profile, basis, alpha, layer, kind):
    """A copy of `profile` with d(alpha) as its row at `layer` (see the
    module doc); alpha = 0 returns the rows unchanged."""
    v = direction(basis, alpha)
    out = dict(profile)
    if v is None:
        return out
    rounded = [float32(x) for x in v]
    directions = list(out['directions'])
    directions[layer] = [float32_text(x) for x in rounded]
    out['directions'] = directions
    if kind == 'projected':
        projected = list(out['projected_directions'])
        projected[layer] = encode_row(rounded)
        out['projected_directions'] = projected
    return out


def profile_bytes(profile):
    """The upload body (compact JSON, as prepare.save_profile writes it)."""
    return json.dumps(profile, ensure_ascii=False, separators=(',', ':')).encode()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
