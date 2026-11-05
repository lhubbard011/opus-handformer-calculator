"""Small builders shared by the recurrent-depth calculator's circuits (handformer/loop).

Every flag, hold and register here is exact: a gather leaks ~1e-13 of its neighbours, so a
gathered integer is re-quantized (robust steps) before it is used in arithmetic, and every
0/1 a gate or a select reads comes from a robust step (``pr.clamp01_units``).
"""

from __future__ import annotations

from .. import common as cm
from .. import primitives as pr
from ..model import Form, HeadSpec

BINMINUS: Form = {"is_minus": 1.0, "prev_num": 1.0, "prev_rpar": 1.0, "one": -1.0}

# Bytes an expression copy can hold (a cell), and the input dim that flags each.
CONTENT = {**cm.ALPHABET, ord("~"): "is_tilde", ord(" "): "is_space"}
# "the next position is this byte": one exact 0/1 dim per byte a position can predict
HOLD = {byte: f"H_{dim}" for byte, dim in {**CONTENT, ord("="): "is_eq", cm.EOS: "is_bos"}.items()}
DIGIT_HOLD = [HOLD[ord("0") + d] for d in range(10)]


def gather(name: str, target: Form, value: dict[str, Form]) -> HeadSpec:
    """``gather_head`` with any value map (one or many dims moved from position ``target``)."""
    return HeadSpec(name=name, query=[pr.scale(target, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
                    key=[{"pos": 1.0}, {"pos2": 1.0}], value=value)


def relu(x: Form, dst: str, label: str, k: float = 1.0):
    return pr.relu_unit(x, {dst: k}, label)


def flag(x: Form, dst: str, label: str, sign: float = 1.0) -> list:
    """``dst += sign * [x >= 1]`` as a robust step (exact for x within 1/4 of an integer)."""
    units = pr.clamp01_units(x, {dst: sign})
    for u, part in zip(units, ("rises at 1/2", "caps at 1")):
        u.label = f"{label} ({part})"
    return units


def gflag(x: Form, gate: Form, dst: str, label: str, sign: float = 1.0) -> list:
    """``flag`` silenced unless the 0/1 form ``gate`` is 1."""
    return flag(pr.gated(x, gate), dst, label, sign)


def hat(x: Form, dst: str, label: str, gate: Form | None = None, sign: float = 1.0) -> list:
    """``dst += sign * [x == 0]`` (two robust steps), silenced unless ``gate`` is 1."""
    up, down = pr.add(x, pr.const(1.0)), x
    if gate is not None:
        up, down = pr.gated(up, gate), pr.gated(down, gate)
    return [*flag(up, dst, label, sign), *flag(down, dst, label, -sign)]


def all_of(*flags: str | Form) -> Form:
    """A 0/1 AND as a form: 1 when every flag is 1, <= 0 otherwise (a gate or a flag input)."""
    terms = [{f: 1.0} if isinstance(f, str) else f for f in flags]
    return pr.add(*terms, pr.const(1.0 - len(terms)))


def sel(x: Form, gate: Form, dst: str, k: float = 1.0) -> list:
    """``dst += k * x * gate`` (signed x, 0/1 gate form)."""
    return pr.times_flag_units(x, gate, {dst: k})


def sel_to(x: Form, gate: Form, write: Form) -> list:
    return pr.times_flag_units(x, gate, write)


def count_up(x: Form, hi: int, dst: str, label: str, gate: Form | None = None, k: float = 1.0) -> list:
    """``dst += k * x`` for an integer x in 0..hi, exactly: the sum of the robust steps
    [x >= t], t = 1..hi (re-quantizes a gathered integer)."""
    units = []
    for t in range(1, hi + 1):
        xt = pr.add(x, pr.const(-(t - 1.0)))
        units += flag(xt if gate is None else pr.gated(xt, gate), dst, f"{label}: [x >= {t}]", k)
    return units


def tens(x: Form, hi: int, dst: str, label: str, gate: Form | None = None, k: float = 1.0) -> list:
    """``dst += k * floor(x / 10)`` for an integer x in 0..hi: robust steps [x >= 10 m]."""
    units = []
    for m in range(1, hi // 10 + 1):
        xt = pr.add(x, pr.const(-(10.0 * m - 1.0)))
        units += flag(xt if gate is None else pr.gated(xt, gate), dst, f"{label}: [x >= {10 * m}]", k)
    return units


def onehot_times(oh: str, scalar: Form, gate: Form, dst: str, label: str, top: int = 9) -> list:
    """``dst += v * scalar`` where the exact one-hot ``oh<v>`` is set (0 <= v * scalar <= 81):
    one unit per v, silenced by 100 (more than any product) unless that v is the one and
    ``gate`` holds."""
    return [relu(pr.add(pr.scale(scalar, float(v)), pr.scale({f"{oh}{v}": 1.0}, 100.0),
                        pr.scale(gate, 100.0), pr.const(-200.0)), dst, label) for v in range(1, top + 1)]


def onehot(x: Form, prefix: str, label: str, gate: Form | None = None, values=range(10)) -> list:
    """``prefix<v> = [x == v]`` for an integer x in 0..9 and each v in ``values`` (robust hats)."""
    units = []
    for v in values:
        units += hat(pr.add(x, pr.const(-float(v))), f"{prefix}{v}", f"{label}: is it {v}?", gate)
    return units


def hold_digit(x: Form, gate: Form, label: str) -> list:
    """Hold the digit byte of the integer x (0..9) where the 0/1 form ``gate`` is 1."""
    units = []
    for v in range(10):
        units += hat(pr.add(x, pr.const(-float(v))), DIGIT_HOLD[v], f"{label}: digit {v}", gate)
    return units


def copy_value(x: Form, gate: Form, dst: str) -> list:
    """``dst += x`` where ``gate`` is 1 (a register passed to the next position unchanged)."""
    return sel(x, gate, dst)
