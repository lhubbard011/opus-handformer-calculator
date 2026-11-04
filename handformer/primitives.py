"""Reusable hand-weight constructions. Every op circuit is built from these.

All forms are linear forms over named residual dims ({dim: coef}); constants
are expressed through the always-1 dim ``one``. Predicates over integer-valued
scalars are exact because the ReLU breakpoints sit on integer boundaries.
"""

from __future__ import annotations

from .model import Form, HeadSpec, UnitSpec

# Global numeric bound: float64 resolution of beta*pos^2 and the big-M used
# to gate position features. See DESIGN.md "Global numeric bound".
LEN_MAX = 1e6
BIG = LEN_MAX  # gate constant: subtracting BIG*(1-g) silences any unit whose input < LEN_MAX

FIND_BETA = 60.0  # find the (unique) marker; BOS sink gets half of this
GATHER_BETA = 30.0  # quadratic position lookup sharpness (gap = beta per step)
LATEST_BETA = 40.0  # "max position among matches" lookup


def add(*forms: Form, **scaled: float) -> Form:
    """Sum of linear forms. ``add(f, g)`` -> f + g."""
    out: Form = {}
    for f in forms:
        for d, c in f.items():
            out[d] = out.get(d, 0.0) + c
    for d, c in scaled.items():
        out[d] = out.get(d, 0.0) + c
    return out


def scale(form: Form, k: float) -> Form:
    return {d: c * k for d, c in form.items()}


def const(k: float) -> Form:
    return {"one": k}


# ---------------------------------------------------------------- attention


def find_head(name: str, marker: str, dst_pos: str, dst_flag: str | None = None) -> HeadSpec:
    """Position of the unique ``marker`` byte seen so far (0 if absent).

    The BOS byte is an attention sink scoring half the marker, so with no
    marker visible the head reads BOS: pos 0, flag 0. With the marker visible
    the leak onto BOS is e^-30.
    """
    value = {"pos": {dst_pos: 1.0}}
    if dst_flag is not None:
        value[marker] = {dst_flag: 1.0}
    return HeadSpec(
        name=name,
        query=[{"one": FIND_BETA}],
        key=[{marker: 1.0, "is_bos": 0.5}],
        value=value,
    )


def gather_head(name: str, target: Form, dst: str, src: str = "val") -> HeadSpec:
    """Read ``src`` at absolute position ``target`` (a linear form of the query).

    score = 2*beta*t*k - beta*k^2 = -beta*(k - t)^2 + const(query).
    Adjacent keys differ by beta, independent of sequence length.
    """
    return HeadSpec(
        name=name,
        query=[scale(target, 2 * GATHER_BETA), const(-GATHER_BETA)],
        key=[{"pos": 1.0}, {"pos2": 1.0}],
        value={src: {dst: 1.0}},
    )


def floor_div_head(name: str, s2: Form, s_xc: Form, dst: str) -> HeadSpec:
    """``dst = floor(x / s)`` for integers x >= 0, s >= 1, by attention over a ruler.

    Every visible key k offers itself (value = its position). The score is
    ``-beta * (s*k - xc)^2`` with ``xc = x - (s-1)/2``, maximised at
    k = floor(x/s): for x = js + m, ``xc/s`` lies strictly within 1/2 of j.
    Expanded, score = -beta*s^2 * k^2 + 2*beta*(s*xc) * k + const, so the query
    needs only ``s^2`` and ``s*xc`` (products obtained with the pos^2 trick)
    and the keys only ``pos^2`` and ``pos``. Gap to the runner-up >= beta*s.
    Requires floor(x/s) <= query position.
    """
    return HeadSpec(
        name=name,
        query=[scale(s2, -GATHER_BETA), scale(s_xc, 2 * GATHER_BETA)],
        key=[{"pos2": 1.0}, {"pos": 1.0}],
        value={"pos": {dst: 1.0}},
    )


def product_form(sum_sq: str, a_sq: str, b_sq: Form) -> Form:
    """``x*y = ((x+y)^2 - x^2 - y^2) / 2`` from squares in the residual.

    The squares come from gathering ``pos2`` at positions x+y, x and y: the
    positional features double as a lookup table of squares.
    """
    return add(scale({sum_sq: 1.0}, 0.5), scale({a_sq: 1.0}, -0.5), scale(b_sq, -0.5))


BIG_SQ = BIG * BIG * 10.0  # gate for features that scale like BIG * pos (exact in float64)
NEXT_BETA = 60.0  # equality sharpness in next_match_head
NEXT_BONUS = 30.0  # bonus for matching keys inside the equal-pointer group


def latest_feature(match: Form, dst: str, bos: bool = True) -> list[UnitSpec]:
    """Key feature for ``latest_head``: ``2*(pos + 1)`` where ``match`` (0/1) holds.

    BOS also gets +1, so with no match the head reads BOS (position 0). The
    factor 2 keeps a matching BOS (2+1) strictly below a match at position 1
    (4); without it they tie and attention averages them.
    """
    units = [UnitSpec(read=add({"pos": 1.0}, const(1.0 - BIG), scale(match, BIG)), write={dst: 2.0})]
    if bos:  # add the fallback once per class, not once per match form
        units.append(UnitSpec(read={"is_bos": 1.0}, write={dst: 1.0}))
    return units


def newest_frame_earliest(predicates: Form, required: float, frame: str, dst: str) -> UnitSpec:
    """Key feature ``BIG*frame_k - pos`` where the predicate AND holds, else 0.

    Lexicographic order: keys in the newest frame beat keys in older frames
    (frame starts differ by >= 1, worth BIG), and inside a frame the earliest
    key wins. With ``latest_head`` this finds the first match in the current frame.
    """
    return UnitSpec(
        read=add({frame: BIG, "pos": -1.0}, scale(predicates, BIG_SQ), const(-BIG_SQ * required)),
        write={dst: 1.0},
    )


def next_match_head(name: str, pm: str, pm2: str, match: Form, lt: Form, dst: str,
                    dst_found: str | None = None) -> HeadSpec:
    """Position of the first key after t that matches a class.

    Per class, every position carries ``pm`` = the latest match strictly before
    it (the class's latest-match value gathered at pos-1) and ``pm2 = pm^2``
    (pos^2 gathered at pm). Let ``lt`` be the latest match at or before t. The
    first match r after t is the only *matching* key with pm_r == lt; the
    other keys with pm == lt are the non-matches in (lt, r).

    score = -beta*(pm_k - lt)^2 + bonus*match_k
          = -beta*pm2_k + 2*beta*lt*pm_k + bonus*match_k + const
    Target scores +bonus, its group 0, every other key <= bonus - beta.
    With no match after t (up to the query), ``dst_found`` reads ~0.
    """
    value: dict[str, Form] = {"pos": {dst: 1.0}}
    if dst_found is not None:
        for d, c in match.items():
            value.setdefault(d, {})[dst_found] = c
    return HeadSpec(
        name=name,
        query=[const(-NEXT_BETA), scale(lt, 2 * NEXT_BETA), const(NEXT_BONUS)],
        key=[{pm2: 1.0}, {pm: 1.0}, dict(match)],
        value=value,
    )


def latest_head(name: str, key_feature: str, dst: str) -> HeadSpec:
    """Read the maximum of ``key_feature`` over visible keys into ``dst``.

    ``key_feature`` must be "position if predicate else 0" (see ``pos_if``), so
    this returns the position of the latest key satisfying the predicate.
    """
    return HeadSpec(
        name=name,
        query=[{"one": LATEST_BETA}],
        key=[{key_feature: 1.0}],
        value={"pos": {dst: 1.0}},
    )


# ---------------------------------------------------------------- MLP units


def pos_if(predicates: Form, required: float, dst: str) -> UnitSpec:
    """``dst += pos`` if the 0/1 predicate sum equals ``required``, else 0.

    ``predicates`` is a sum of 0/1 terms whose maximum is ``required`` (an
    AND of ``required`` conditions, terms may carry signs, e.g. seen_eq - seen_semi).
    """
    return UnitSpec(
        read=add({"pos": 1.0}, scale(predicates, BIG), const(-BIG * required)),
        write={dst: 1.0},
    )


def earliest_if(predicates: Form, required: float, dst: str) -> UnitSpec:
    """``dst += BIG - pos`` if the predicate AND holds, else 0.

    Paired with ``latest_head`` this finds the *earliest* matching position.
    """
    return UnitSpec(
        read=add({"pos": -1.0}, scale(predicates, BIG), const(-BIG * (required - 1.0))),
        write={dst: 1.0},
    )


def times_flag_units(x: Form, flag: Form, write: Form) -> list[UnitSpec]:
    """``write * x * flag`` for a 0/1 ``flag`` and signed ``|x| < BIG``."""
    off = add(scale(flag, -BIG), const(BIG))  # BIG * (1 - flag)
    return [
        UnitSpec(read=add(x, scale(off, -1.0)), write=dict(write)),
        UnitSpec(read=add(scale(x, -1.0), scale(off, -1.0)), write=scale(write, -1.0)),
    ]


def relu_unit(x: Form, write: Form) -> UnitSpec:
    return UnitSpec(read=dict(x), write=dict(write))


STEP_SLOPE = 4.0  # robust steps are exact for inputs within +-1/4 of an integer


def clamp01_units(x: Form, write: Form) -> list[UnitSpec]:
    """``write * [x >= 1]`` for (near-)integer x, exactly 0 or 1.

    clamp(K*(x - 1/2), 0, 1) with K = 4: the breakpoints sit at x = 1/2 and
    x = 3/4, so noise of up to 1/4 on an integer input never leaks into the
    output. Exact flags matter because flags gate selects that scale by BIG.
    """
    z = add(scale(x, STEP_SLOPE), const(-STEP_SLOPE / 2.0))
    return [
        UnitSpec(read=z, write=dict(write)),
        UnitSpec(read=add(z, const(-1.0)), write=scale(write, -1.0)),
    ]


def hat_units(x: Form, center: float, write: Form) -> list[UnitSpec]:
    """``write * [x == center]`` for (near-)integer x: step(x >= c) - step(x >= c+1)."""
    return [
        *clamp01_units(add(x, const(-(center - 1.0))), write),
        *clamp01_units(add(x, const(-center)), scale(write, -1.0)),
    ]


def gated(x: Form, gate: Form, on_when: float = 1.0) -> Form:
    """Silence a unit unless ``gate == on_when`` (gate is 0/1).

    Adds ``-BIG * |gate - on_when|`` to the unit's input; valid when the unit's
    ungated input is below LEN_MAX, which holds for every unit here.
    """
    if on_when == 1.0:
        return add(x, scale(gate, BIG), const(-BIG))
    return add(x, scale(gate, -BIG))
