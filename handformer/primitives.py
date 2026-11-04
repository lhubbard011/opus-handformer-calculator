"""Reusable hand-weight constructions. Every op circuit is built from these.

All forms are linear forms over named residual dims ({dim: coef}); constants
are expressed through the always-1 dim ``one``. Predicates over integer-valued
scalars are exact because the ReLU breakpoints sit strictly between integers
(robust steps: at 1/2 and 3/4, see ``clamp01_units``), and attention lookups are
exact because each score is built so the intended key leads every other key by
a fixed margin (beta = 30..60 logits) at any length: softmax acts as argmax with
leak O(n * e^-30). DESIGN.md §4 and §11 tabulate these; §12 lists the constants.
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

    Query columns ``[2*beta*t, -beta]`` against key columns ``[pos, pos2]``:

        score(k) = 2*beta*t*k - beta*k^2 = -beta*(k - t)^2 + beta*t^2

    The ``beta*t^2`` term is the same for every key, so softmax ignores it. For
    integer t the target scores 0 and its two neighbours -beta (= -30), others
    less: the gap is beta per step whatever the length, so the read is ``src``
    at t up to a leak of ~2e^-30. Example: t = 5 reads ``val`` of byte 5.

    ``t`` may be any linear form the residual holds (an offset from a marker,
    a reflection ``2S-1-q``, a clamped pointer). Gathering ``pos2`` at x is
    the square table (returns x^2, see ``product_form``).
    Limits: t must be an exact integer (t = j + 1/2 ties j and j+1 and the read
    averages them). Under the causal mask a target beyond the query, or below
    0, silently lands on the nearest visible key (the query itself, or BOS),
    so circuits clamp onto a zero-valued fence on purpose. float64 rounding
    of beta*pos^2 stays far below the beta gap up to ~1e5-1e6 positions
    (DESIGN.md §12).
    Used for nearly every copy: ~90 call sites across all circuits.
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
    and the keys only ``pos^2`` and ``pos``.

    Why exact: with d = xc/s - j, |d| <= (s-1)/(2s) < 1/2, so k = j is the
    closest integer and score(j) - score(j+-1) = beta*s^2*(1 - 2|d|) >= beta*s:
    the runner-up trails by at least beta*s (>= 30). Example: x = 17, s = 5:
    xc = 15, s*k - xc = 0 at k = 3 = floor(17/5).
    Requires floor(x/s) <= query position (k ranges over visible keys only) and
    x, s exact integers. Used by multiplication (line index j) and division
    (grid row).
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
    positional features double as a lookup table of squares. ``sum_sq`` and
    ``a_sq`` name the dims holding (x+y)^2 and x^2; ``b_sq`` is a form for y^2.
    The result is a linear form, so it costs no dim: readers use it directly. Exact for
    integers (float64 holds integers below 2^53 exactly); limited by the
    gather, so x+y must be a visible position (<= the query's). Example:
    x = 3, y = 4: (49 - 9 - 16) / 2 = 12. Used by division and multiplication.
    """
    return add(scale({sum_sq: 1.0}, 0.5), scale({a_sq: 1.0}, -0.5), scale(b_sq, -0.5))


BIG_SQ = BIG * BIG * 10.0  # gate for features that scale like BIG * pos (exact in float64)
NEXT_BETA = 60.0  # equality sharpness in next_match_head
NEXT_BONUS = 30.0  # bonus for matching keys inside the equal-pointer group


def latest_feature(match: Form, dst: str, bos: bool = True) -> list[UnitSpec]:
    """Key feature for ``latest_head``: ``2*(pos + 1)`` where ``match`` (0/1) holds.

    The unit is ``2 * ReLU(pos + 1 - BIG*(1 - match))``: with match = 1 it
    writes 2(pos+1), with match = 0 its input is below 0 for every pos < BIG-1,
    so it writes exactly 0 (``match`` must be an exact 0/1 flag; a residue e
    would shift the input by BIG*e). Features of successive matches differ by
    >= 2, i.e. >= 2*LATEST_BETA = 80 logits in ``latest_head``.

    BOS also gets +1, so with no match the head reads BOS (position 0): BOS
    scores 1*40 against 0 for every other key. The factor 2 keeps a matching
    BOS (2+1 = 3) strictly below a match at position 1 (4); with ``pos + 1``
    they would both score 2 and attention would average them (reading 0.5).
    Several match forms may write into one ``dst`` (frame.py does); pass
    ``bos=False`` after the first so the fallback is added once per class.
    Example: matches at 3 and 7 give features 8 and 16; the head reads 7.
    """
    units = [UnitSpec(read=add({"pos": 1.0}, const(1.0 - BIG), scale(match, BIG)), write={dst: 2.0},
                      label=f"landmark tag for '{dst}': writes 2·(position+1) on a matching byte, so a "
                            "'latest' head can find the most recent one")]
    if bos:  # add the fallback once per class, not once per match form
        units.append(UnitSpec(read={"is_bos": 1.0}, write={dst: 1.0},
                              label=f"fallback for '{dst}': the start byte scores 1, so 'no match yet' reads position 0"))
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
        label=f"'earliest in this frame' tag for '{dst}': newer frames score higher, earlier positions "
              "within a frame score higher",
    )


def next_match_head(name: str, pm: str, pm2: str, match: Form, lt: Form, dst: str,
                    dst_found: str | None = None) -> HeadSpec:
    """Position of the first key after t that matches a class.

    Per class, every position carries ``pm`` = the latest match strictly before
    it (the class's latest-match value gathered at pos-1) and ``pm2 = pm^2``
    (pos^2 gathered at pm). Let ``lt`` be the latest match at or before t. The
    first match r after t is the only *matching* key with pm_r == lt; the
    other keys with pm == lt are the non-matches in (lt, r). (Keys up to lt
    have pm < lt; keys after r have pm >= r > lt.) This is how causal
    attention searches forward without counting or a window (DESIGN.md §4.3).

    score = -beta*(pm_k - lt)^2 + bonus*match_k
          = -beta*pm2_k + 2*beta*lt*pm_k + bonus*match_k + const
    Query columns ``[-beta, 2*beta*lt, bonus]`` against keys ``[pm2, pm, match]``.
    Target scores +bonus (30), its group 0, every other key <= bonus - beta
    (-30, since pm is an integer != lt): a gap of 30 at any length.
    Example: matches at 2 and 9, t = 4: lt = 2; keys 3..8 have pm = 2 and no
    match (score 0), key 9 has pm = 2 and matches (score 30): reads 9.
    With no match after t (up to the query), the group of non-matches wins
    and averages, so ``dst`` is meaningless and ``dst_found`` reads ~0 (it
    sums ``match`` of the winners). ``match``, ``pm`` and ``lt`` must be exact
    (0/1 and integers). Used by the parser (expr.py): first ')' after p,
    first '*' '/' and first '+' '-' after p (choosing o), and the end of b.
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

    score(k) = LATEST_BETA * feature_k (query ``[40]`` against key
    ``[feature]``), value = ``pos``. The latest match leads the previous one by
    40 * (feature gap): >= 40 for ``pos_if`` features (gap 1), >= 80 for
    ``latest_feature`` (gap 2), so the read is an exact argmax. Other
    orderings come from the feature: ``earliest_if`` (BIG - pos) and
    ``newest_frame_earliest`` (BIG*frame - pos) reuse this head.
    Limit: with no matching key every feature is 0, all keys tie and the read
    is their mean position. Callers either guarantee a match (e.g. addition
    also tags the frame's '=') or use ``latest_feature``'s BOS fallback.
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
        label=f"position tag for '{dst}': writes this byte's position when all its conditions hold, else 0",
    )


def earliest_if(predicates: Form, required: float, dst: str) -> UnitSpec:
    """``dst += BIG - pos`` if the predicate AND holds, else 0.

    Paired with ``latest_head`` this finds the *earliest* matching position.
    """
    return UnitSpec(
        read=add({"pos": -1.0}, scale(predicates, BIG), const(-BIG * (required - 1.0))),
        write={dst: 1.0},
        label=f"'earliest' tag for '{dst}': earlier matching positions score higher",
    )


def times_flag_units(x: Form, flag: Form, write: Form) -> list[UnitSpec]:
    """``write * x * flag`` for a 0/1 ``flag`` and signed ``|x| < BIG``."""
    off = add(scale(flag, -BIG), const(BIG))  # BIG * (1 - flag)
    names = ", ".join(write)
    return [
        UnitSpec(read=add(x, scale(off, -1.0)), write=dict(write),
                 label=f"select into {names}: passes the value through only when the flag is 1 (positive part)"),
        UnitSpec(read=add(scale(x, -1.0), scale(off, -1.0)), write=scale(write, -1.0),
                 label=f"select into {names}: passes the value through only when the flag is 1 (negative part)"),
    ]


def relu_unit(x: Form, write: Form, label: str = "") -> UnitSpec:
    return UnitSpec(read=dict(x), write=dict(write), label=label)


STEP_SLOPE = 4.0  # robust steps are exact for inputs within +-1/4 of an integer


def clamp01_units(x: Form, write: Form) -> list[UnitSpec]:
    """``write * [x >= 1]`` for (near-)integer x, exactly 0 or 1.

    clamp(K*(x - 1/2), 0, 1) with K = 4: the breakpoints sit at x = 1/2 and
    x = 3/4, so noise of up to 1/4 on an integer input never leaks into the
    output. Exact flags matter because flags gate selects that scale by BIG.

    Two units on z = 4x - 2: ``ReLU(z) - ReLU(z - 1)``. For x <= 1/4 both are 0;
    for x >= 3/4 the difference is exactly 1. An integer x (any sign) +- 1/4
    never lands on the ramp (z stays outside (-1, 1), the claim
    tests/test_claims.py checks). Example: x = 2 gives ReLU(6) - ReLU(5) = 1;
    x = 0.0000012 (attention leak) gives 0. A step with breakpoints at the
    integers would pass that 1e-6 through, and a BIG-scaled select would turn
    it into a whole position (DESIGN.md §12, lesson 1). ``x`` must be within
    1/4 of an integer; between, the output is fractional. Building block of
    ``hat_units``, ``exhausted_units`` and most flags (~60 call sites).
    """
    z = add(scale(x, STEP_SLOPE), const(-STEP_SLOPE / 2.0))
    names = ", ".join(write)
    return [
        UnitSpec(read=z, write=dict(write), label=f"yes/no test into {names}: rises from 0 to 1 as the input passes 1/2",
                 claims=(("outside", -1.0, 1.0),)),  # input within 1/4 of an integer: never on the ramp
        UnitSpec(read=add(z, const(-1.0)), write=scale(write, -1.0),
                 label=f"yes/no test into {names}: caps the answer at exactly 1 (input at or above 3/4)"),
    ]


def hat_units(x: Form, center: float, write: Form) -> list[UnitSpec]:
    """``write * [x == center]`` for (near-)integer x: step(x >= c) - step(x >= c+1)."""
    units = [
        *clamp01_units(add(x, const(-(center - 1.0))), write),
        *clamp01_units(add(x, const(-center)), scale(write, -1.0)),
    ]
    names = ", ".join(write)
    for u, part in zip(units, ("on at ≥ c", "on at ≥ c", "off again at ≥ c+1", "off again at ≥ c+1")):
        u.label = f"equality test into {names}: 1 exactly when the input equals {center:g} ({part})"
    return units


def gated(x: Form, gate: Form, on_when: float = 1.0) -> Form:
    """Silence a unit unless ``gate == on_when`` (gate is 0/1).

    Adds ``-BIG * |gate - on_when|`` to the unit's input; valid when the unit's
    ungated input is below LEN_MAX, which holds for every unit here.

    on_when = 1: ``x + BIG*gate - BIG`` = x - BIG*(1 - gate);
    on_when = 0: ``x - BIG*gate``. When the gate holds the input is unchanged;
    otherwise it is x - 10^6 < 0 and ReLU outputs exactly 0. The gate must be
    an exact 0/1 flag (from a robust step): a residue e moves the input by
    BIG*e. This one mechanism replaces routing: phases, operator selection
    (``arith._Namespace.unit`` gates every output-writing unit by ``act_*``)
    and digit outputs (``common.ramp_mod10_units``) all use it.
    """
    if on_when == 1.0:
        return add(x, scale(gate, BIG), const(-BIG))
    return add(x, scale(gate, -BIG))
