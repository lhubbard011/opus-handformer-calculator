"""Power circuit: a frame ``[ a ^ b = ; expansion ]`` that rewrites, not computes.

The frame's result is one step of binary exponentiation written as text; the
rewrite splices it into the line and the existing * and / circuits do the
arithmetic. With b an integer (a is any signed decimal, copied verbatim):

  b negative  ->  (1/a^|b|)        (uses the prompt's ~N precision)
  b == 0      ->  1
  b == 1      ->  a
  b odd       ->  (a*a^(b-1))      (only b's last digit changes: odd - 1 never borrows)
  b even      ->  (a*a)^(b/2)      (halving is local: out_j = 5*[d_{j-1} odd] + floor(d_j/2),
                                    dropping a leading 0)

so a^b takes O(log b) power frames. Expansions that are expressions are
parenthesised: substituted bare, 1/2^3 would become 1/2*2^2 = (1/2)*4. Each case is a list of segments (fixed
bytes, spans copied from the frame, or computed digits) laid out by region
flags over u = q - E - 1, the index of the next expansion byte; byte 0 of the
frame body is ';' (no working).

Global layers 4..11.
"""

from __future__ import annotations

from . import common as cm
from . import primitives as pr
from .model import Form, HeadSpec, Layout, ModelBuilder

CASES = ("negexp", "zero", "unit", "odd", "even")
NAMES = list(cm.ALPHABET.values())

# The frame operator is Pf (stable): this frame's body writes operators, which moves P.
LA: Form = {"Pf": 1.0, "F": -1.0, "one": -1.0}  # base bytes, verbatim (sign, digits, '.')
A0: Form = {"F": 1.0, "one": 1.0}
B0: Form = {"Pf": 1.0, "sbp": 1.0, "one": 1.0}  # first exponent digit
LB: Form = {"E": 1.0, "Pf": -1.0, "sbp": -1.0, "one": -1.0}
U: Form = {"pos": 1.0, "E": -1.0, "one": -1.0}


def _c(k: float) -> Form:
    return pr.const(k)


# The five expansions as segment lists; a segment's length is a linear form in LA (base
# bytes), LB (exponent digits) and lead, so its bounds are linear too (bounds in builder()).
#   negexp ( 1 / a ^ |b| )      : |b| copied from B0, skipping the '-'
#   odd    ( a * a ^ b' d-1 )   : b' = b's first LB-1 digits, then its last digit minus 1
#   even   ( a * a ) ^ half     : LB - lead digits of b/2, each 5*[d_{j-1} odd] + floor(d_j/2)
#   zero   1 ;  unit  a
def _segments() -> dict[str, list[tuple[str, Form, object]]]:
    """case -> [(kind, length, payload)]; kind: fixed (byte dim), copy (start form), dec, half."""
    one = _c(1.0)
    return {
        "negexp": [("fixed", one, "is_lpar"), ("fixed", one, "oh1"), ("fixed", one, "is_slash"), ("copy", LA, A0),
                   ("fixed", one, "is_caret"), ("copy", LB, B0), ("fixed", one, "is_rpar")],
        "zero": [("fixed", one, "oh1")],
        "unit": [("copy", LA, A0)],
        "odd": [("fixed", one, "is_lpar"), ("copy", LA, A0), ("fixed", one, "is_star"), ("copy", LA, A0),
                ("fixed", one, "is_caret"), ("copy", pr.add(LB, _c(-1.0)), B0), ("dec", one, None),
                ("fixed", one, "is_rpar")],
        "even": [("fixed", one, "is_lpar"), ("copy", LA, A0), ("fixed", one, "is_star"), ("copy", LA, A0),
                 ("fixed", one, "is_rpar"), ("fixed", one, "is_caret"), ("half", pr.add(LB, {"lead": -1.0}), None)],
    }


def builder() -> ModelBuilder:
    """The power rewrite circuit at global layers 4..11, on private dims.

    Exponent facts (L4-5), one case flag (L6), segment region flags over u (L7-8),
    copy sources (L9), then the expansion bytes (L10: O_* dims, D0..D9, frame_end,
    work_end), whose writes arith gates by act_pow.
    """
    segs = _segments()
    private = ["sbp", "ld", "d0", "LnzB", "fl2", "nzb", "d0one", "lbge2", "lb1", *CASES, "lead", "dj", "dj1", "flj", "flj1",
               "semi", "src", *(f"Ycp_{d}" for d in NAMES)]
    for case, lst in segs.items():
        private += [f"lt_{case}_{i}" for i in range(len(lst) + 1)]
        private += [f"seg_{case}_{i}" for i in range(len(lst))]
        private += [f"endraw_{case}", f"end_{case}"]
    lay = Layout()
    lay.add(*private)
    b = ModelBuilder(lay)

    # Pf (the frame operator) is ready after the frame locator's layer 3.
    # Layer 4: exponent sign sbp, last digit ld (at E-1), latest nonzero digit LnzB, and
    # fl2 = floor(ld/2) = sum_{t=1..4} [ld >= 2t].
    l4 = b.at(4)
    l4.heads += [
        pr.gather_head("exponent sign", {"Pf": 1.0, "one": 1.0}, "sbp", src="is_minus"),
        pr.gather_head("last exponent digit", pr.add({"E": 1.0}, _c(-1.0)), "ld"),
        pr.gather_head("exponent's last nonzero", pr.add({"E": 1.0}, _c(-1.0)), "LnzB", src="Lnz"),
    ]
    for t in range(1, 5):
        l4.units += pr.clamp01_units(pr.add({"ld": 1.0}, _c(-2.0 * t + 1.0)), {"fl2": 1.0})  # floor(ld/2)

    # Layer 5: first digit d0, nzb = [LnzB >= B0] (a nonzero digit inside the exponent: b != 0),
    # d0one = [d0 == 1], lbge2 = [LB >= 2], lb1 = [LB == 1].
    l5 = b.at(5)
    l5.heads.append(pr.gather_head("first exponent digit", B0, "d0"))
    l5.units += pr.clamp01_units(pr.add({"LnzB": 1.0}, pr.scale(B0, -1.0), _c(1.0)), {"nzb": 1.0})  # b != 0
    l5.units += pr.hat_units({"d0": 1.0}, 1.0, {"d0one": 1.0})
    l5.units += pr.clamp01_units(pr.add(LB, _c(-1.0)), {"lbge2": 1.0})
    l5.units += pr.hat_units(LB, 1.0, {"lb1": 1.0})

    # b mod 2 = ld - 2*floor(ld/2) (0 or 1, exact): b's parity is its last digit's. Read by the
    # odd/even case flags below.
    odd = {"ld": 1.0, "fl2": -2.0}  # parity of b
    # Layer 6: in a power frame exactly one case flag is 1 (ReLU of 0/1 sums: AND of n terms = ReLU(sum - (n-1))),
    # plus lead and semi. 'odd' also fires for b == 1, so a -1 unit cancels it there.
    l6c = b.at(6)
    l6c.units += [
        pr.relu_unit({"sbp": 1.0}, {"negexp": 1.0},
                     label="case flag 'negexp': the exponent starts with '-' (sbp), so this step writes (1/a^|b|)"),
        pr.relu_unit({"one": 1.0, "sbp": -1.0, "nzb": -1.0}, {"zero": 1.0},
                     label="case flag 'zero': exponent not negative (sbp 0) and has no nonzero digit (nzb 0), so this step writes 1"),
        pr.relu_unit({"lb1": 1.0, "d0one": 1.0, "sbp": -1.0, "one": -1.0}, {"unit": 1.0},  # b == 1
                     label="case flag 'unit': exponent is one digit (lb1), that digit is 1 (d0one), not negative; this step writes a"),
        pr.relu_unit(pr.add(odd, {"nzb": 1.0, "sbp": -1.0, "one": -1.0}), {"odd": 1.0},
                     label="case flag 'odd' (main): last exponent digit odd (ld - 2*fl2 = 1), b != 0 (nzb), b not negative (sbp 0)"),
        pr.relu_unit(pr.add(odd, {"nzb": 1.0, "sbp": -1.0, "lb1": 1.0, "d0one": 1.0, "one": -3.0}), {"odd": -1.0},
                     label="case flag 'odd' (correction): cancels it for b == 1 (one digit, lb1, equal to 1, d0one), which is the 'unit' case"),
        pr.relu_unit(pr.add(pr.scale(odd, -1.0), {"nzb": 1.0, "sbp": -1.0}), {"even": 1.0},
                     label="case flag 'even': last exponent digit even (ld - 2*fl2 = 0), b != 0 (nzb), b not negative (sbp 0)"),
        pr.relu_unit({"d0one": 1.0, "lbge2": 1.0, "one": -1.0}, {"lead": 1.0},  # halving drops a leading 0
                     label="'lead': first exponent digit is 1 (d0one) and b has >= 2 digits (lbge2), so b/2 starts with a 0 to drop"),
    ]
    l6c.units += pr.hat_units(pr.add(U, _c(1.0)), 0.0, {"semi": 1.0})  # body byte 0 is ';'

    # cumulative segment boundaries per case (linear forms; lead is known from layer 7 on)
    # bounds[case][i] = sum of the lengths of segments 0..i-1 (segment i is cum_i <= u < cum_{i+1}).
    bounds: dict[str, list[Form]] = {}
    for case, lst in segs.items():
        cum, acc = [_c(0.0)], _c(0.0)
        for _, length, _ in lst:
            acc = pr.add(acc, length)
            cum.append(acc)
        bounds[case] = cum

    # Layer 7: the halving digit's inputs d_j, d_{j-1} and their floor(/2); lt flags per bound.
    # The half segment starts at bounds["even"][6] = 2*LA + 4 ('(' a '*' a ')' '^'); output byte
    # u - start is digit j = u - start + lead of b (lead skips the dropped 0). At j = 0, d_{j-1}
    # reads the '^' at B0-1, value 0.
    l7 = b.at(7)
    half_start = bounds["even"][6]
    j = pr.add(U, pr.scale(half_start, -1.0), {"lead": 1.0})  # exponent digit index for the half digit
    l7.heads += [
        pr.gather_head("d_j", pr.add(B0, j), "dj"),
        pr.gather_head("d_{j-1}", pr.add(B0, j, _c(-1.0)), "dj1"),
    ]
    for t in range(1, 5):
        l7.units += pr.clamp01_units(pr.add({"dj": 1.0}, _c(-2.0 * t + 1.0)), {"flj": 1.0})
        l7.units += pr.clamp01_units(pr.add({"dj1": 1.0}, _c(-2.0 * t + 1.0)), {"flj1": 1.0})
    for case, cum in bounds.items():
        for i, c in enumerate(cum):  # lt_i = [u < cum_i]
            l7.units += pr.clamp01_units(pr.add(c, pr.scale(U, -1.0)), {f"lt_{case}_{i}": 1.0})
        l7.units += pr.hat_units(pr.add(U, pr.scale(cum[-1], -1.0)), 0.0, {f"endraw_{case}": 1.0})

    # Layer 8: seg_{case}_i = [cum_i <= u < cum_{i+1}] AND case active (exact 0/1); end_{case}.
    l8 = b.at(8)
    for case, lst in segs.items():
        for i in range(len(lst)):
            l8.units.append(pr.relu_unit({f"lt_{case}_{i + 1}": 1.0, f"lt_{case}_{i}": -1.0, case: 1.0, "one": -1.0},
                                         {f"seg_{case}_{i}": 1.0},
                                         label=f"region flag 'seg_{case}_{i}': u is in {case} segment {i} ({lst[i][0]}),"
                                               f" [u < its end] - [u < its start], and the {case} case is active"))
        l8.units.append(pr.relu_unit({f"endraw_{case}": 1.0, case: 1.0, "one": -1.0}, {f"end_{case}": 1.0},
                                     label=f"'end_{case}': u equals the {case} expansion's length (endraw) and that case is"
                                           f" active, so the next byte closes the frame"))

    # Layer 9: src = copy start + (u - segment start) on the active copy segment (at most one).
    l9 = b.at(9)
    copy_flags: Form = {}
    for case, lst in segs.items():
        for i, (kind, _, payload) in enumerate(lst):
            if kind == "copy":
                start = pr.add(payload, U, pr.scale(bounds[case][i], -1.0))
                l9.units += pr.times_flag_units(start, {f"seg_{case}_{i}": 1.0}, {"src": 1.0})
                copy_flags[f"seg_{case}_{i}"] = 1.0

    # Layer 10: emit the next expansion byte: copied byte (gather at src), fixed byte, last
    # digit - 1, or the half digit; then ']' at the end and ';' as body byte 0.
    l10 = b.at(10)
    l10.heads.append(HeadSpec(
        name="expansion copy",
        query=[pr.scale({"src": 1.0}, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
        key=[{"pos": 1.0}, {"pos2": 1.0}],
        value={d: {f"Ycp_{d}": 1.0} for d in NAMES},
    ))
    for d in NAMES:
        l10.units.append(pr.relu_unit(pr.add({f"Ycp_{d}": 1.0}, copy_flags, _c(-1.0)), {f"O_{d}": 1.0},
                                      label=f"emit byte {d}: the byte at the copy source (Ycp_{d}) is {d} and u is in a"
                                            f" copy segment of the active case"))
    for case, lst in segs.items():
        for i, (kind, _, payload) in enumerate(lst):
            flag = {f"seg_{case}_{i}": 1.0}
            if kind == "fixed":
                l10.units.append(pr.relu_unit(flag, {f"O_{payload}": 1.0},
                                              label=f"emit the fixed byte {payload} while u is in {case} segment {i}"))
            elif kind == "dec":  # odd exponent: last digit - 1
                l10.units += cm.ramp_mod10_units({"ld": 1.0, "one": -1.0}, 0, 8, flag, 1.0)
            elif kind == "half":  # 5*[d_{j-1} odd] + floor(d_j/2)
                hv = {"dj1": 5.0, "flj1": -10.0, "flj": 1.0}
                l10.units += cm.ramp_mod10_units(hv, 0, 9, flag, 1.0)
        l10.units.append(pr.relu_unit({f"end_{case}": 1.0}, {"frame_end": 1.0},
                                      label=f"emit ']' (frame_end) once the {case} expansion is complete (end_{case})"))
    l10.units.append(pr.relu_unit({"semi": 1.0}, {"work_end": 1.0},
                                  label="emit the END byte ';' (work_end) as body byte 0 (semi): a power frame has no working"))
    return b
