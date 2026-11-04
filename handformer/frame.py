"""Shared locator: markers, frame-relative flags, phases, operator routing.

Every circuit reads these dims instead of finding markers itself.

Sequence grammar (all written by the model after the prompt)::

    \\n X_0 = [ redex = working ; result ] X_1 = [ ... ] X_n \\n

A *frame* ``[ ... ]`` is one reduction step: the copied redex ``a op b``,
``=``, that op's working, ``;`` (or ``<``), the result. After ``]`` the model
writes the next expression line X_{t+1}, ended by ``=`` (more to do) or
``\\n`` (done).

Every marker is found as the *latest* occurrence (``latest_head`` over a
``pos+1 if byte`` feature), so every frame sees its own markers. Flags are
frame-relative: ``seen_eq = [latest '=' > latest '[']`` and so on.

Layer 1 attn  flags of the previous byte (digit? ')'?) -> unary vs binary '-'
Layer 1 mlp   latest-match key features for every marker class
Layer 2 attn  latest heads: F '[', Rb ']', E '=', S ';'/'<', R '|', P any op,
              plus classes used by the parser: Lg '(' or ']', Lrp ')',
              Lmul '*', Ladd '+'/'-', Lnd non-digit
Layer 2 mlp   open = [F > Rb]; seen_eq/semi/bar = [E/S/R > F]
Layer 3 attn  Ls = Rb at F (start of the source line); p = Lg at F-1
              (innermost '(' of the source line, or its start);
              previous-match pointers PM_c = L_c at pos-1;
              operator type at P; sign at S; (P-F)^2 for multiplication
              operand signs sa, sb (a '-' right after '[' / after the operator)
Layer 3 mlp   eff_sb = sb XOR [op is -]; phase1 (emit '['), phase2 (copy), phase4 (rewrite)
Layer 4 mlp   act_add: + or - with sa == eff_sb (add magnitudes); act_sub: signs differ
              (subtract magnitudes); act_mul: '*'. Each AND is one or two ReLUs.
"""

from __future__ import annotations

from . import primitives as pr
from .model import Form, HeadSpec, ModelBuilder

# A '-' is unary (a sign) when the byte before it is not part of a number
# (digit or '.') and not ')'.
# "binary minus" as a 0/1 match form: is_minus + prev_digit + prev_rpar - 1 equals 1
# exactly for a binary '-', 0 for a unary one, <= 0 for any other byte.
BINMINUS: Form = {"is_minus": 1.0, "prev_digit": 1.0, "prev_rpar": 1.0, "one": -1.0}

# marker -> the match forms of its byte class (one feature unit each; at most one fires)
MARKERS: dict[str, list[Form]] = {
    "F": [{"is_lbr": 1.0}],
    "Rb": [{"is_rbr": 1.0}],
    "E": [{"is_eq": 1.0}],
    "S": [{"is_semi": 1.0}],
    "R": [{"is_bar": 1.0}],
    "P": [{"is_plus": 1.0}, {"is_star": 1.0}, BINMINUS],  # binary operators only
    "Lg": [{"is_lpar": 1.0}, {"is_rbr": 1.0}],
    "Lrp": [{"is_rpar": 1.0}],
    "Lmul": [{"is_star": 1.0}],
    "Ladd": [{"is_plus": 1.0}, BINMINUS],
    "Lnd": [{"one": 1.0, "is_digit": -1.0, "is_dot": -1.0}],  # not part of a number
    "Ldot": [{"is_dot": 1.0}],
    "Lnz": [{"nz": 1.0}],  # nonzero digit (trailing-zero count of a result)
}
NEXT_CLASSES = ("Lrp", "Lmul", "Ladd", "Lnd")  # classes searched with next_match
OPS = {"add": "is_plus", "sub": "is_minus", "mul": "is_star"}

DIMS = (
    "prev_digit", "prev_rpar", "is_unary", "is_binminus",
    *(f"f_{m}" for m in MARKERS), *MARKERS,
    "open", "seen_eq", "seen_semi", "seen_bar",
    "Ls", "p", *(f"PM_{c}" for c in NEXT_CLASSES), *(f"typ_{op}" for op in OPS),
    "neg", "sa", "sb", "eff_sb", "PF2",
    *(f"act_{op}" for op in OPS), "phase1", "phase2", "phase4",
)

# Fences of the frame's operands, past their signs: a in (F+sa, P), b in (P+sb, E).
FA: Form = {"F": 1.0, "sa": 1.0}
PB: Form = {"P": 1.0, "sb": 1.0}


def build(b: ModelBuilder) -> None:
    prev = pr.add({"pos": 1.0}, pr.const(-1.0))
    l1 = b.at(1)
    l1.heads += [
        HeadSpec(  # "digit" here means part of a number: a digit or '.'
            name="prev byte in a number?",
            query=[pr.scale(prev, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
            key=[{"pos": 1.0}, {"pos2": 1.0}],
            value={"is_digit": {"prev_digit": 1.0}, "is_dot": {"prev_digit": 1.0}},
        ),
        pr.gather_head("prev byte ')'?", prev, "prev_rpar", src="is_rpar"),
    ]
    l1.units.append(pr.relu_unit({"is_minus": 1.0, "prev_digit": -1.0, "prev_rpar": -1.0},
                                 {"is_unary": 1.0}))
    l1.units.append(pr.relu_unit(BINMINUS, {"is_binminus": 1.0}))  # exact 0/1, for later key classes
    for m, matches in MARKERS.items():
        for k, match in enumerate(matches):
            l1.units += pr.latest_feature(match, f"f_{m}", bos=(k == 0))

    l2 = b.at(2)
    l2.heads += [pr.latest_head(f"latest {m}", f"f_{m}", m) for m in MARKERS]
    l2.units += pr.clamp01_units({"F": 1.0, "Rb": -1.0}, {"open": 1.0})
    for m, flag in (("E", "seen_eq"), ("S", "seen_semi"), ("R", "seen_bar")):
        l2.units += pr.clamp01_units({m: 1.0, "F": -1.0}, {flag: 1.0})

    l3 = b.at(3)
    l3.heads += [
        pr.gather_head("line start", {"F": 1.0}, "Ls", src="Rb"),
        pr.gather_head("innermost group", pr.add({"F": 1.0}, pr.const(-1.0)), "p", src="Lg"),
        *(pr.gather_head(f"prev {c}", prev, f"PM_{c}", src=c) for c in NEXT_CLASSES),
        HeadSpec(
            name="op type",
            query=[pr.scale({"P": 1.0}, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
            key=[{"pos": 1.0}, {"pos2": 1.0}],
            value={dim: {f"typ_{op}": 1.0} for op, dim in OPS.items()},
        ),
        pr.gather_head("borrow sign", {"S": 1.0}, "neg", src="is_lt"),
        pr.gather_head("sign of a", pr.add({"F": 1.0}, pr.const(1.0)), "sa", src="is_minus"),
        pr.gather_head("sign of b", pr.add({"P": 1.0}, pr.const(1.0)), "sb", src="is_minus"),
    ]
    # effective sign of b for + and -: a - b == a + (-b)
    l3.units += [
        pr.relu_unit({"sb": 1.0, "typ_sub": -1.0}, {"eff_sb": 1.0}),
        pr.relu_unit({"typ_sub": 1.0, "sb": -1.0}, {"eff_sb": 1.0}),
        pr.relu_unit({"is_eq": 1.0, "open": -1.0}, {"phase1": 1.0}),
        pr.relu_unit({"open": 1.0, "seen_eq": -1.0}, {"phase2": 1.0}),
        pr.relu_unit({"one": 1.0, "open": -1.0, "is_eq": -1.0}, {"phase4": 1.0}),
    ]

    l4 = b.at(4)
    l4.heads.append(pr.gather_head("(P-Fa)^2", pr.add({"P": 1.0}, pr.scale(FA, -1.0)), "PF2", src="pos2"))
    # Routing: + and - with equal (effective) signs add magnitudes, otherwise subtract them.
    work = {"open": 1.0, "seen_eq": 1.0}
    additive = pr.add(work, {"typ_add": 1.0, "typ_sub": 1.0})  # == 3 when routable
    l4.units += [
        pr.relu_unit(pr.add(additive, {"sa": 1.0, "eff_sb": 1.0}, pr.const(-4.0)), {"act_add": 1.0}),
        pr.relu_unit(pr.add(additive, {"sa": -1.0, "eff_sb": -1.0}, pr.const(-2.0)), {"act_add": 1.0}),
        pr.relu_unit(pr.add(additive, {"sa": 1.0, "eff_sb": -1.0}, pr.const(-3.0)), {"act_sub": 1.0}),
        pr.relu_unit(pr.add(additive, {"sa": -1.0, "eff_sb": 1.0}, pr.const(-3.0)), {"act_sub": 1.0}),
        pr.relu_unit(pr.add(work, {"typ_mul": 1.0}, pr.const(-2.0)), {"act_mul": 1.0}),
    ]
