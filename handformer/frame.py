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
              Lmul '*', Ladd '+'/'-', Lnd non-digit, Lsp ' '
Layer 2 mlp   open = [F > Rb]; seen_eq/semi/bar = [E/S/R > F]
Layer 3 attn  Ls = Rb at F (start of the source line); p = Lg at F-1
              (innermost '(' of the source line, or its start);
              previous-match pointers PM_c = L_c at pos-1;
              operator type at P; sign at S; (P-F)^2 for multiplication
              operand signs sa, sb (a '-' right after '[' / after the operator)
Layer 3 mlp   til_line (the line is the prompt, which ends in "~N"); Lend at layer 4
Layer 3-9     precision N: the digits between '~' and the prompt's '=', spaces skipped
Layer 4 mlp   act_add: + or - with sa == eff_sb (add magnitudes); act_sub: signs differ
              (subtract magnitudes); act_mul: '*'. Each AND is one or two ReLUs.
Layer 8 mlp   phase1 (emit '['), phase2 (copy), phase4 (rewrite); 2 and 4 are off in a
              strip frame (sp or pf, strip.py: a space, or brackets around one number)

Strip frames. A line holding a space, or whose innermost group is a single number, is
not parsed: its frame is ``[ ]`` or ``[()]`` and the line is rewritten without that space
or those brackets (strip.py finds them and writes it):
``\n(1) + 2=[ ](1)+ 2=[ ](1)+2=[()]1+2=[1+2=3;3]3``.
"""

from __future__ import annotations

from . import primitives as pr
from .model import Form, HeadSpec, ModelBuilder

# A '-' is unary (a sign) when the byte before it is not part of a number
# (digit or '.') and not ')'.
# "binary minus" as a 0/1 match form: is_minus + prev_digit + prev_rpar - 1 equals 1
# exactly for a binary '-', 0 for a unary one, <= 0 for any other byte.
# (prev_digit, prev_rpar: layer-1 heads reading byte pos-1; one byte, so at most one is 1.)
# The P and Ladd classes match with it in the layer-1 MLP, so a sign '-' is never an operator.
BINMINUS: Form = {"is_minus": 1.0, "prev_digit": 1.0, "prev_rpar": 1.0, "one": -1.0}

# marker -> the match forms of its byte class (one feature unit each; at most one fires)
MARKERS: dict[str, list[Form]] = {
    "F": [{"is_lbr": 1.0}],
    "Rb": [{"is_rbr": 1.0}],
    "E": [{"is_eq": 1.0}],
    "S": [{"is_semi": 1.0}],
    "R": [{"is_bar": 1.0}],
    "P": [{"is_plus": 1.0}, {"is_star": 1.0}, {"is_slash": 1.0}, {"is_caret": 1.0}, BINMINUS],  # binary ops only
    "Lg": [{"is_lpar": 1.0}, {"is_rbr": 1.0}],
    "Lrp": [{"is_rpar": 1.0}],
    "Lmul": [{"is_star": 1.0}, {"is_slash": 1.0}],  # multiplicative: * and /
    "Lpow": [{"is_caret": 1.0}],  # powers: highest precedence, right-associative
    "Ladd": [{"is_plus": 1.0}, BINMINUS],
    "Lnd": [{"one": 1.0, "is_digit": -1.0, "is_dot": -1.0}],  # not part of a number
    "Ldot": [{"is_dot": 1.0}],
    "Lnz": [{"nz": 1.0}],  # nonzero digit (trailing-zero count of a result)
    "Ptil": [{"is_tilde": 1.0}],  # the prompt's precision marker (0 if absent)
    "Lsp": [{"is_space": 1.0}],  # a space (removed by a space frame)
}
NEXT_CLASSES = ("Lrp", "Lmul", "Ladd", "Lnd")  # classes searched with next_match
OPS = {"add": "is_plus", "sub": "is_minus", "mul": "is_star", "div": "is_slash", "pow": "is_caret"}
N_DIGITS = 4  # precision ~N supports N <= 9999

DIMS = (
    "prev_digit", "prev_rpar", "is_unary", "is_binminus",
    *(f"f_{m}" for m in MARKERS), *MARKERS,
    "open", "seen_eq", "seen_semi", "seen_bar",
    "Ls", "p", *(f"PM_{c}" for c in NEXT_CLASSES), *(f"typ_{op}" for op in OPS),
    "Pf", "neg", "sa", "sb", "PF2",
    "f_E0", "E0", "has_til", "f_Ndg", "Ndg", *(f"qN{k}" for k in range(N_DIGITS)),
    *(f"Nd{k}" for k in range(N_DIGITS)), "N", "til_line", "Lend",
    *(f"act_{op}" for op in OPS), "phase1", "phase2", "phase4",
)
LE: Form = {"F": 1.0, "one": -1.0}  # the source line's '=' (the frame's '[' is right after it)

# Fences of the frame's operands, past their signs: a in (F+sa, P), b in (P+sb, E).
FA: Form = {"F": 1.0, "sa": 1.0}
PB: Form = {"P": 1.0, "sb": 1.0}


def build(b: ModelBuilder) -> None:
    """Write the shared locator into layers 1-4: marker positions, frame flags, phases,
    operand signs sa/sb, precision N, and the act_* flags that gate every operator circuit."""
    # address of the previous byte; at BOS it is -1 and the gather clamps onto BOS (all zeros)
    prev = pr.add({"pos": 1.0}, pr.const(-1.0))
    # Layer 1. Heads: is the previous byte part of a number, or ')'? (unary vs binary '-').
    # MLP: is_unary / is_binminus, and the tags f_<m> that the layer-2 latest heads maximise.
    l1 = b.at(1)
    l1.heads += [
        # prev_digit = is_digit + is_dot at pos-1 (a gather, score -beta(k-(pos-1))^2); written as a
        # raw HeadSpec because gather_head takes one source dim. The two are exclusive, so 0/1.
        HeadSpec(  # "digit" here means part of a number: a digit or '.'
            name="prev byte in a number?",
            query=[pr.scale(prev, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
            key=[{"pos": 1.0}, {"pos2": 1.0}],
            value={"is_digit": {"prev_digit": 1.0}, "is_dot": {"prev_digit": 1.0}},
        ),
        pr.gather_head("prev byte ')'?", prev, "prev_rpar", src="is_rpar"),
    ]
    # Same-layer MLP sees these heads. is_unary = ReLU(is_minus - prev_digit - prev_rpar): 1 only for
    # a sign '-'; read by the parser (sign of a at a_lo, layer 9; '-' before A, layer 12).
    # is_binminus is read as the "first + or - after p" key class (expr layer 5).
    l1.units.append(pr.relu_unit({"is_minus": 1.0, "prev_digit": -1.0, "prev_rpar": -1.0},
                                 {"is_unary": 1.0},
                                 label="'is_unary': a '-' whose previous byte is not part of a number and not ')', so it is a sign, not subtraction"))
    l1.units.append(pr.relu_unit(BINMINUS, {"is_binminus": 1.0},
                                 label="'is_binminus': a '-' right after a digit, '.' or ')' is binary subtraction; exact 0/1 so later key classes can use it"))  # exact 0/1, for later key classes
    # f_m = 2(pos+1) on a byte of class m, else 0; BOS adds 1 once per class (bos only for the
    # first form), so "none yet" reads position 0 and a BOS match (3) never ties pos 1 (4).
    for m, matches in MARKERS.items():
        for k, match in enumerate(matches):
            l1.units += pr.latest_feature(match, f"f_{m}", bos=(k == 0))

    # the prompt's '=' is the earliest '=' in the sequence
    # f_E0 = BIG - pos on an '=', else 0: the latest head's max then lands on the earliest '='
    l1.units.append(pr.earliest_if({"is_eq": 1.0}, 1.0, "f_E0"))

    # Layer 2. Heads: each marker m = position of the latest byte of its class (the key with the
    # largest f_m, read with beta 40; BOS = 0 if none); E0 = the prompt's '='. MLP: precision
    # offsets tN_k and the frame flags open / seen_*, computed from this layer's heads.
    l2 = b.at(2)
    l2.heads += [pr.latest_head(f"latest {m}", f"f_{m}", m) for m in MARKERS]
    l2.heads.append(pr.latest_head("prompt '='", "f_E0", "E0"))
    # Precision N: the digits between '~' and the prompt's '=' (N <= 9999), read in layers 3-9.
    # has_til = [Ptil >= 1] (Ptil reads BOS = 0 without a '~').
    l2.units += pr.clamp01_units({"Ptil": 1.0}, {"has_til": 1.0})
    # open = [F - Rb >= 1]: a '[' after the latest ']'. seen_eq/semi/bar = [E/S/R - F >= 1]: that
    # marker was already written inside the current frame. Robust steps give exact 0/1 flags.
    l2.units += pr.clamp01_units({"F": 1.0, "Rb": -1.0}, {"open": 1.0})
    for m, flag in (("E", "seen_eq"), ("S", "seen_semi"), ("R", "seen_bar")):
        l2.units += pr.clamp01_units({m: 1.0, "F": -1.0}, {flag: 1.0})

    # Layer 3. Heads: pointers read at layer-2 markers (line start, innermost group, previous
    # matches, frame operator, signs, precision digits). MLP: N and the three phase flags.
    l3 = b.at(3)
    l3.heads += [
        # Ls = latest ']' as seen from the frame's '[' (the previous frame's ']', or BOS);
        # the source line lies strictly between Ls and Le = F-1 (its '=').
        pr.gather_head("line start", {"F": 1.0}, "Ls", src="Rb"),
        # p = latest '(' or ']' at Le: the innermost group's '(' (no '(' follows it), else Ls
        pr.gather_head("innermost group", pr.add({"F": 1.0}, pr.const(-1.0)), "p", src="Lg"),
        # PM_c = latest c-match strictly before this position: the per-key pointer that
        # next_match_head compares (parser layers 4-5)
        *(pr.gather_head(f"prev {c}", prev, f"PM_{c}", src=c) for c in NEXT_CLASSES),
        # The frame's own operator: the latest binary operator at the frame's '='. Stable even
        # when a frame body writes operators (a power frame writes "a*a^n").
        pr.gather_head("frame operator", {"E": 1.0}, "Pf", src="P"),
        # neg = [end marker at S is '<']: the subtraction's result is negative (read by subtraction)
        pr.gather_head("borrow sign", {"S": 1.0}, "neg", src="is_lt"),
        # sa = [byte after '[' is '-'], sb = [byte after the operator P is '-']: the operands' signs.
        # Circuits work on magnitudes, fenced at F+sa and P+sb (FA, PB above).
        pr.gather_head("sign of a", pr.add({"F": 1.0}, pr.const(1.0)), "sa", src="is_minus"),
        pr.gather_head("sign of b", pr.add({"P": 1.0}, pr.const(1.0)), "sb", src="is_minus"),
    ]
    # the prompt line carries "~N" after the expression; no rewrite copies it
    # til_line = [Ptil > Ls]: the '~' lies in the source line, so this is the prompt line
    l3.units += pr.clamp01_units({"Ptil": 1.0, "Ls": -1.0}, {"til_line": 1.0})

    # Precision digits. A digit of N is a digit with a '~' before it: is_digit AND has_til (the
    # prompt has one '~', so the expression's digits, before it, never match). Layer 3 tags them,
    # layer 4 finds the latest one (Ndg; BOS = 0, whose val is 0, when none is left). Then right to
    # left, one per layer: qN_0 = Ndg at the prompt's '=' - 1, qN_k = Ndg at qN_(k-1) - 1; spaces
    # are skipped because they are not digits. The next layer reads Nd_k = val at qN_k, the 10^k
    # digit of N (0 where N has none).
    b.at(3).units += pr.latest_feature({"is_digit": 1.0, "has_til": 1.0, "one": -1.0}, "f_Ndg")
    b.at(4).heads.append(pr.latest_head("latest digit of N", "f_Ndg", "Ndg"))
    for k in range(N_DIGITS):
        before = pr.add({"E0": 1.0} if k == 0 else {f"qN{k - 1}": 1.0}, pr.const(-1.0))
        b.at(5 + k).heads.append(pr.gather_head(f"precision digit {k} position", before, f"qN{k}", src="Ndg"))
        b.at(6 + k).heads.append(pr.gather_head(f"precision digit {k}", {f"qN{k}": 1.0}, f"Nd{k}"))
    # N = has_til * sum_k 10^k Nd_k (0 without '~'); read by the parser's division pads (layer 14)
    b.at(5 + N_DIGITS).units += pr.times_flag_units({f"Nd{k}": 10.0 ** k for k in range(N_DIGITS)},
                                                    {"has_til": 1.0}, {"N": 1.0})

    # Layer 4. Heads: (P-Fa)^2 for multiplication and the frame operator's one-hot type.
    # MLP: the act_* routing flags, read by every gated operator-circuit unit (arith.py).
    l4 = b.at(4)
    # Lend: where the source line's expression ends. Le, or the '~' on the prompt line
    # (Lend = Le + til_line * (Ptil - Le)). Read by the rewrite (expr) and the space frame.
    l4.units.append(pr.relu_unit(LE, {"Lend": 1.0},
                                 label="'Lend' base: Le, where the line's expression ends; the prompt line's '~N' correction is added next"))
    l4.units += pr.times_flag_units({"Ptil": 1.0, "F": -1.0, "one": 1.0}, {"til_line": 1.0}, {"Lend": 1.0})
    l4.heads += [
        # square table: pos2 gathered at x = P - (F+sa) is x^2, x = (digits of |a|) + 1
        # (read by multiplication.py's S2)
        pr.gather_head("(P-Fa)^2", pr.add({"P": 1.0}, pr.scale(FA, -1.0)), "PF2", src="pos2"),
        # typ_<op> = [byte at Pf is <op>]: one gather at Pf, a value entry per operator byte
        HeadSpec(
            name="frame operator type",
            query=[pr.scale({"Pf": 1.0}, 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
            key=[{"pos": 1.0}, {"pos2": 1.0}],
            value={dim: {f"typ_{op}": 1.0} for op, dim in OPS.items()},
        ),
    ]
    # Routing. For + and -, b's effective sign is sb XOR [op is -]: equal to sa -> add
    # magnitudes, different -> subtract them. Each case is one exact ReLU over 0/1 flags.
    # A case is ReLU(plus|minus ± sa ± sb - c): +1 on a flag that must be 1, -1 on one that must
    # be 0, c = 2 + (number of flags that must be 1); only the exact pattern reaches 1, others <= 0.
    # work = open + seen_eq == 2 while the frame's working is being written (phase 3)
    work = {"open": 1.0, "seen_eq": 1.0}
    plus = pr.add(work, {"typ_add": 1.0})  # == 3 for a '+' frame
    minus = pr.add(work, {"typ_sub": 1.0})  # == 3 for a '-' frame
    l4.units += [
        pr.relu_unit(pr.add(plus, {"sa": 1.0, "sb": 1.0}, pr.const(-4.0)), {"act_add": 1.0},
                     label="'act_add' case: working a '+' frame with a and b both negative (sa=sb=1): add magnitudes"),
        pr.relu_unit(pr.add(plus, {"sa": -1.0, "sb": -1.0}, pr.const(-2.0)), {"act_add": 1.0},
                     label="'act_add' case: working a '+' frame with a and b both positive: add magnitudes"),
        pr.relu_unit(pr.add(minus, {"sa": 1.0, "sb": -1.0}, pr.const(-3.0)), {"act_add": 1.0},
                     label="'act_add' case: '-' frame, a negative, b positive: b's effective sign equals a's, add magnitudes"),
        pr.relu_unit(pr.add(minus, {"sa": -1.0, "sb": 1.0}, pr.const(-3.0)), {"act_add": 1.0},
                     label="'act_add' case: '-' frame, a positive, b negative (a - -b): effective signs equal, add magnitudes"),
        pr.relu_unit(pr.add(plus, {"sa": 1.0, "sb": -1.0}, pr.const(-3.0)), {"act_sub": 1.0},
                     label="'act_sub' case: '+' frame, a negative, b positive: signs differ, subtract magnitudes"),
        pr.relu_unit(pr.add(plus, {"sa": -1.0, "sb": 1.0}, pr.const(-3.0)), {"act_sub": 1.0},
                     label="'act_sub' case: '+' frame, a positive, b negative: signs differ, subtract magnitudes"),
        pr.relu_unit(pr.add(minus, {"sa": 1.0, "sb": 1.0}, pr.const(-4.0)), {"act_sub": 1.0},
                     label="'act_sub' case: '-' frame, both a and b negative: effective signs differ, subtract magnitudes"),
        pr.relu_unit(pr.add(minus, {"sa": -1.0, "sb": -1.0}, pr.const(-2.0)), {"act_sub": 1.0},
                     label="'act_sub' case: '-' frame, both a and b positive: plain subtraction of magnitudes"),
        pr.relu_unit(pr.add(work, {"typ_mul": 1.0}, pr.const(-2.0)), {"act_mul": 1.0},
                     label="'act_mul': AND of open frame, past its '=', frame operator '*': enables the multiplication circuit"),
        pr.relu_unit(pr.add(work, {"typ_div": 1.0}, pr.const(-2.0)), {"act_div": 1.0},
                     label="'act_div': AND of open frame, past its '=', frame operator '/': enables the division circuit"),
        pr.relu_unit(pr.add(work, {"typ_pow": 1.0}, pr.const(-2.0)), {"act_pow": 1.0},
                     label="'act_pow': AND of open frame, past its '=', frame operator '^': enables the power-rewrite circuit"),
    ]

    # Layer 8 MLP. Phases (exact: ReLUs of 0/1 flags); read by the parser's byte votes in layer 22.
    # In a strip frame (sp or pf = 1, strip.py; never both) there is no redex to copy and no
    # result to write: phase2 and phase4 are off and strip.py writes the frame and the line.
    b.at(8).units += [
        pr.relu_unit({"is_eq": 1.0, "open": -1.0}, {"phase1": 1.0},
                     label="'phase1': at a line-ending '=' with no frame open, so the next byte is '[' (start a new frame)"),
        pr.relu_unit({"open": 1.0, "seen_eq": -1.0, "sp": -1.0, "pf": -1.0}, {"phase2": 1.0},
                     label="'phase2': a frame is open but has no '=' yet and is not a strip frame, so the model is copying the redex into it"),
        pr.relu_unit({"one": 1.0, "open": -1.0, "is_eq": -1.0, "sp": -1.0, "pf": -1.0}, {"phase4": 1.0},
                     label="'phase4': frame closed, not on '=', not a strip frame: the model is rewriting the expression line after ']'"),
    ]
