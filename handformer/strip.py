"""Strip frames: bytes with nothing to compute are removed from the line, in the weights.

Two kinds, each a frame that removes bytes and computes nothing:

  space        the line holds a space: the frame is ``[ ]`` and the next line is the line
               without its last space
  brackets     the line's innermost group is a single number, ``(5)``: the frame is ``[()]``
               and the next line is the line without that pair of brackets

    \\n(1) + 2=[ ](1)+ 2=[ ](1)+2=[()]1+2=[1+2=3;3]3

One space or one pair goes per frame, as one operator goes per frame. Removing every space
in one rewrite would need, at each output byte, the number of spaces before its source byte:
a count, which attention gives only as a fraction of the position. Removing one is an
offset: copy the line, reading one byte further from the removed byte on (two for the
brackets). Spaces go first (a group is only judged once the line has none), so "1 2+3" is
12+3, exactly as if the spaces were never typed.

Finding them (every pointer read is the frame's own, so the flags hold from the frame's
'[' to the next line's '='):
  sp_at, sp     the latest space before Lend; sp = [it is in the line]
  pf            the innermost group p..c is "(" number ")": the byte at p is '(', the
                byte at c is ')', and no binary operator lies between (the latest one
                before c is before p); and the line has no space
  cn            the number is negative and the '(' follows a sign: -(-5) is 5, so both
                signs go too (as the parser cancels -(-v) for a result); without this,
                -(-5)+1 would become --5+1, which the parser reads as -(-5+1)

Writing (frame.py switches the parser's phases 2 and 4 off while sp or pf is 1;
r = pos - Rb, the bytes of the new line written so far, t = Ls + 1 + r):
  at '['        ' ' (space) or '(' (brackets): the body names what goes
  in the body   ']' after the ' '; ')' after the '(' and then ']'
  after ']'     the byte at st_src = t + the bytes skipped so far: [t >= sp_at] for a
                space, [t >= p] + [t+1 >= c] for brackets, 3[t >= p-1] + [t+3 >= c]
                when the signs cancel
  st_src = Lend '=' if the line still has work (another space, an operator, or a '(' in
                the bytes just written), else newline: the line is the answer

The copy is the parser's own rewrite copy (expr layers 21-22): in a strip frame its source
position src4 is st_src, and its one-hot Y4 also covers the space, so the line's other
spaces are copied on for later frames to remove. Copying stops by itself at Lend: the byte
there is '=' or '~', which has no copy vote.

Layers 4-11, 21 and 22, beside the parser (frame.py provides Lsp, Lend; expr.py p and c).
"""

from __future__ import annotations

from . import common as cm
from . import primitives as pr
from .expr import NAMES
from .model import ModelBuilder

DIMS = (
    "sp_at", "sp", "sp_skip", "sp_more_at", "sp_more",
    "lp_p", "rp_c", "Pc", "pg", "pf", "um_p", "mi_p", "cn", "pf_skip",
    "st_op", "st_lp", "st_cont", "st_src", "st_end",
)
OUT_DIMS = ("out_space", "out_close")


def build(b: ModelBuilder) -> None:
    """Find a space or a bracketed single number, then write its strip frame and the line."""
    # t = Ls + 1 + r: the source byte copied when nothing has been skipped yet
    t = {"pos": 1.0, "Rb": -1.0, "Ls": 1.0, "one": 1.0}
    strip = {"sp": 1.0, "pf": 1.0}  # 1 in a strip frame (the two never hold together)
    # rewriting a strip frame's line: closed frame, not on the '=' that ends the new line
    rewriting = pr.add(strip, {"open": -1.0, "is_eq": -1.0})

    # Layer 4 heads: lp_p = [byte at p is '('] (p is the line start when it has no group);
    # um_p = [a sign '-' before it]; mi_p = [a '-' after it] (a sign: it follows '(').
    b.at(4).heads += [
        pr.gather_head("group opens with (", {"p": 1.0}, "lp_p", src="is_lpar"),
        pr.gather_head("sign before the group", pr.add({"p": 1.0}, pr.const(-1.0)), "um_p", src="is_unary"),
        pr.gather_head("sign inside the group", pr.add({"p": 1.0}, pr.const(1.0)), "mi_p", src="is_minus"),
    ]

    # Layer 5. Head: sp_at = the latest space before Lend, read at Lend - 1. MLP: sp = [sp_at > Ls]:
    # without a space in the line sp_at is at or before Ls (BOS, or a space of an earlier line).
    l5 = b.at(5)
    l5.heads.append(pr.gather_head("space in the line", pr.add({"Lend": 1.0}, pr.const(-1.0)), "sp_at", src="Lsp"))
    l5.units += pr.clamp01_units({"sp_at": 1.0, "Ls": -1.0}, {"sp": 1.0})

    # Layer 6. Heads: Pc = the latest binary operator before c; rp_c = [byte at c is ')'] (c is
    # Le, an '=', when the group has no ')'); sp_more_at = the latest space before sp_at.
    # MLP: sp_skip = [t >= sp_at] in a space frame; pg = [Pc < p], no operator inside p..c;
    # sp_more = [another space is left in the line].
    l6 = b.at(6)
    l6.heads += [
        pr.gather_head("operator before c", pr.add({"c": 1.0}, pr.const(-1.0)), "Pc", src="P"),
        pr.gather_head("group closes with )", {"c": 1.0}, "rp_c", src="is_rpar"),
        pr.gather_head("another space in the line", pr.add({"sp_at": 1.0}, pr.const(-1.0)), "sp_more_at", src="Lsp"),
    ]
    l6.units += pr.clamp01_units(pr.gated(pr.add(t, {"sp_at": -1.0}, pr.const(1.0)), {"sp": 1.0}), {"sp_skip": 1.0})
    l6.units += pr.clamp01_units({"p": 1.0, "Pc": -1.0}, {"pg": 1.0})
    l6.units += pr.clamp01_units({"sp_more_at": 1.0, "Ls": -1.0}, {"sp_more": 1.0})

    # Layer 7 MLP: pf = lp_p AND rp_c AND pg AND NOT sp (one exact ReLU of 0/1 flags);
    # st_op = [P > Rb], a binary operator among the bytes written since ']' (they have no
    # space, so a '-' there is binary only after a number or ')'); st_lp = [Lg > Rb], a '('
    # among them (Lg is the latest '(' or ']', and the latest ']' is Rb itself).
    l7 = b.at(7)
    l7.units.append(pr.relu_unit({"lp_p": 1.0, "rp_c": 1.0, "pg": 1.0, "sp": -1.0, "one": -2.0}, {"pf": 1.0},
                                 label="'pf': the innermost group is '(' number ')' (opens with '(', closes with ')', no operator inside) and the line has no space: a bracket strip frame"))
    l7.units += pr.clamp01_units({"P": 1.0, "Rb": -1.0}, {"st_op": 1.0})
    l7.units += pr.clamp01_units({"Lg": 1.0, "Rb": -1.0}, {"st_lp": 1.0})

    # Layer 8 MLP: cn = pf AND um_p AND mi_p (the signs cancel); st_cont = [sp_more + st_op +
    # st_lp >= 1]: the new line needs another frame.
    l8 = b.at(8)
    l8.units.append(pr.relu_unit({"pf": 1.0, "um_p": 1.0, "mi_p": 1.0, "one": -2.0}, {"cn": 1.0},
                                 label="'cn': a bracket frame around a negative number right after a sign, -(-5): both signs are dropped with the brackets"))
    l8.units += pr.clamp01_units({"sp_more": 1.0, "st_op": 1.0, "st_lp": 1.0}, {"st_cont": 1.0})

    # Layer 9 MLP: pf_skip, the bytes a bracket frame has skipped so far. Without cn: [t >= p]
    # (the '(') + [t + 1 >= c] (the ')', one byte later in t). With cn: 3[t >= p - 1] (the
    # '-(-' run) + [t + 3 >= c]. Each step is gated by its case (pf - cn = 1, or cn = 1).
    l9 = b.at(9)
    plain = {"pf": 1.0, "cn": -1.0}
    l9.units += pr.clamp01_units(pr.gated(pr.add(t, {"p": -1.0}, pr.const(1.0)), plain), {"pf_skip": 1.0})
    l9.units += pr.clamp01_units(pr.gated(pr.add(t, {"c": -1.0}, pr.const(2.0)), plain), {"pf_skip": 1.0})
    l9.units += pr.clamp01_units(pr.gated(pr.add(t, {"p": -1.0}, pr.const(2.0)), {"cn": 1.0}), {"pf_skip": 3.0})
    l9.units += pr.clamp01_units(pr.gated(pr.add(t, {"c": -1.0}, pr.const(4.0)), {"cn": 1.0}), {"pf_skip": 1.0})

    # Layer 10 MLP: st_src = t + sp_skip + pf_skip, the byte to copy next.
    b.at(10).units.append(pr.relu_unit(pr.add(t, {"sp_skip": 1.0, "pf_skip": 1.0}), {"st_src": 1.0},
                                       label="'st_src': the byte a strip frame's rewrite copies next, Ls+1+r plus the bytes skipped so far"))
    # Layer 11 MLP: st_end = [st_src == Lend], the line is done.
    b.at(11).units += pr.hat_units({"st_src": 1.0, "Lend": -1.0}, 0.0, {"st_end": 1.0})

    # Layer 21 MLP: src4 += strip * st_src. The parser's own terms of src4 are off in a strip
    # frame (expr layer 21), so the layer-22 rewrite copy reads Y4 at st_src.
    b.at(21).units += pr.times_flag_units({"st_src": 1.0}, strip, {"src4": 1.0})

    # Layer 22 MLP: the votes. Each is an exact AND of 0/1 flags, like the parser's.
    l22 = b.at(22)
    l22.units += [
        pr.relu_unit({"is_lbr": 1.0, "sp": 1.0, "one": -1.0}, {"out_space": 1.0},
                     label="emit ' ' at the '[' of a space frame (the line has a space): the frame's body is the space it removes"),
        pr.relu_unit({"is_space": 1.0, "open": 1.0, "sp": 1.0, "one": -2.0}, {"out_close": 1.0},
                     label="emit ']' after the space frame's ' ': close it, the line without that space follows"),
        pr.relu_unit({"is_lbr": 1.0, "pf": 1.0, "one": -1.0}, {"O_is_lpar": 1.0},
                     label="emit '(' at the '[' of a bracket frame (the innermost group is one number): the body names the brackets it removes"),
        pr.relu_unit({"is_lpar": 1.0, "open": 1.0, "pf": 1.0, "one": -2.0}, {"O_is_rpar": 1.0},
                     label="emit ')' after the bracket frame's '('"),
        pr.relu_unit({"is_rpar": 1.0, "open": 1.0, "pf": 1.0, "one": -2.0}, {"out_close": 1.0},
                     label="emit ']' after the bracket frame's '()': close it, the line without those brackets follows"),
        pr.relu_unit(pr.add(rewriting, {"st_end": 1.0, "st_cont": 1.0}, pr.const(-2.0)), {"out_eq": 1.0},
                     label="emit '=' at the end of a strip frame's line when it has more work (a space, an operator or a '(')"),
        pr.relu_unit(pr.add(rewriting, {"st_end": 1.0, "st_cont": -1.0}, pr.const(-1.0)), {"out_nl": 1.0},
                     label="emit newline at the end of a strip frame's line with nothing left to do: it is the answer"),
    ]
    for d in NAMES.values():
        l22.units.append(pr.relu_unit(pr.add(rewriting, {f"Y4_{d}": 1.0}, pr.const(-1.0)), {f"O_{d}": 1.0},
                                      label=f"emit byte {d} in a strip frame's rewrite: AND of the rewrite copy's byte Y4_{d} and strip frame closed"))
    l22.units.append(pr.relu_unit(pr.add(rewriting, {"Y4_is_space": 1.0}, pr.const(-1.0)), {"out_space": 1.0},
                                  label="emit ' ' in a strip frame's rewrite: another space of the line, copied for a later frame to remove"))

    # ' ' and ']' are written by the frame's own flags, which no other vote shares a position
    # with (phase2 and phase4 are off in a strip frame): CONTROL_LOGIT, like '[' and '='.
    b.logit("out_space", ord(" "), cm.CONTROL_LOGIT)
    b.logit("out_close", cm.FRAME_CLOSE, cm.CONTROL_LOGIT)
