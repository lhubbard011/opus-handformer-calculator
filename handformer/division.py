"""Division circuit, inside a frame ``[ A / B = <grid> ; answer ]`` (integers).

The parser has already scaled the operands (a*10^(frac_b+N), b*10^frac_a), so
this circuit computes the integer quotient floor(|A| / |B|) by long division;
the rewrite places the decimal point (N digits) afterwards.

The working is a uniform grid of rows with stride s = L + 2 (L = len B):

  rows 0..8   multiples M_1..M_9 of B: L+1 digits LSB first, then ','.
              M_k = M_{k-1} + B, digit by digit with addition's recovered carry.
  row 9       q_0 = 0 and R_0 = 0: a '0' and L+1 zeros (the starting remainder)
  row 9+i     q_i, then R_i (L+1 digits LSB first), for i = 1..len A:
                R'_i = 10*R_{i-1} + a_i          (a shift: R'[j] = R_{i-1}[j-1], R'[0] = a_i)
                q_i  = #{k : R'_i >= M_k}        (k = 1..9)
                R_i  = R'_i - M_{q_i}            (subtraction's recovered borrow)
  then ';' and the answer: the q column from its first nonzero digit
  (row 9+len A + 1 holds no row: that is where ';' goes).

How q_i is decided in one pass: every remainder digit R_{i-1}[c-1], as it is
written at column c, compares itself with M_k[c] for all nine k and stores
gt_k / lt_k. At the q_i position, nine latest-match heads find, for each k,
the most significant differing digit of R'_i vs M_k (falling back to the
row start when all digits above the units agree), and read its gt/lt flags;
the units digit a_i vs M_k[0] = (k*b_0) mod 10 settles ties. Then
q = sum_k [R'_i >= M_k].

Coordinates: the next byte goes to x = q - E cells after '='; row = floor(x/s)
(floor-division head), col = x - row*s (square-table product). Keys use their
own coordinates (x - 1) the same way.

Sign: sa XOR sb, emitted only for a nonzero quotient. B = 0 is undefined.

Global layers 4..12.
"""

from __future__ import annotations

from . import common as cm
from . import primitives as pr
from .frame import FA, PB
from .model import HeadSpec, Layout, ModelBuilder

K = range(1, 10)
PRIVATE = (
    "S2", "X2", "SX2", "XS2", "SXS2", "NS2", "N2", *(f"Bo{v}" for v in range(10)),
    "tM0", "tM1", "row", "rowS",
    "RS2", "R2", "RSS2", "RS2S",
    "rowle8", "row9", "rowge10", "col0", "col1", "col2", "colsep", "endrow",
    "Srowge9", "Srow9", "Srowge10", "Scol0", "Scolge1", "tBc", "tBc1",
    "Mp0", "Mp1", "Bc", "Bc1", "qv", "Rp", "Rp1", "ai", *(f"m{k}" for k in K),
    *(f"gt{k}" for k in K), *(f"lt{k}" for k in K), "carry", "Rj", "Rj1",
    "QS2", "Q2", *(f"dkey{k}" for k in K), "rskey", "zq", *(f"c0{k}" for k in K), "tMq", "tMq1",
    *(f"GT{k}" for k in K), *(f"LT{k}" for k in K), "MqJ", "MqJ1", "Zq",
    "qd", "br", "found_q", "sgn", "start", "TS2", "T2", "src", "Av",
)

S_ = {"E": 1.0, "P": -1.0, "sb": -1.0, "one": 1.0}  # s = L + 2, L = E - (P+sb) - 1
NA = pr.add({"P": 1.0}, pr.scale(FA, -1.0), pr.const(-1.0))  # len A
X = {"pos": 1.0, "E": -1.0}  # next byte's cell index after '='
XS = {"pos": 1.0, "E": -1.0, "one": -1.0}  # this byte's own cell index
S2 = {"S2": 1.0}
SX = pr.product_form("SX2", "X2", S2)
SXS = pr.product_form("SXS2", "XS2", S2)
COL = pr.add(X, pr.scale(pr.product_form("RS2", "R2", S2), -1.0))
COLS = pr.add(XS, pr.scale(pr.product_form("RSS2", "RS2S", S2), -1.0))
W = {"E": 1.0, "one": 1.0}  # first working cell
LQ = pr.add(W, pr.scale(S_, 9.0), pr.product_form("NS2", "N2", S2))  # last quotient digit: W + (9+nA)s
IN_WORK = {"seen_eq": 1.0, "seen_semi": -1.0}


def _ruler(name: str, x: dict, sx: dict, dst: str) -> HeadSpec:
    sxc = pr.add(sx, pr.scale(S2, -0.5), pr.scale(S_, 0.5))  # s*(x - (s-1)/2)
    return pr.floor_div_head(name, S2, sxc, dst)


def builder() -> ModelBuilder:
    """The long-division circuit at global layers 4..12, on private dims.

    Grid geometry (L4-6), the next cell's operands (L7), the q_i vote (L8-9),
    the grid digits / ',' / ';' (L10), then the answer copied from the q column
    (L11-12). Writes D0..D9, emit_comma, work_end, emit_minus, frame_end; arith
    gates those writes by act_div.
    """
    lay = Layout()
    lay.add(*PRIVATE)
    b = ModelBuilder(lay)
    working = {"seen_semi": 1.0}

    # Layer 4: squares (pos2 gathered at s, x, x+s, ...) for the products row*s, x*s, nA*s
    # that place cells, and b_0 (B's units digit, at E-1) as a one-hot for M_k[0].
    l4 = b.at(4)
    l4.heads += [
        pr.gather_head("s^2", S_, "S2", src="pos2"),
        pr.gather_head("x^2", X, "X2", src="pos2"),
        pr.gather_head("(x+s)^2", pr.add(X, S_), "SX2", src="pos2"),
        pr.gather_head("xs^2", XS, "XS2", src="pos2"),
        pr.gather_head("(xs+s)^2", pr.add(XS, S_), "SXS2", src="pos2"),
        pr.gather_head("(nA+s)^2", pr.add(NA, S_), "NS2", src="pos2"),
        pr.gather_head("nA^2", NA, "N2", src="pos2"),
        HeadSpec(name="b_0 one-hot",
                 query=[pr.scale(pr.add({"E": 1.0}, pr.const(-1.0)), 2 * pr.GATHER_BETA), pr.const(-pr.GATHER_BETA)],
                 key=[{"pos": 1.0}, {"pos2": 1.0}],
                 value={f"oh{v}": {f"Bo{v}": 1.0} for v in range(10)}),
    ]
    # previous multiples row, same column / column-1 (M_0 = 0: clamp onto '=')
    l4.units += [
        pr.relu_unit(pr.add({"pos": 1.0, "E": -1.0}, pr.scale(S_, -1.0), pr.const(1.0)), {"tM0": 1.0},
                    label="'tM0' = x - s + 1: offset after '=' of the previous multiple's digit at this column; "
                          "clamps to 0 ('=', M_0 = 0)"),
        pr.relu_unit(pr.add({"pos": 1.0, "E": -1.0}, pr.scale(S_, -1.0)), {"tM1": 1.0},
                    label="'tM1' = x - s: offset after '=' of the previous multiple's digit at column - 1 (for the "
                          "carry); clamps to 0 ('=', M_0 = 0)"),
    ]

    # Layer 5: row = floor(x/s) for the next cell, rowS = floor((x-1)/s) for this byte's own
    # cell (ruler heads; they need only s^2 and s*x from layer 4).
    l5 = b.at(5)
    l5.heads += [_ruler("row = floor(x/s)", X, SX, "row"), _ruler("own row", XS, SXS, "rowS")]

    # Layer 6: col = x - row*s (COL; COLS for this byte), then exact 0/1 region flags.
    l6 = b.at(6)
    l6.heads += [
        pr.gather_head("(row+s)^2", pr.add({"row": 1.0}, S_), "RS2", src="pos2"),
        pr.gather_head("row^2", {"row": 1.0}, "R2", src="pos2"),
        pr.gather_head("(rowS+s)^2", pr.add({"rowS": 1.0}, S_), "RSS2", src="pos2"),
        pr.gather_head("rowS^2", {"rowS": 1.0}, "RS2S", src="pos2"),
    ]
    row = {"row": 1.0}
    # Next cell: rowle8 = [row <= 8] (multiples), row9 = the zero row, rowge10 = q|R rows,
    # endrow = [row == 10 + nA] (the row after the last q, where ';' goes); col0/1/2 pick the
    # q cell and R[0], R[1]; colsep = [col == s-1], the ',' closing a multiples row.
    l6.units += pr.clamp01_units(pr.add(pr.const(9.0), pr.scale(row, -1.0)), {"rowle8": 1.0})
    l6.units += pr.hat_units(row, 9.0, {"row9": 1.0})
    l6.units += pr.clamp01_units(pr.add(row, pr.const(-9.0)), {"rowge10": 1.0})
    l6.units += pr.hat_units(pr.add(row, pr.const(-10.0), pr.scale(NA, -1.0)), 0.0, {"endrow": 1.0})
    l6.units += pr.hat_units(COL, 0.0, {"col0": 1.0})
    l6.units += pr.hat_units(COL, 1.0, {"col1": 1.0})
    l6.units += pr.hat_units(COL, 2.0, {"col2": 1.0})
    l6.units += pr.hat_units(pr.add(COL, pr.scale(S_, -1.0), pr.const(1.0)), 0.0, {"colsep": 1.0})
    # This byte (S*, own row/col): which cells are comparison keys (L7 gt/lt, L8 dkey/rskey)
    # and q cells for the answer start (L8 zq); Scolge1 = a digit cell, not the row start.
    l6.units += pr.clamp01_units(pr.add({"rowS": 1.0}, pr.const(-8.0)), {"Srowge9": 1.0})
    l6.units += pr.hat_units({"rowS": 1.0}, 9.0, {"Srow9": 1.0})
    l6.units += pr.clamp01_units(pr.add({"rowS": 1.0}, pr.const(-9.0)), {"Srowge10": 1.0})
    l6.units += pr.hat_units(COLS, 0.0, {"Scol0": 1.0})
    l6.units += pr.clamp01_units(COLS, {"Scolge1": 1.0})
    # B[col], B[col-1] at E-1-col, E-col (clamped onto b's left fence, value 0)
    l6.units += [
        pr.relu_unit(pr.add({"E": 1.0}, pr.const(-1.0), pr.scale(COL, -1.0), pr.scale(PB, -1.0)), {"tBc": 1.0},
                    label="'tBc': offset from b's left fence of B[col] (E-1-col); past B's top digit it clamps to the "
                          "fence, value 0"),
        pr.relu_unit(pr.add({"E": 1.0}, pr.scale(COL, -1.0), pr.scale(PB, -1.0)), {"tBc1": 1.0},
                    label="'tBc1': offset from b's left fence of B[col-1] (E-col, for the carry); clamps to the "
                          "fence, value 0"),
    ]

    # Layer 7: gather the next cell's operands; cell c sits at position E+1+c and this byte at
    # cell x-1, so pos - s is one row up and one column left of the next cell (R_{i-1}[j-1] = R'[j]),
    # pos+1-col is this row's q cell, and a_i (row 9+i) is A's i-th digit at FA + i.
    l7 = b.at(7)
    l7.heads += [
        pr.gather_head("M_k[col]", {"E": 1.0, "tM0": 1.0}, "Mp0"),
        pr.gather_head("M_k[col-1]", {"E": 1.0, "tM1": 1.0}, "Mp1"),
        pr.gather_head("B[col]", pr.add(PB, {"tBc": 1.0}), "Bc"),
        pr.gather_head("B[col-1]", pr.add(PB, {"tBc1": 1.0}), "Bc1"),
        pr.gather_head("q of this row", pr.add({"pos": 1.0}, pr.const(1.0), pr.scale(COL, -1.0)), "qv"),
        pr.gather_head("R_{i-1}[j-1]", pr.add({"pos": 1.0}, pr.scale(S_, -1.0)), "Rp"),
        pr.gather_head("R_{i-1}[j-2]", pr.add({"pos": 1.0}, pr.scale(S_, -1.0), pr.const(-1.0)), "Rp1"),
        pr.gather_head("a_i", pr.add(FA, row, pr.const(-9.0)), "ai"),
        *(pr.gather_head(f"M_{k}[own col]", pr.add(W, pr.scale(S_, k - 1.0), COLS), f"m{k}") for k in K),
    ]
    # This remainder digit vs multiple k at the next row's alignment. Only remainder
    # cells (column >= 1) carry flags: the row-start cell is the comparison heads'
    # "all upper digits equal" fallback and must read gt = lt = 0.
    # Byte R_{i-1}[c-1] at column c becomes R'_i[c] one row later, so it compares with M_k[c]
    # (m_k): gt_k = [val - m_k >= 1], lt_k = [m_k - val >= 1]. rem_cell = -BIG*(1 - Scolge1)
    # forces both to 0 at column 0. Read in L8 (dkey_k) and by L9's comparison heads.
    rem_cell = pr.add(pr.scale({"Scolge1": 1.0}, pr.BIG), pr.const(-pr.BIG))
    for k in K:
        l7.units += pr.clamp01_units(pr.add({"val": 1.0, f"m{k}": -1.0}, rem_cell), {f"gt{k}": 1.0})
        l7.units += pr.clamp01_units(pr.add({f"m{k}": 1.0, "val": -1.0}, rem_cell), {f"lt{k}": 1.0})
    # M_k[col] = M_{k-1}[col] + B[col] + carry. The carry into col is recovered from the digit
    # just written (this byte) and y = M_{k-1}[col-1] + B[col-1] (<= 18, carry-in <= 1):
    # carry = [y >= 10] (gated to digit bytes: col 0 has no carry-in) + [y == 9 and this byte is 0].
    # The +20 - 20*is0 shift moves y out of the hat's range unless this byte is '0'.
    y = {"Mp1": 1.0, "Bc1": 1.0}  # multiples: carry recovered from the previous output digit
    l7.units += pr.clamp01_units(pr.add(y, pr.const(-9.0), {"is_digit": pr.BIG}, pr.const(-pr.BIG)), {"carry": 1.0})
    l7.units += pr.hat_units(pr.add(y, {"is0": -20.0}, pr.const(20.0)), 9.0, {"carry": 1.0})
    l7.units.append(pr.relu_unit({"Rp": 1.0}, {"Rj": 1.0},
                                     label="R'[j] = R_{i-1}[j-1]: copies the shifted remainder digit (a digit, "
                                           "so ReLU is the identity)"))  # R'[j]: a_i when j == 0
    l7.units += pr.times_flag_units({"ai": 1.0, "Rp": -1.0}, {"col1": 1.0}, {"Rj": 1.0})
    l7.units.append(pr.relu_unit({"Rp1": 1.0}, {"Rj1": 1.0},
                                     label="R'[j-1] = R_{i-1}[j-2]: copies the shifted remainder digit below "
                                           "(for the borrow)"))  # R'[j-1]: a_i when j-1 == 0
    l7.units += pr.times_flag_units({"ai": 1.0, "Rp1": -1.0}, {"col2": 1.0}, {"Rj1": 1.0})

    # Layer 8: q*s squares (for M_q addresses), the key features of L9's latest heads
    # (dkey_k, rskey, zq) and the units test c0_k = [a_i >= M_k[0]].
    l8 = b.at(8)
    l8.heads += [
        pr.gather_head("(q+s)^2", pr.add({"qv": 1.0}, S_), "QS2", src="pos2"),
        pr.gather_head("q^2", {"qv": 1.0}, "Q2", src="pos2"),
    ]
    key_base = pr.add(IN_WORK, {"is_digit": 1.0, "Srowge9": 1.0, "Scolge1": 1.0})  # == 4 on remainder digits
    # dkey_k = pos on remainder digits (rows >= 9, col >= 1) that differ from M_k (gt_k or lt_k);
    # rskey = pos on q cells (col 0, rows >= 9), the "all upper digits agree" fallback.
    for k in K:
        l8.units.append(pr.pos_if(pr.add(key_base, {f"gt{k}": 1.0, f"lt{k}": 1.0}), 5.0, f"dkey{k}"))
        m0 = {f"Bo{v}": float((k * v) % 10) for v in range(10)}  # M_k[0] = (k*b_0) mod 10
        l8.units += pr.clamp01_units(pr.add({"ai": 1.0}, pr.scale(m0, -1.0), pr.const(1.0)), {f"c0{k}": 1.0})
    l8.units.append(pr.pos_if(pr.add(IN_WORK, {"Scol0": 1.0, "Srowge9": 1.0}), 3.0, "rskey"))
    # zq: the earliest nonzero q (rows >= 10) of the newest frame = the answer's first digit.
    # The dummy tags row 9's q_0 cell at BIG*F - pos - BIG/2: below every real q of this frame,
    # above all older frames, so with q = 0 the L9 head still has a definite in-frame target.
    l8.units.append(pr.newest_frame_earliest(pr.add(IN_WORK, {"Scol0": 1.0, "Srowge10": 1.0, "nz": 1.0}),
                                             4.0, "F", "zq"))
    dummy = pr.newest_frame_earliest(pr.add(IN_WORK, {"Scol0": 1.0, "Srow9": 1.0}), 3.0, "F", "zq")
    dummy.read = pr.add(dummy.read, pr.const(-pr.BIG / 2))  # loses to any real nonzero q of this frame
    l8.units.append(dummy)
    # qs = q*s, q = this row's quotient digit (qv). M_q is multiples row q-1 and its digit j
    # sits at column j, i.e. at E + (q-1)s + col, since R[j] is at col = j+1.
    qs = pr.product_form("QS2", "Q2", S2)
    l8.units += [  # M_q[j], M_q[j-1]; q = 0 clamps onto '=' (value 0)
        pr.relu_unit(pr.add(qs, pr.scale(S_, -1.0), COL), {"tMq": 1.0},
                    label="'tMq' = (q-1)s + col: offset after '=' of M_q[j] in multiples row q-1; q = 0 clamps to "
                          "0 ('=', value 0)"),
        pr.relu_unit(pr.add(qs, pr.scale(S_, -1.0), COL, pr.const(-1.0)), {"tMq1": 1.0},
                    label="'tMq1' = (q-1)s + col - 1: offset after '=' of M_q[j-1] (for the borrow); q = 0 clamps "
                          "to 0 ('=')"),
    ]

    # Layer 9: decide q_i = sum_k [R'_i >= M_k] in one pass, and the remainder's borrow.
    # Querying from the top cell of row 8+i, head k reads the latest key <= the query: digits
    # are LSB first, so that is the most significant differing digit of R' vs M_k, or (none
    # differ) the previous row's q cell, whose gt = lt = 0. Older rows sit earlier and lose.
    l9 = b.at(9)
    for k in K:
        l9.heads.append(HeadSpec(
            name=f"R' vs M_{k}: top differing digit",
            query=[{"one": pr.LATEST_BETA}],
            key=[{f"dkey{k}": 1.0, "rskey": 1.0}],
            value={f"gt{k}": {f"GT{k}": 1.0}, f"lt{k}": {f"LT{k}": 1.0}},
        ))
    l9.heads += [
        pr.gather_head("M_q[j]", {"E": 1.0, "tMq": 1.0}, "MqJ"),
        pr.gather_head("M_q[j-1]", {"E": 1.0, "tMq1": 1.0}, "MqJ1"),
        pr.latest_head("first nonzero q", "zq", "Zq"),
    ]
    # The vote: GT_k, LT_k are exclusive 0/1, so GT_k + ReLU(c0_k - GT_k - LT_k) = [R' >= M_k]
    # exactly; the nine sum into qd, which L10 writes as the q digit (0..9).
    for k in K:  # [R' >= M_k] = top differing digit greater, or all equal and a_i >= M_k[0]
        l9.units.append(pr.relu_unit({f"GT{k}": 1.0}, {"qd": 1.0},
                                        label=f"q += 1 if R' > M_{k} at their top differing digit (GT{k})"))
        l9.units.append(pr.relu_unit({f"c0{k}": 1.0, f"GT{k}": -1.0, f"LT{k}": -1.0}, {"qd": 1.0},
                                        label=f"q += 1 if all upper digits of R' and M_{k} agree (GT{k}=LT{k}=0) "
                                              f"and units digit a_i >= M_{k}[0] (c0{k})"))
    # R_i[j] = R'[j] - M_q[j] - borrow. With yb = R'[j-1] - M_q[j-1]: borrow = [yb <= -1]
    # + [yb == 0 and the digit just written is 9] (the +20 - 20*is9 shift). not_j0 = -BIG*col1:
    # no borrow into j = 0 (this byte is then the q digit, not a remainder digit).
    yb = {"Rj1": 1.0, "MqJ1": -1.0}  # remainder: borrow recovered from the previous output digit
    not_j0 = pr.scale({"col1": 1.0}, -pr.BIG)
    l9.units += pr.clamp01_units(pr.add(pr.scale(yb, -1.0), not_j0), {"br": 1.0})
    l9.units += pr.hat_units(pr.add(yb, {"is9": -20.0}, pr.const(20.0), not_j0), 0.0, {"br": 1.0})
    # found_q = [Zq > W + 9s]: the first-nonzero-q head landed past row 9's dummy, i.e. q != 0.
    l9.units += pr.clamp01_units(pr.add({"Zq": 1.0}, pr.scale(W, -1.0), pr.scale(S_, -9.0)), {"found_q": 1.0})

    # Layer 10: write the grid. Each digit source is gated to its region (before ';'):
    # multiples (M_{k-1} + B + carry) mod 10, row 9 zeros, q_i = qd at col 0, and
    # R_i[j] = (R'[j] - M_q[j] - borrow) mod 10; then ',' / ';', the sign and the answer start.
    l10 = b.at(10)
    l10.units += cm.ramp_mod10_units(pr.add({"Mp0": 1.0, "Bc": 1.0, "carry": 1.0}), 0, 19,
                                     {"rowle8": 1.0, "colsep": -1.0, "seen_semi": -1.0}, 1.0)
    l10.units += cm.ramp_mod10_units(pr.const(0.0), 0, 0, {"row9": 1.0, "seen_semi": -1.0}, 1.0)
    l10.units += cm.ramp_mod10_units({"qd": 1.0}, 0, 9,
                                     {"rowge10": 1.0, "col0": 1.0, "endrow": -1.0, "seen_semi": -1.0, "one": -1.0}, 1.0)
    l10.units += cm.ramp_mod10_units({"Rj": 1.0, "MqJ": -1.0, "br": -1.0}, -10, 9,
                                     {"rowge10": 1.0, "col0": -1.0, "seen_semi": -1.0}, 1.0)
    l10.units.append(pr.relu_unit({"rowle8": 1.0, "colsep": 1.0, "seen_semi": -1.0, "one": -1.0}, {"emit_comma": 1.0}))
    l10.units.append(pr.relu_unit({"endrow": 1.0, "col0": 1.0, "seen_semi": -1.0, "one": -1.0}, {"work_end": 1.0}))
    # sign = sa XOR sb, only for a nonzero quotient
    for x_, y_ in (("sa", "sb"), ("sb", "sa")):
        l10.units.append(pr.relu_unit({x_: 1.0, y_: -1.0, "found_q": 1.0, "one": -1.0}, {"sgn": 1.0},
                                      label=f"sign half of sa XOR sb: 1 when {x_}=1, {y_}=0 and the quotient is "
                                            f"nonzero (found_q)"))
    l10.units.append(pr.relu_unit(LQ, {"start": 1.0},
                                  label="default answer start: position of the last quotient digit, "
                                        "W + (9 + len A)s (used when the quotient is 0)"))
    # start = LQ + (Zq - LQ)*found_q: the first nonzero q digit, else the last q digit (a '0').
    l10.units += pr.times_flag_units(pr.add({"Zq": 1.0}, pr.scale(LQ, -1.0)), {"found_q": 1.0}, {"start": 1.0})

    t = {"pos": 1.0, "S": -1.0, "sgn": -1.0}  # answer digit index after an optional '-'
    # Layer 11: answer address src = start + t*s; ']' once src passes the last q digit (LQ);
    # '-' right after ';' when sgn (control logit 20 beats the digit logit 10).
    l11 = b.at(11)
    l11.heads += [
        pr.gather_head("(t+s)^2", pr.add(t, S_), "TS2", src="pos2"),
        pr.gather_head("t^2", t, "T2", src="pos2"),
    ]
    src = pr.add({"start": 1.0}, pr.product_form("TS2", "T2", S2))
    l11.units.append(pr.relu_unit(src, {"src": 1.0},
                                  label="'src' = start + t*s: grid position of answer digit t, one row "
                                        "(stride s) down the q column per digit"))
    l11.units += pr.clamp01_units(pr.gated(pr.add(src, pr.scale(LQ, -1.0)), working), {"frame_end": 1.0})
    l11.units.append(pr.relu_unit({"sgn": 1.0, "is_semi": 1.0, "one": -1.0}, {"emit_minus": 1.0}))

    # Layer 12: copy the q digit at src as the next answer digit (after ';' only).
    l12 = b.at(12)
    l12.heads.append(pr.gather_head("answer digit", {"src": 1.0}, "Av"))
    l12.units += cm.ramp_mod10_units({"Av": 1.0}, 0, 9, working, 1.0)
    return b
