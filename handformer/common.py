"""Input features, byte constants and small builders shared by every circuit.

Token embedding (one row per byte, all hand-set):

    one         1 for every byte (the bias channel)
    val         digit value 0..9 (0 for non-digits, so fences read as zero)
    oh0..oh9    one-hot digit (one head moves a digit as 10 dims)
    is_digit, nz (nonzero digit), n9 (digit != 9), is0, is9
    is_bos      '\\n' at position 0 (fence, attention sink)
    is_eq '='   is_comma ','   is_bar '|'   is_lt '<'
    is_semi     end of an op's working: ';' (result >= 0) or '<' (result < 0)
    is_lbr '['  opens a frame (one reduction step)
    is_rbr ']'  closes it
    is_lpar '(' is_rpar ')'   is_dot '.' (decimal point)
    is_plus '+' is_minus '-' is_star '*' is_slash '/' is_caret '^'  is_opany: any of them
    is_tilde    '~' (precision marker in the prompt: <expr>~N=)
"""

from __future__ import annotations

from . import primitives as pr
from .model import Form, Layout, ModelBuilder, UnitSpec, VOCAB

BOS = ord("\n")
EOS = ord("\n")
WORK_END = ord(";")
WORK_END_NEG = ord("<")
FRAME_OPEN = ord("[")
FRAME_CLOSE = ord("]")
COMMA = ord(",")
BAR = ord("|")
MINUS = ord("-")

DIGIT_LOGIT = 10.0
CONTROL_LOGIT = 20.0  # beats DIGIT_LOGIT: when a control byte fires, a digit hat may too

INPUT_DIMS = (
    "one", "pos", "pos2", "val", *(f"oh{d}" for d in range(10)),
    "is_digit", "nz", "n9", "is0", "is9", "is_bos",
    "is_eq", "is_semi", "is_lt", "is_comma", "is_bar", "is_lbr", "is_rbr",
    "is_lpar", "is_rpar", "is_dot", "is_plus", "is_minus", "is_star", "is_slash", "is_caret", "is_opany",
    "is_tilde",
)

# Bytes an expression line can contain, and the input dim that flags each.
ALPHABET = {**{ord("0") + d: f"oh{d}" for d in range(10)},
            ord("+"): "is_plus", ord("-"): "is_minus", ord("*"): "is_star", ord("/"): "is_slash",
            ord("^"): "is_caret",
            ord("("): "is_lpar", ord(")"): "is_rpar", ord("."): "is_dot"}


def add_input_dims(lay: Layout) -> None:
    lay.add(*INPUT_DIMS)


def embed(b: ModelBuilder) -> None:
    for byte in range(VOCAB):
        b.embed_byte(byte, {"one": 1.0})
    for d in range(10):
        b.embed_byte(
            ord("0") + d,
            {
                "val": float(d), f"oh{d}": 1.0, "is_digit": 1.0,
                "nz": float(d != 0), "n9": float(d != 9),
                "is0": float(d == 0), "is9": float(d == 9),
            },
        )
    single = {BOS: "is_bos", ord("="): "is_eq", WORK_END: "is_semi", COMMA: "is_comma",
              BAR: "is_bar", FRAME_OPEN: "is_lbr", FRAME_CLOSE: "is_rbr",
              ord("("): "is_lpar", ord(")"): "is_rpar", ord("."): "is_dot", ord("~"): "is_tilde"}
    for byte, dim in single.items():
        b.embed_byte(byte, {dim: 1.0})
    b.embed_byte(WORK_END_NEG, {"is_semi": 1.0, "is_lt": 1.0})
    for byte, dim in ((ord("+"), "is_plus"), (MINUS, "is_minus"), (ord("*"), "is_star"),
                      (ord("/"), "is_slash"), (ord("^"), "is_caret")):
        b.embed_byte(byte, {dim: 1.0, "is_opany": 1.0})


def unembed_digits(b: ModelBuilder, prefix: str = "D") -> None:
    for d in range(10):
        b.logit(f"{prefix}{d}", ord("0") + d, DIGIT_LOGIT)


def digit_offset(lo: Form, hi: Form, i: Form) -> Form:
    """``hi-1-i - lo``: >0 iff the operand has digit i."""
    return pr.add(hi, pr.scale(i, -1.0), pr.scale(lo, -1.0), pr.const(-1.0))


def digit_target_unit(lo: Form, hi: Form, i: Form, dst: str) -> UnitSpec:
    """``dst = max(hi-1-i - lo, 0)``; the gather target is ``lo + dst``."""
    return pr.relu_unit(digit_offset(lo, hi, i), {dst: 1.0},
                        f"digit address into '{dst}': how far digit i sits from the operand's left fence "
                        "(0 once the operand has no digit i, so the lookup lands on the fence, value 0)")


def exhausted_units(lo: Form, hi: Form, i: Form, dst: str) -> list[UnitSpec]:
    """``dst = 1 - [operand has digit i]``."""
    return [
        pr.relu_unit(pr.const(1.0), {dst: 1.0},
                     f"'{dst}' starts at 1 (operand exhausted); the next test subtracts 1 while digit i exists"),
        *pr.clamp01_units(digit_offset(lo, hi, i), {dst: -1.0}),
    ]


def ramp_mod10_units(x: Form, lo: int, hi: int, gate: Form | None, on_when: float,
                     prefix: str = "D") -> list[UnitSpec]:
    """``D[(x mod 10)] += 1`` for integer x in [lo, hi], from shared ReLU ramps.

    hat_k(x) = R(k-1) - 2R(k) + R(k+1) with R(c) = ReLU(x - c); summing hats
    by residue class means each ramp is one unit with a combined write.
    """
    coefs: dict[int, dict[str, float]] = {}
    for k in range(lo, hi + 1):
        dst = f"{prefix}{k % 10}"
        for c, w in ((k - 1, 1.0), (k, -2.0), (k + 1, 1.0)):
            row = coefs.setdefault(c, {})
            row[dst] = row.get(dst, 0.0) + w
    units = []
    for c in sorted(coefs):
        read = pr.add(x, pr.const(-float(c)))
        if gate is not None:
            read = pr.gated(read, gate, on_when)
        write = {d: w for d, w in coefs[c].items() if w != 0.0}
        if write:
            units.append(UnitSpec(read=read, write=write,
                                  label=f"digit ramp at {c}: part of 'last digit of x' (x mod 10) for x in "
                                        f"{lo}..{hi}; neighbouring ramps combine into one bump per value"))
    return units
