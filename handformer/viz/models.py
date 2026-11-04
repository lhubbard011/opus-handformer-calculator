"""The models the visualizer can show. Each is the same kind of thing: hand-written weights built
with model.ModelBuilder. What differs per model is only how its prompt is formed and checked, how
its dims are grouped into circuits, which pointers the page tags on the sequence, and the docs
for its own dims. Display and input framing only; nothing here touches a weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .. import arith, common as cm, csyntax, expr, frame, life

CALC_MARKS = ("Ls", "F", "p", "c", "o", "a_lo", "b_hi", "E", "S", "Rb", "P", "R")
CALC_SUMMARY = (*CALC_MARKS, "phase1", "phase2", "phase4", "act_add", "act_sub", "act_mul", "act_div", "act_pow",
                "open", "seen_eq", "seen_semi", "seen_bar", "drop", "cont", "end2", "end4",
                "in1", "in12", "in123", "src4", "neg", "use_mul")
LIFE_MARKS = ("C", "L", "F")
LIFE_SUMMARY = (*LIFE_MARKS, "is_count", "up_ok", "down_ok", "self", "n", "nxt", "cellp", "pT_slash", "pT_eq",
                *(f"K{d}" for d in range(10)), *(f"a_{k}" for k in life.circuit.NEIGHBOURS))


def _calc_prompt(text: str) -> list[int]:
    text = text.replace(" ", "").removesuffix("=")  # the page adds the '=' itself
    if not text:
        raise ValueError("empty expression")
    bad = sorted({ch for ch in text if ord(ch) not in cm.ALPHABET and ch != "~"})
    if bad:
        raise ValueError(f"unsupported characters: {''.join(bad)!r} (use digits . + - * / ( ) and ~N)")  # = chars_hint
    return arith.prompt(text)


def _calc_category(name: str) -> str:
    if name in cm.INPUT_DIMS:
        return "input"
    if name in frame.DIMS:
        return "frame"
    if name in expr.DIMS:
        return "parser"
    if name in expr.OUT_DIMS or name in arith.OP_OUTPUTS:
        return "output"
    return name.split(".", 1)[0] if "." in name else "other"


def _life_category(name: str) -> str:
    if name in cm.INPUT_DIMS or name in ("alive", "is_colon"):
        return "input"
    return "output" if name in life.OUTPUTS else "life"


@dataclass(frozen=True)
class ModelDef:
    key: str
    title: str
    builder: Callable
    prompt: Callable[[str], list[int]]  # input text -> prompt bytes, or ValueError
    category: Callable[[str], str]
    marks: tuple[str, ...]  # pointer dims tagged on the sequence strip
    summary: tuple[str, ...]  # dims the page reads at every step
    docs: dict[str, str]  # docs for this model's own dims (others come from the glossary)
    max_tokens: int
    default_input: str
    input_hint: str
    input_chars: str  # the characters the input may contain (the page checks these too)
    chars_hint: str  # how the "unsupported characters" message ends
    input_pattern: str  # regex the input must match (input framing only)
    pattern_error: str
    strip_spaces: bool = True  # the calculator and Life ignore spaces; C source keeps them
    prompt_end: str = "="  # the byte that ends the prompt
    max_input: int = 400
    multiline: bool = False  # the page shows a code box instead of a one-line input


C_SAMPLE = """#include <stdio.h>
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* Print the larger of two numbers. */
int main(void) {
    const char *msg = "max: %d\\n";  // a format string
    unsigned long x = 0x1Fu;
    double d = 2.5e-3;
    char c = '\\'';
    if (x > 3) printf(msg, MAX(1, 2));
    return 0;
}
"""
C_SUMMARY = ("E", *(f"r_{r.name}" for r in csyntax.RULES), "r_end", "kw", "norm")

MODELS = {
    "calc": ModelDef("calc", "calculator", arith.builder, _calc_prompt, _calc_category, CALC_MARKS, CALC_SUMMARY,
                     {}, 3000, "2+3*(4-1)", "an expression: + - * / ^ ( ), decimals, ~N for division precision",
                     "".join(sorted(chr(b) for b in cm.ALPHABET)) + "~",
                     "(use digits . + - * / ( ) and ~N)", r"^.+$", "empty expression"),
    "life": ModelDef("life", "game of life", life.builder, life.prompt, _life_category, LIFE_MARKS, LIFE_SUMMARY,
                     life.circuit.DOCS, 4000, "8:....../..#.../...#../.###../....../......",
                     "N:grid: generations (one digit), ':', rows of . and # separated by /",
                     "".join(sorted(life.ALLOWED)), life.circuit.CHARS_HINT, life.circuit.PATTERN,
                     life.circuit.PATTERN_ERROR),
    "c": ModelDef("c", "C syntax highlighter", csyntax.builder, csyntax.prompt, csyntax.circuit.category, ("E",),
                  C_SUMMARY, csyntax.circuit.DOCS, csyntax.circuit.MAX_SOURCE * 2 + 8, C_SAMPLE,
                  "C source: the model writes one lexer state per byte",
                  "".join(sorted(csyntax.circuit.ALLOWED)), csyntax.circuit.CHARS_HINT, csyntax.circuit.PATTERN,
                  csyntax.circuit.PATTERN_ERROR, strip_spaces=False, prompt_end="\x03",
                  max_input=csyntax.circuit.MAX_SOURCE, multiline=True),
}
