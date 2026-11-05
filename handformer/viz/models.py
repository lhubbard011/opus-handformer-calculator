"""The models the visualizer can show. Each is the same kind of thing: hand-written weights built
with model.ModelBuilder. What differs per model is only how its prompt is formed and checked, how
its dims are grouped into circuits, which pointers the page tags on the sequence, and the docs
for its own dims. Display and input framing only; nothing here touches a weight.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from .. import arith, common as cm, cparse, csyntax, expr, frame, life, strip
from ..lm import viz as lmviz
from ..rewrite import viz as loopviz
from ..loop import docs as depthdocs, viz as depthviz

log = logging.getLogger(__name__)
SHOWN_FILE = Path(__file__).with_name("models.yaml")  # which models the drop-down offers

CALC_MARKS = ("Ls", "F", "p", "c", "o", "a_lo", "b_hi", "E", "S", "Rb", "P", "R")
CALC_SUMMARY = (*CALC_MARKS, "phase1", "phase2", "phase4", "act_add", "act_sub", "act_mul", "act_div", "act_pow",
                "open", "seen_eq", "seen_semi", "seen_bar", "drop", "cont", "end2", "end4",
                "in1", "in12", "in123", "src4", "neg", "use_mul", "sp", "sp_at", "pf", "cn", "st_src", "st_end", "st_cont")
LIFE_MARKS = ("C", "L", "F")
LIFE_SUMMARY = (*LIFE_MARKS, "is_count", "up_ok", "down_ok", "self", "n", "nxt", "cellp", "pT_slash", "pT_eq",
                *(f"K{d}" for d in range(10)), *(f"a_{k}" for k in life.circuit.NEIGHBOURS))


def _calc_prompt(text: str) -> list[int]:
    """The input exactly as typed, as its UTF-8 bytes, framed as ``\\n<input>=``. Nothing is
    stripped or refused for what it contains: spaces (which the weights remove, strip.py),
    letters, '=' or 'é' reach the weights, and what they do with bytes they have no input
    features for is the model's business (the run lists those as unknown_chars)."""
    if not text:
        raise ValueError("empty expression")
    unknown = sorted({ch for ch in text if len(ch.encode("utf-8")) > 1 or ord(ch) not in cm.ALPHABET and ch not in "~ "})
    if unknown:
        log.info("calculator input has bytes it has no features for: %r", "".join(unknown))
    return [cm.BOS, *text.encode("utf-8"), ord("=")]  # = arith.prompt, for any bytes


def _calc_category(name: str) -> str:
    if name in cm.INPUT_DIMS:
        return "input"
    if name in frame.DIMS:
        return "frame"
    if name in expr.DIMS or name in strip.DIMS:  # a strip step is a rewrite of the line
        return "parser"
    if name in expr.OUT_DIMS or name in strip.OUT_DIMS or name in arith.OP_OUTPUTS:
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
    docs: Mapping[str, str]  # docs for this model's own dims (others come from the glossary)
    max_tokens: int
    default_input: str
    input_hint: str
    input_chars: str  # the characters the model has input features for (the page checks these too)
    chars_hint: str  # how the "unsupported characters" message ends
    input_pattern: str  # regex the input must match (input framing only)
    pattern_error: str
    strip_spaces: bool = True  # the calculator and Life ignore spaces; C source keeps them
    prompt_end: str = "="  # the byte that ends the prompt ("" for none)
    prompt_prefix: str = "\n"  # the byte the prompt starts with
    max_input: int = 400
    multiline: bool = False  # the page shows a code box instead of a one-line input
    node_names: tuple[tuple[str, str], ...] = ()  # parser: node code -> name (the page's tree view)
    lenient: bool = False  # any input goes in, as its UTF-8 bytes (to see what the weights do); runs list the unknown ones
    extra_info: dict = field(default_factory=dict)  # model-specific facts the page needs (sent with /api/info)
    max_new: int | None = None  # at most this many written bytes per run (None: up to max_tokens)
    sampling: bool = False  # the page may ask for seeded sampling instead of greedy decoding (language models)


# The recurrent-depth calculator: the roles, the walk, the working's registers, the router
DEPTH_MARKS = ("fz", "p", "c", "o", "a_lo", "b_hi", "ptrN", "cp_t", "dg_addr")
DEPTH_SUMMARY = (*DEPTH_MARKS, "is_cell", "lat", "k_s", "k_w", "k_q", "k_r", "is_eq", "think", "route", "halt",
                 "act_sp", "act_br", "act_red", "opp", "opm", "opmul", "opdiv", "opc",
                 "wt", "ans", "preE", "z_mid", "ptr", "mk", "to_mid", "to_root", "PWa", "PWm", "PWq", "PQ", "PR",
                 "nc", "dA", "dB", "cA", "cB", "cm", "kk", "qd", "qt", "rd", "rn")

# The rewriting loop (bidirectional block): the root's decisions, the cell's zone, its column work, what it carries
REWRITE_SUMMARY = ("act_sp", "act_br", "act_red", "rwn", "rwp", "macc", "dwork", "dsub", "grow_any", "go", "route",
                "p", "c", "o", "a_lo", "b_hi", "kk", "qc", "K", "Q", "F", "ni", "Fp", "lo_B", "m_B", "g_B", "gelo",
                "ltend", "blank", "red_B", "br_B", "pw_B", "in_col", "aj", "bj", "cy", "bw1", "inx", "cR")


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

PARSE_MARKS = ("E", "M", "Lsep", "la_pos", "mpos")
PARSE_SUMMARY = (*PARSE_MARKS, "first", "isclose", "after_M", *C_SUMMARY[1:],
                 *(f"r_{r.name}" for r in cparse.rules()))
PARSE_MAX_INPUT = 120  # about 140 output bytes per token: 120 bytes of C is ~5-10k bytes of records
PARSE_SAMPLE = """if (ready) {
    count + 1;
} else
    while (waiting) ;
"""

MODELS = {
    "calc": ModelDef("calc", "calculator", arith.builder, _calc_prompt, _calc_category, CALC_MARKS, CALC_SUMMARY,
                     {}, 3000, "2+3*(4-1)", "an expression: + - * / ^ ( ), decimals, ~N for division precision",
                     "".join(sorted(chr(b) for b in cm.ALPHABET)) + "~ ",  # spaces: the weights remove them (strip.py)
                     "(use digits . + - * / ( ) and ~N)", r"^.+$", "empty expression", strip_spaces=False,
                     lenient=True),
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
    "parse": ModelDef("parse", "C parser (stage 2: toy grammar)", cparse.builder, cparse.prompt,
                      cparse.circuit.category, PARSE_MARKS, PARSE_SUMMARY, cparse.docs(),
                      PARSE_MAX_INPUT * 141 + 64, PARSE_SAMPLE,
                      "C: if / else / while, { } blocks, ( ) and + over identifiers and numbers; the model writes "
                      "a lexer letter per byte, then one record per parser action",
                      "".join(sorted(csyntax.circuit.ALLOWED)), csyntax.circuit.CHARS_HINT, r"\S",
                      "empty source", strip_spaces=False, prompt_end="\x03", max_input=PARSE_MAX_INPUT,
                      multiline=True, node_names=tuple(cparse.toy.NAMES.items())),
    # The calculator as one looped bidirectional block rewriting the prompt in place (handformer/rewrite,
    # REWRITE.md): set aside, not a decoder (its block reads later positions).
    "rewrite": ModelDef("rewrite", "calculator, looped rewriter (bidirectional)", loopviz.builder, loopviz.prompt, loopviz.category,
                     ("o", "a_lo", "b_hi", "p", "c", "lo_B"), REWRITE_SUMMARY, loopviz.DOCS, loopviz.MAX_INPUT + 2,
                     loopviz.SAMPLE, loopviz.HINT,
                     loopviz.CHARS, loopviz.CHARS_HINT,
                     r"^.+$", "empty expression", strip_spaces=False, max_input=loopviz.MAX_INPUT,
                     extra_info={"looped": True}),
    # The calculator as a recurrent-depth decoder (handformer/loop, LOOPED.md): it thinks in latent
    # positions, each iterating the block as deep as its router asks, and writes only the answer.
    "loop": ModelDef("loop", "calculator, recurrent depth (answer first)", depthviz.builder, depthviz.prompt,
                     depthviz.category, DEPTH_MARKS, DEPTH_SUMMARY, depthdocs.docs(), depthviz.MAX_INPUT + 2,
                     depthviz.SAMPLE, depthviz.HINT, depthviz.CHARS, depthviz.CHARS_HINT,
                     r"^.+$", "empty expression", strip_spaces=False, max_input=depthviz.MAX_INPUT,
                     extra_info={"depth": True}),
    # Needs the corpus (python tools/fetch_corpus.py); offered locally with --models, not on the site.
    "lm": ModelDef("lm", "language model (stage 1: trigram heads)", lmviz.builder, lmviz.prompt, lmviz.category,
                   (), (), lmviz.LazyDocs(), lmviz.MAX_INPUT + 1 + lmviz.MAX_NEW, lmviz.SAMPLE, lmviz.HINT,
                   "".join(chr(c) for c in range(32, 127)) + "\n\t", "", r"[\s\S]", "empty text",
                   strip_spaces=False, prompt_end="", prompt_prefix="\x02", max_input=lmviz.MAX_INPUT, multiline=True,
                   lenient=True, extra_info={"byte_names": lmviz.BYTE_NAMES}, max_new=lmviz.MAX_NEW, sampling=True),
    # The trained reference under the same constraints (no LayerNorm, no biases, p and p^2 only), to
    # compare with "lm" head by head (tools/train_reference.py --arch engine; local only, like "lm").
    "lmt": ModelDef("lmt", "language model, trained (1 layer, no LayerNorm)", lmviz.trained_builder, lmviz.trained_prompt,
                    lmviz.trained_category, (), (), lmviz.TRAINED_DOCS, lmviz.TRAINED_CTX, lmviz.SAMPLE,
                    lmviz.TRAINED_HINT, "".join(chr(c) for c in range(32, 127)) + "\n\t", "", r"[\s\S]",
                    "empty text", strip_spaces=False, prompt_end="", prompt_prefix="", max_input=lmviz.TRAINED_CTX - 1,
                    multiline=True, lenient=True, max_new=lmviz.MAX_NEW, sampling=True),
}


def shown(path: Path = SHOWN_FILE) -> tuple[str, ...]:
    """The model keys the visualizer offers, in menu order (models.yaml ``shown``). The first is the default."""
    data = yaml.safe_load(path.read_text())
    keys = data.get("shown") if isinstance(data, dict) else None
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
        raise ValueError(f"{path}: 'shown' must be a non-empty list of model keys (one of {', '.join(MODELS)})")
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        raise ValueError(f"{path}: unknown model(s) {', '.join(unknown)} (one of {', '.join(MODELS)})")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: a model is listed twice in 'shown'")
    log.info("visualizer shows %d of %d models: %s", len(keys), len(MODELS), ", ".join(keys))
    return tuple(keys)
