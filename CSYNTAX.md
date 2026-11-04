# A C syntax highlighter, written by hand into a transformer

`handformer/csyntax/` is a third model built with the same engine and primitives as the
calculator and Game of Life: 6 layers, 24 attention heads, 114 MLP units, 1,788 nonzero weights.
It lexes C source following C11's lexical grammar: keywords, identifiers (including universal
character names), pp-numbers, string literals and character constants with escapes and prefixes
(`L` `u` `U` `u8`), block and line comments, preprocessing directives (§6.10: `#`, or its digraph
`%:` or trigraph `??=`, first on a line), backslash-newline line splices (translation phase 2), and
the trigraph `??/` acting as a backslash. The visualizer colours the code with its output.

The point is not the highlighter. It is that the model **is a ruleset you can read**: the lexer
below is a table of 47 rules, each rule is one unit with its own named dimension, and at every
byte exactly one rule fires. The table is the model.

```
$ python -m handformer.csyntax hello.c
int main(void) { /* hi */
kkk_iiiiokkkko_o_BDCCCCCE
  char *s = "a\"b"; // done
__kkkk_oi_o_ssStszo_Lcccccc
```

**Try it:** pick "C syntax highlighter" in the visualizer's model menu, or
[open it directly](https://lhubbard011.github.io/opus-handformer-calculator/#m=c). The **Code**
panel shows the source coloured by what the model has written so far; the step panel names the
rule that fired.

## How it works

The prompt is the source, then an end marker byte (`\x03`; C itself contains newlines). For
every source byte the model writes one **state letter**: what the lexer is in after that byte
(`k` keyword, `i` identifier, `s` inside a string, `S` an escaping backslash, `B` `D` `C` `E`
the parts of a block comment, `p` a directive, …). After the last byte it writes `\n`.

The trick is the same as in the other two models: **the text is the memory.** At every step the
current byte *is* the model's previous letter, so the lexer state is just the byte at the query
position, read by the embedding [circuit.py:96](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L96 "src:c.state"). The rest is exact lookups into the prompt
[circuit.py:273](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L273 "src:c.lookups"): the source byte being classified (s = position − E + 1, E = the end marker),
the bytes on either side, and for keywords the next 14 bytes.

Each rule is an AND of 0/1 conditions, compiled to one ReLU unit [circuit.py:221](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L221 "src:c.compile"): its state
group (the previous letter), the byte's class, and its neighbours'. A **fallback** rule ("anything
else in a string stays in the string") is its group minus its siblings, so a group always gives
exactly one rule. The unembedding reads the rule dims directly: a rule's dim votes for its letter.

**Keywords** are decided at a word's first byte [circuit.py:294](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L294 "src:c.keywords"): one unit per C11 keyword
(all 44) checks that the next bytes spell it and that the word ends there. The rest of the word
keeps the state its first byte got.

**"First on its line"** for `#` uses two latest-match heads, the latest newline and the latest
non-space byte, read at the byte before: if no non-space byte comes after the newline, `#` starts a
directive. **"The start"** is a position test, ReLU(1 − pos) [circuit.py:261](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L261 "src:c.start"), because the start
byte is a newline like every newline in the source.

## The rules

The whole table in the code [circuit.py:119](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L119 "src:c.rules"). Every rule links to the line that defines it; the
unit, its dim, its vote and its explanation in the visualizer all come from that one line.

| state | rule | when | writes | source |
|---|---|---|---|---|
| line comment | `line_end` | a newline ends a line comment (unless a backslash or '??/' splices the lines) | `_` whitespace outside any token | [circuit.py:121](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L121 "src:dim:c:r_line_end") |
| line comment | `line` | otherwise: anything else continues the line comment | `c` inside a line comment | [circuit.py:123](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L123 "src:dim:c:r_line") |
| after '/' opening a block comment | `block_star` | the byte after an opening '/' is its '*' | `D` the '*' that opens a block comment | [circuit.py:125](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L125 "src:dim:c:r_block_star") |
| block comment | `block_end` | '*/' closes the block comment (a '*' that opened it does not count) | `E` the '/' that closes a block comment | [circuit.py:126](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L126 "src:dim:c:r_block_end") |
| block comment | `block` | otherwise: anything else stays in the block comment | `C` inside a block comment | [circuit.py:128](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L128 "src:dim:c:r_block") |
| string | `str_escape` | in a string, a backslash starts an escape | `S` a backslash starting an escape in a string | [circuit.py:132](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L132 "src:dim:c:r_str_escape") |
| string | `str_trigraph` | in a string, '??/' is a backslash (trigraph): its first '?' | `G` the first '?' of a '??/' (a trigraph backslash) in a string | [circuit.py:133](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L133 "src:dim:c:r_str_trigraph") |
| string | `str_close` | in a string, '"' closes it | `z` the closing quote of a string | [circuit.py:134](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L134 "src:dim:c:r_str_close") |
| string | `str_newline` | an unterminated string ends at the newline | `_` whitespace outside any token | [circuit.py:135](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L135 "src:dim:c:r_str_newline") |
| string | `str` | otherwise: anything else is string body | `s` a string literal: its opening quote or body | [circuit.py:136](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L136 "src:dim:c:r_str") |
| string, after '?' of '??/' | `str_trigraph2` | the second '?' of a '??/' in a string | `H` its second '?' | [circuit.py:138](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L138 "src:dim:c:r_str_trigraph2") |
| string, after '??' of '??/' | `str_trigraph3` | the '/' of a '??/' in a string acts as the escaping backslash | `S` a backslash starting an escape in a string | [circuit.py:139](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L139 "src:dim:c:r_str_trigraph3") |
| after a backslash in a string | `str_escaped` | the byte after an escaping backslash is part of the escape | `t` the escaped character in a string | [circuit.py:140](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L140 "src:dim:c:r_str_escaped") |
| character constant | `chr_escape` | in a character constant, a backslash starts an escape | `A` a backslash in a character constant | [circuit.py:142](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L142 "src:dim:c:r_chr_escape") |
| character constant | `chr_trigraph` | in a character constant, '??/' is a backslash: its first '?' | `g` the first '?' of a '??/' in a character constant | [circuit.py:143](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L143 "src:dim:c:r_chr_trigraph") |
| character constant | `chr_close` | in a character constant, "'" closes it | `y` the closing quote of a character constant | [circuit.py:144](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L144 "src:dim:c:r_chr_close") |
| character constant | `chr_newline` | an unterminated character constant ends at the newline | `_` whitespace outside any token | [circuit.py:145](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L145 "src:dim:c:r_chr_newline") |
| character constant | `chr` | otherwise: anything else is character-constant body | `a` a character constant: its opening quote or body | [circuit.py:146](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L146 "src:dim:c:r_chr") |
| char, after '?' of '??/' | `chr_trigraph2` | the second '?' of a '??/' in a character constant | `h` its second '?' | [circuit.py:148](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L148 "src:dim:c:r_chr_trigraph2") |
| char, after '??' of '??/' | `chr_trigraph3` | the '/' of a '??/' in a character constant acts as the escaping backslash | `A` a backslash in a character constant | [circuit.py:149](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L149 "src:dim:c:r_chr_trigraph3") |
| after a backslash in a character constant | `chr_escaped` | the byte after an escaping backslash is part of the escape | `b` the escaped character in a character constant | [circuit.py:150](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L150 "src:dim:c:r_chr_escaped") |
| string or character prefix | `pfx_str` | after a prefix, '"' opens the string | `s` a string literal: its opening quote or body | [circuit.py:155](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L155 "src:dim:c:r_pfx_str") |
| string or character prefix | `pfx_chr` | after a prefix, "'" opens the character constant | `a` a character constant: its opening quote or body | [circuit.py:156](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L156 "src:dim:c:r_pfx_chr") |
| string or character prefix | `pfx` | otherwise: the '8' of u8 is still prefix | `P` a string or character prefix (L u U u8) | [circuit.py:157](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L157 "src:dim:c:r_pfx") |
| directive | `pp_end` | a newline ends the directive (unless a backslash or '??/' splices the lines) | `_` whitespace outside any token | [circuit.py:159](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L159 "src:dim:c:r_pp_end") |
| directive | `pp` | otherwise: anything else continues the directive | `p` a preprocessing directive | [circuit.py:161](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L161 "src:dim:c:r_pp") |
| number | `num` | a pp-number continues with a digit, letter, '_', '.' or an exponent sign | `n` a number (pp-number) | [circuit.py:163](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L163 "src:dim:c:r_num") |
| number | `num_splice` | a backslash-newline inside a number is deleted: the number goes on | `n` a number (pp-number) | [circuit.py:165](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L165 "src:dim:c:r_num_splice") |
| keyword | `word_k` | a keyword continues with identifier characters | `k` a keyword | [circuit.py:170](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L170 "src:dim:c:r_word_k") |
| identifier | `word_i` | an identifier continues with identifier characters | `i` an identifier | [circuit.py:171](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L171 "src:dim:c:r_word_i") |
| keyword | `word_k_more` | a backslash-newline or universal character name inside a word goes on (the keyword check sees up to it) | `k` a keyword | [circuit.py:172](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L172 "src:dim:c:r_word_k_more") |
| identifier | `word_i_more` | a backslash-newline or a universal character name (\u, \U) continues an identifier | `i` an identifier | [circuit.py:174](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L174 "src:dim:c:r_word_i_more") |
| outside any token | `open_block` | '/' followed by '*' opens a block comment | `B` the '/' that opens a block comment | [circuit.py:177](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L177 "src:dim:c:r_open_block") |
| outside any token | `open_line` | '/' followed by '/' opens a line comment | `L` the first '/' of a line comment | [circuit.py:178](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L178 "src:dim:c:r_open_line") |
| outside any token | `open_str` | '"' opens a string literal | `s` a string literal: its opening quote or body | [circuit.py:179](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L179 "src:dim:c:r_open_str") |
| outside any token | `open_chr` | "'" opens a character constant | `a` a character constant: its opening quote or body | [circuit.py:180](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L180 "src:dim:c:r_open_chr") |
| outside any token | `pp_start` | '#' first on its line starts a preprocessing directive | `p` a preprocessing directive | [circuit.py:181](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L181 "src:dim:c:r_pp_start") |
| outside any token | `pp_digraph` | '%:' (the digraph for '#') first on its line starts a directive | `p` a preprocessing directive | [circuit.py:183](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L183 "src:dim:c:r_pp_digraph") |
| outside any token | `pp_trigraph` | '??=' (the trigraph for '#') first on its line starts a directive | `p` a preprocessing directive | [circuit.py:185](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L185 "src:dim:c:r_pp_trigraph") |
| outside any token | `prefix` | L u U (or u8) right before a quote is a string or character prefix | `P` a string or character prefix (L u U u8) | [circuit.py:187](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L187 "src:dim:c:r_prefix") |
| outside any token | `ucn_start` | a universal character name (\u, \U) starts an identifier | `i` an identifier | [circuit.py:189](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L189 "src:dim:c:r_ucn_start") |
| outside any token | `num_start` | a digit starts a number | `n` a number (pp-number) | [circuit.py:190](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L190 "src:dim:c:r_num_start") |
| outside any token | `num_dot` | '.' followed by a digit starts a number | `n` a number (pp-number) | [circuit.py:191](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L191 "src:dim:c:r_num_dot") |
| outside any token | `keyword` | a word that is a C11 keyword | `k` a keyword | [circuit.py:192](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L192 "src:dim:c:r_keyword") |
| outside any token | `ident` | any other word is an identifier | `i` an identifier | [circuit.py:193](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L193 "src:dim:c:r_ident") |
| outside any token | `space` | whitespace outside a token | `_` whitespace outside any token | [circuit.py:194](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L194 "src:dim:c:r_space") |
| outside any token | `punct` | otherwise: any other byte outside a token is a punctuator | `o` a punctuator or operator | [circuit.py:195](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L195 "src:dim:c:r_punct") |

## Check it by breaking it

Knock out the rule `str_escape` (a backslash in a string starts an escape) and an escaped quote
closes the string:

```
puts("say \"hi\"");
iiiiosssssStssStzoo     normal
iiiiossssssziioszoo     without str_escape: the string ends at \", and hi becomes identifiers
```

Knock out `keyword` and `int` becomes `oii`: its first byte is claimed by no rule, so the
fallback "any other byte outside a token is a punctuator" takes it, and `nt` starts a new word.
That is the "exactly one rule" bookkeeping made visible. In the visualizer, step to a byte, find
its rule dim in the Residual stream panel, and press "knock out".

## Tests and claims

`tests/test_csyntax.py` checks the model byte for byte against a separately written plain-Python
lexer (`tests/csyntax_reference.py`) on a realistic program, 30 edge cases (nested-looking
comments, escapes, unterminated literals, line splices, every keyword and every keyword plus a
letter), and 150 random fuzz strings over the bytes that make lexers go wrong. It also checks the
claims that make the table trustworthy: at every output byte exactly one rule dim is 1, every
head reads one exact value, and every yes/no step stays off its ramp (no exceptions).

## Limits

This is a lexer, so it highlights what editors highlight: `typedef` names look like identifiers and
macros are not expanded (both need a parser, not a lexer). Within lexing, three edge cases are
documented rather than handled, because keywords and comment openers are matched at fixed offsets:
a keyword joined to more letters by a line splice (`int\` newline `x` is the identifier `intx`)
still shows as a keyword up to the splice; a splice between the two bytes of `/*` or `//` does not
open a comment; and `??/*` is not read as an escaped `*`. The other trigraphs and digraphs are
punctuators, which is how they are coloured. Input is printable ASCII, up to 6,000 characters.
