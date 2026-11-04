# A C syntax highlighter, written by hand into a transformer

`handformer/csyntax/` is a third model built with the same engine and primitives as the
calculator and Game of Life: 6 layers, 25 attention heads, 149 MLP units, 2,087 nonzero weights.
It lexes C source following C11's lexical grammar: keywords, identifiers (including universal
character names), pp-numbers, string literals and character constants with escapes and prefixes
(`L` `u` `U` `u8`), block and line comments, preprocessing directives (§6.10: `#`, or its digraph
`%:` or trigraph `??=`, first on a line), backslash-newline line splices (translation phase 2),
the trigraph `??/` acting as a backslash, and punctuators by the longest match (§6.4.6: `+=` is one
token, `+++` is `++` then `+`). The visualizer colours the code with its output, and the C parser
([CPARSE.md](CPARSE.md)) includes these weights to find its tokens.

The point is not the highlighter. It is that the model **is a ruleset you can read**: the lexer
below is a table of 53 rules, each rule is one unit with its own named dimension, and at every
byte exactly one rule fires. The table is the model.

```
$ python -m handformer.csyntax hello.c
int main(void) { /* hi */
kkk_iiiiokkkko_o_BDCCCCCE
  char *s = "a\"b"; // done
__kkkk_oi_o_ssStszo_Lcccccc
x += y->z; a <<= 2;
i_qj_iqjio_i_rlj_no
```

A punctuator's first letter says how long it is (`o` one byte, `q` two, `r` three, `v` four), and
its later letters count down what is left (`m` two more, `l` one more, `j` the last).

**Try it:** run `python -m handformer.viz` and pick "C syntax highlighter" in the model menu
(the public site shows only the calculator for now: `handformer/viz/models.yaml`). The **Code**
panel shows the source coloured by what the model has written so far; the step panel names the
rule that fired.

## How it works

The prompt is the source, then an end marker byte (`\x03`; C itself contains newlines). For
every source byte the model writes one **state letter**: what the lexer is in after that byte
(`k` keyword, `i` identifier, `s` inside a string, `S` an escaping backslash, `B` `D` `C` `E`
the parts of a block comment, `p` a directive, …). After the last byte it writes `\n`.

The trick is the same as in the other two models: **the text is the memory.** At every step the
current byte *is* the model's previous letter, so the lexer state is just the byte at the query
position, read by the embedding [circuit.py:124](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L124 "src:c.state"). The rest is exact lookups into the prompt
[circuit.py:326](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L326 "src:c.lookups"): the source byte being classified (s = position − E + 1, E = the end marker),
the bytes on either side, and for keywords the next 14 bytes.

Each rule is an AND of 0/1 conditions, compiled to one ReLU unit [circuit.py:261](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L261 "src:c.compile"): its state
group (the previous letter), the byte's class, and its neighbours'. A **fallback** rule ("anything
else in a string stays in the string") is its group minus its siblings, so a group always gives
exactly one rule. The unembedding reads the rule dims directly: a rule's dim votes for its letter.

**Punctuators** are decided the same way: at a punctuator's first byte, one unit per punctuator of
two or more bytes checks the bytes ahead [circuit.py:360](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L360 "src:dim:c:pl2"), and the longest match picks the
first letter. The countdown letters then carry the length, so the next byte needs no lookup:
the previous letter says whether the token goes on. That also makes every token's start visible
in the letters, which is what the parser reads.

**Keywords** are decided at a word's first byte [circuit.py:352](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L352 "src:c.keywords"): one unit per C11 keyword
(all 44) checks that the next bytes spell it and that the word ends there. The rest of the word
keeps the state its first byte got.

**"First on its line"** for `#` uses two latest-match heads, the latest newline and the latest
non-space byte, read at the byte before: if no non-space byte comes after the newline, `#` starts a
directive. **"The start"** is a position test, ReLU(1 − pos) [circuit.py:314](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L314 "src:c.start"), because the start
byte is a newline like every newline in the source.

## The rules

The whole table in the code [circuit.py:151](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L151 "src:c.rules"). Every rule links to the line that defines it; the
unit, its dim, its vote and its explanation in the visualizer all come from that one line.

| state | rule | when | writes | source |
|---|---|---|---|---|
| line comment | `line_end` | a newline ends a line comment (unless a backslash or '??/' splices the lines) | `_` whitespace outside any token | [circuit.py:153](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L153 "src:dim:c:r_line_end") |
| line comment | `line` | otherwise: anything else continues the line comment | `c` inside a line comment | [circuit.py:155](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L155 "src:dim:c:r_line") |
| after '/' opening a block comment | `block_star` | the byte after an opening '/' is its '*' | `D` the '*' that opens a block comment | [circuit.py:157](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L157 "src:dim:c:r_block_star") |
| block comment | `block_end` | '*/' closes the block comment (a '*' that opened it does not count) | `E` the '/' that closes a block comment | [circuit.py:158](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L158 "src:dim:c:r_block_end") |
| block comment | `block` | otherwise: anything else stays in the block comment | `C` inside a block comment | [circuit.py:160](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L160 "src:dim:c:r_block") |
| string | `str_escape` | in a string, a backslash starts an escape | `S` a backslash starting an escape in a string | [circuit.py:164](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L164 "src:dim:c:r_str_escape") |
| string | `str_trigraph` | in a string, '??/' is a backslash (trigraph): its first '?' | `G` the first '?' of a '??/' (a trigraph backslash) in a string | [circuit.py:165](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L165 "src:dim:c:r_str_trigraph") |
| string | `str_close` | in a string, '"' closes it | `z` the closing quote of a string | [circuit.py:166](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L166 "src:dim:c:r_str_close") |
| string | `str_newline` | an unterminated string ends at the newline | `_` whitespace outside any token | [circuit.py:167](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L167 "src:dim:c:r_str_newline") |
| string | `str` | otherwise: anything else is string body | `s` a string literal: its opening quote or body | [circuit.py:168](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L168 "src:dim:c:r_str") |
| string, after '?' of '??/' | `str_trigraph2` | the second '?' of a '??/' in a string | `H` its second '?' | [circuit.py:170](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L170 "src:dim:c:r_str_trigraph2") |
| string, after '??' of '??/' | `str_trigraph3` | the '/' of a '??/' in a string acts as the escaping backslash | `S` a backslash starting an escape in a string | [circuit.py:171](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L171 "src:dim:c:r_str_trigraph3") |
| after a backslash in a string | `str_escaped` | the byte after an escaping backslash is part of the escape | `t` the escaped character in a string | [circuit.py:172](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L172 "src:dim:c:r_str_escaped") |
| character constant | `chr_escape` | in a character constant, a backslash starts an escape | `A` a backslash in a character constant | [circuit.py:174](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L174 "src:dim:c:r_chr_escape") |
| character constant | `chr_trigraph` | in a character constant, '??/' is a backslash: its first '?' | `g` the first '?' of a '??/' in a character constant | [circuit.py:175](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L175 "src:dim:c:r_chr_trigraph") |
| character constant | `chr_close` | in a character constant, "'" closes it | `y` the closing quote of a character constant | [circuit.py:176](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L176 "src:dim:c:r_chr_close") |
| character constant | `chr_newline` | an unterminated character constant ends at the newline | `_` whitespace outside any token | [circuit.py:177](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L177 "src:dim:c:r_chr_newline") |
| character constant | `chr` | otherwise: anything else is character-constant body | `a` a character constant: its opening quote or body | [circuit.py:178](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L178 "src:dim:c:r_chr") |
| char, after '?' of '??/' | `chr_trigraph2` | the second '?' of a '??/' in a character constant | `h` its second '?' | [circuit.py:180](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L180 "src:dim:c:r_chr_trigraph2") |
| char, after '??' of '??/' | `chr_trigraph3` | the '/' of a '??/' in a character constant acts as the escaping backslash | `A` a backslash in a character constant | [circuit.py:181](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L181 "src:dim:c:r_chr_trigraph3") |
| after a backslash in a character constant | `chr_escaped` | the byte after an escaping backslash is part of the escape | `b` the escaped character in a character constant | [circuit.py:182](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L182 "src:dim:c:r_chr_escaped") |
| string or character prefix | `pfx_str` | after a prefix, '"' opens the string | `s` a string literal: its opening quote or body | [circuit.py:187](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L187 "src:dim:c:r_pfx_str") |
| string or character prefix | `pfx_chr` | after a prefix, "'" opens the character constant | `a` a character constant: its opening quote or body | [circuit.py:188](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L188 "src:dim:c:r_pfx_chr") |
| string or character prefix | `pfx` | otherwise: the '8' of u8 is still prefix | `P` a string or character prefix (L u U u8) | [circuit.py:189](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L189 "src:dim:c:r_pfx") |
| directive | `pp_end` | a newline ends the directive (unless a backslash or '??/' splices the lines) | `_` whitespace outside any token | [circuit.py:191](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L191 "src:dim:c:r_pp_end") |
| directive | `pp` | otherwise: anything else continues the directive | `p` a preprocessing directive | [circuit.py:193](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L193 "src:dim:c:r_pp") |
| number | `num` | a pp-number continues with a digit, letter, '_', '.' or an exponent sign | `n` a number (pp-number) | [circuit.py:195](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L195 "src:dim:c:r_num") |
| number | `num_splice` | a backslash-newline inside a number is deleted: the number goes on | `n` a number (pp-number) | [circuit.py:197](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L197 "src:dim:c:r_num_splice") |
| keyword | `word_k` | a keyword continues with identifier characters | `k` a keyword | [circuit.py:202](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L202 "src:dim:c:r_word_k") |
| identifier | `word_i` | an identifier continues with identifier characters | `i` an identifier | [circuit.py:203](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L203 "src:dim:c:r_word_i") |
| keyword | `word_k_more` | a backslash-newline or universal character name inside a word goes on (the keyword check sees up to it) | `k` a keyword | [circuit.py:204](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L204 "src:dim:c:r_word_k_more") |
| identifier | `word_i_more` | a backslash-newline or a universal character name (\u, \U) continues an identifier | `i` an identifier | [circuit.py:206](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L206 "src:dim:c:r_word_i_more") |
| punctuator | `punct_last` | the last byte of a punctuator (one more was to come) | `j` the last byte of a punctuator | [circuit.py:210](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L210 "src:dim:c:r_punct_last") |
| punctuator | `punct_more` | a punctuator byte with one more to come | `l` a punctuator byte with one more to come | [circuit.py:211](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L211 "src:dim:c:r_punct_more") |
| punctuator | `punct_more2` | the second byte of a four-byte punctuator: two more to come | `m` a punctuator byte with two more to come | [circuit.py:212](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L212 "src:dim:c:r_punct_more2") |
| outside any token | `open_block` | '/' followed by '*' opens a block comment | `B` the '/' that opens a block comment | [circuit.py:214](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L214 "src:dim:c:r_open_block") |
| outside any token | `open_line` | '/' followed by '/' opens a line comment | `L` the first '/' of a line comment | [circuit.py:215](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L215 "src:dim:c:r_open_line") |
| outside any token | `open_str` | '"' opens a string literal | `s` a string literal: its opening quote or body | [circuit.py:216](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L216 "src:dim:c:r_open_str") |
| outside any token | `open_chr` | "'" opens a character constant | `a` a character constant: its opening quote or body | [circuit.py:217](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L217 "src:dim:c:r_open_chr") |
| outside any token | `pp_start` | '#' first on its line starts a preprocessing directive | `p` a preprocessing directive | [circuit.py:218](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L218 "src:dim:c:r_pp_start") |
| outside any token | `pp_digraph` | '%:' (the digraph for '#') first on its line starts a directive | `p` a preprocessing directive | [circuit.py:220](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L220 "src:dim:c:r_pp_digraph") |
| outside any token | `pp_trigraph` | '??=' (the trigraph for '#') first on its line starts a directive | `p` a preprocessing directive | [circuit.py:222](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L222 "src:dim:c:r_pp_trigraph") |
| outside any token | `prefix` | L u U (or u8) right before a quote is a string or character prefix | `P` a string or character prefix (L u U u8) | [circuit.py:224](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L224 "src:dim:c:r_prefix") |
| outside any token | `ucn_start` | a universal character name (\u, \U) starts an identifier | `i` an identifier | [circuit.py:226](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L226 "src:dim:c:r_ucn_start") |
| outside any token | `num_start` | a digit starts a number | `n` a number (pp-number) | [circuit.py:227](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L227 "src:dim:c:r_num_start") |
| outside any token | `num_dot` | '.' followed by a digit starts a number | `n` a number (pp-number) | [circuit.py:228](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L228 "src:dim:c:r_num_dot") |
| outside any token | `keyword` | a word that is a C11 keyword | `k` a keyword | [circuit.py:229](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L229 "src:dim:c:r_keyword") |
| outside any token | `ident` | any other word is an identifier | `i` an identifier | [circuit.py:230](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L230 "src:dim:c:r_ident") |
| outside any token | `space` | whitespace outside a token | `_` whitespace outside any token | [circuit.py:231](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L231 "src:dim:c:r_space") |
| outside any token | `punct` | otherwise: any other byte outside a token is a punctuator: one byte (no longer punctuator starts here) | `o` a one-byte punctuator or operator | [circuit.py:233](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L233 "src:dim:c:r_punct") |
| outside any token | `punct2` | otherwise: any other byte outside a token is a punctuator: two bytes long (e.g. '+=', '->') | `q` the first byte of a two-byte punctuator | [circuit.py:233](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L233 "src:dim:c:r_punct2") |
| outside any token | `punct3` | otherwise: any other byte outside a token is a punctuator: three bytes long ('...', '<<=', '>>=') | `r` the first byte of a three-byte punctuator | [circuit.py:233](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L233 "src:dim:c:r_punct3") |
| outside any token | `punct4` | otherwise: any other byte outside a token is a punctuator: four bytes long ('%:%:') | `v` the first byte of a four-byte punctuator ('%:%:') | [circuit.py:233](https://github.com/lhubbard011/opus-handformer-calculator/blob/main/handformer/csyntax/circuit.py#L233 "src:dim:c:r_punct4") |

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
lexer (`tests/csyntax_reference.py`) on a realistic program, 62 edge cases (nested-looking
comments, escapes, unterminated literals, line splices, longest-match punctuators, digraphs) plus
every keyword and every keyword plus a letter, and 150 random fuzz strings over the bytes that make lexers go wrong. It also checks the
claims that make the table trustworthy: at every output byte exactly one rule dim is 1, every
head reads one exact value, and every yes/no step stays off its ramp (no exceptions).

## Limits

This is a lexer, so it highlights what editors highlight: `typedef` names look like identifiers and
macros are not expanded (both need a parser, not a lexer). Within lexing, three edge cases are
documented rather than handled, because keywords and comment openers are matched at fixed offsets:
a keyword joined to more letters by a line splice (`int\` newline `x` is the identifier `intx`)
still shows as a keyword up to the splice; a splice between the two bytes of `/*` or `//` does not
open a comment; and `??/*` is not read as an escaped `*`. Digraphs are punctuators like any other
(`<:` is two bytes, `%:%:` four); the other trigraphs are read as their raw bytes, so `??(`
is three one-byte punctuators, not `[`. Input is printable ASCII, up to 6,000 characters.
