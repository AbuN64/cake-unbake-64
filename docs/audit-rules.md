# Audit rules

`unbake audit --against BASE` judges what a change adds over BASE; the tree's older leftovers never block anyone. A
row is a rule and its subject: a path's count of a row must rise over BASE for the audit to name it. There is no
baseline file, no allowlist and no per-name exemption.

## One finding shape

Every audit answer, old rule or new, lists findings as
`{Code, Severity: Refuse|Warn, File, Line, Message, NextStep}` (plain, rich or json like every unbake answer), then a
summary: `summary: 2 Refuse, 5 Warn (source.hex-value 2 Refuse, source.hex-review 5 Warn)`. A Refuse exits 1; a Warn
exits 0 and never blocks. Severity is fixed per rule by its certainty, below; no project switch changes it.

A Warn is standardized, not ignored: the sweep records every Warn a change adds, and a card resolves each one its change
adds or states the reason in the commit message.

## Rules

| Code | Severity | Refuses/warns when a change adds |
|---|---|---|
| `source.hex-value` | Refuse | a hex literal in a value context (arithmetic, compare, index, argument, assignment, case, array size) that the hexcensus table marks DEC: a mechanical rewrite to decimal. Masks, bit operands, GBI arguments, sentinels, byte-structured and hardware addresses are kept |
| `source.hex-review` | Warn | a hex literal of a REVIEW class: power of two, large value, address, float bits |
| `source.raw-data` | Refuse | a word/byte array that certainly holds another type: a long NUL-terminated string, a table of symbol addresses, MIPS code with its `jr ra` |
| `source.raw-data-review` | Warn | a raw array that looks like floats, short text or bare addresses |
| `source.raw-offset-cast` | Warn | a struct offset through a cast (`*(T *)(p + 0x2C)`, `((u8 *)p)[0x40]`, `w[0x320 / 4]`) |
| `source.placeholder-name` | Refuse | a symbol, file or folder named by address or placeholder: `func_8XXXXXXX`, `D_8XXXXXXX`, `*_padding_<addr>`, `unk_`, `sub_`, numbered `*Block17`/`*Words1` |
| `source.stand-in` | Refuse | `INCLUDE_ASM`, `INCLUDE_DATA`, `INCLUDE_RODATA`, a register `__asm__` global, or a C function with an `__asm__` body |
| `source.non-matching` | Refuse | a `NON_MATCHING` branch |
| `source.fakematch-note` | Refuse | a `FAKEMATCH` note |
| `source.guard-gap` | Warn | a function that is C under one VERSION guard and raw words under the other |
| `source.sdk-name` | Refuse | a non-SGI name under a library root, or an SGI-reserved name (`__os*`, `_` + capital) outside one |
| `source.generated-drift` | Refuse | a hand edit of README/docs/map/progress that `unbake sync` would regenerate |

Proven pad classes (`*TrailingBytes*`, `*Padding*`, `LinkGapWords`, `RetainedWords`, at most 8 bytes) stay allowed by
`raw-data`. A literal the base already spelled as often is not added: rewriting a neighbour to decimal moves no held
literal into `hex-value`.

## Project config

Thresholds and roots live in `config.toml` `[audit.hex]`, `[audit.rawdata]`, `[audit.placeholder]`, `[audit.sdk]` and
`[audit.generated]`; each key defaults to the value in `rules.toml` `[audit.*]`. The rule logic is generic.
