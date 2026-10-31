# Address tables of an older build

`unbake type-pointers --spec VERSION:LOW:HIGH:DELTA ...` types the address tables an older link left in a ROM
(jump tables, handler lists) as pointer arrays of the older build's symbols. `DELTA` is the older build's vram minus
the ROM offset, as in `type-pool`. `--check` reports and writes nothing; `--dest` moves the sources (default
`src/DATA/EARLIER_BUILD`).

## Proof rule

A span is typed only when every word is either

- zero (a null pointer), or
- an address in the RAM window (`0x80000000..0x80800000`) that is exactly the address of a symbol in
  `versions/<v>/earlier_build_symbols.txt`, the symbols the typed OLD-DATA and OLD-RODATA windows declare. Symbols are
  compared on their low 24 bits, so a pool symbol kept without the `0x80` segment bits resolves.

One word that is no address, or an address no symbol has, refuses the whole span. The refusal row names the span
(`version`, `rom`, `end`) and the first word that failed (`word_rom`, `word`, `why`). A table is never half typed, a
word is never guessed to a nearby symbol, and no raw word is emitted.

The rule has no version in it. The same command closes every version: a version is a ROM and its own
`earlier_build_symbols.txt`. A table that two versions hold at the same older address has the same body and gets one
source guarded `#if defined(VERSION_A) || defined(VERSION_B)`.

## Output

Per typed table, `EARLIER_ADDRTABLE_<vram>.CPP` holds an `extern char sym[];` per distinct target and the array
`void *g_earlierAddrTable_<vram>[N] = { sym, ..., 0 };`, and `earlier_build_symbols.txt` gains
`g_earlierAddrTable_<vram> = 0x<vram>; // rom:0x.. size:0x..`. The result lists `tables` (version, rom, end, delta,
`widened`), `unfit` (the refusals), `written` and `bytes`.

An edge that splits a word (a neighbouring range owns its first or last byte) is widened to the word and listed in
`widened`; the project row for the window and the retirement of the neighbour's overlapping blob row are the
caller's, as for `type-pool` windows.

Tables whose targets are code of the older build resolve only once those targets have symbols: the reader does not
mint a symbol for an address it cannot name.

## Labels from the current build

`unbake reconstruct-labels --spec VERSION:LOW:HIGH:DELTA ... [--check]` gives the older build's targets their symbols
from evidence. The current build (`build/<v>/<project>.elf`) holds the same switch or pointer array when a run of the
older ROM has the same entry count, the same zero entries and one difference between each older word and the current
word (the older build's delta of that array). Each label is then `<owner>` plus offset of the current symbol or function
holding the current word; a function owner also gets its start (`owner.start + delta`). Rows are appended to
`versions/<v>/earlier_build_symbols.txt` as `g_earlier_<owner>[_<offset>] = 0x<addr>; // <version>:<table>@<address>`,
so `type-pointers` closes the run on the next call.

- Only current tables of at least four entries are matched; a shorter run fits any table and is listed unmatched.
- A run is matched in its own version, then in the project's other versions. Copies of one array give the same
  symbols; tables that give different symbols refuse the run as ambiguous.
- A run whose labels do not agree on one delta is refused with the closest table's measured disagreement (`labels`,
  `agree`, `differ`, `deltas`); two runs of a version that give one name two addresses are both refused.
- A word without the segment bits (such as `0x0024E410`) is no pointer the linker can place: it is listed with that
  reason and given no symbol.

The result lists `tables`, `unfit`, the `symbols` added per version and the `labels` matched.
