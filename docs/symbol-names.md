# Portable source names

`resources/data/symbol-names.toml` is the one target table. `unbake.api.name_issues`
validates proposed symbols, sections and relative paths against every row;
`unbake.api.naming_findings` inspects source sets. Recipe Maker uses both at name
judgment, structure proposal/patch construction and before applying saved patches.
Debug/SDK evidence establishes identity, not an exemption from target portability.
Aliases stay ordinary C names and `#define`; no caller adds an object symbol prefix.

| Target | C prefix | Identifier limit | Section limit | Case policy |
| --- | --- | --- | --- | --- |
| elf-gcc | empty | no fixed limit | no fixed limit | sensitive |
| coff-mingw | i386 `_`; x86_64/arm64 empty | no fixed limit | 8 bytes in PE images | fold paths and portable symbol names |
| macho-clang | `_` | no fixed limit | 16 bytes | sensitive |
| android-clang | empty | no fixed limit | no fixed limit | sensitive |
| msvc | i386 `_`; x86_64/arm64 empty | 247 bytes (C) | 8 bytes in PE images | fold paths and portable symbol names |

Zero limits mean no fixed toolchain limit, not zero allowed bytes. COFF objects
can use string-table section names longer than eight bytes; the portable rule also
covers linked PE images. C++ names differ by ABI; an intended C boundary must use
`extern "C"`. The checker rejects explicit `extern "C++"` in game C. It never
infers the compilation language from `.CPP`: projects also compile those files as C.
Compilation remains responsible for arbitrary C++ semantics and macro expansion.

The ASCII identifier subset, all C/C++ keyword sets, compiler extensions, reserved
identifier namespaces, section lengths, Windows device names and case-folded
collisions are checked. Windows COFF symbols are case-sensitive; portable symbol
case uniqueness is a conservative source policy, not a claim about its linker.
File component limits are 255 bytes. C names may not start with the compiler's
reserved global underscore namespace. The table identifies implementation-owned SDK namespaces (`__os`/`__OS` followed
by a capital letter, `_G_` and `_GBI_`) in source declarations. Only their
reserved-namespace violation is exempt: character, length, collision and alias
checks still apply. Recipe proposals never receive this exemption. Configured
library roots keep their existing exclusion boundary. No project or per-name
allowlist is used.

`unbake audit --against REV` scans all non-library C/header files (including
untracked additions) and compares naming findings to reviewed project tool config
at `config/symbol-name-baseline.json`. An absent baseline allows no naming debt.
The schema-1 file has `revision` (the reviewed source commit) and `files` mapping
relative paths to rows `{reason, lines}`. A finding is identified by file and full
reason with occurrence counts; line numbers are remediation hints. Line shifts
are free. A new path, reason or additional occurrence fails. This is not a general
waiver, and Recipe proposals must still pass every target's naming rules.

A successful audit atomically replaces the baseline with only remaining findings,
under the project writer lock. Fixed entries disappear; a later reintroduction
fails. An unsuccessful audit leaves the baseline unchanged. Audit never creates
or expands it; initial debt requires review. Commit baseline reductions alongside
the source fixes so other checkouts inherit the ratchet. `notes` reports accepted
debt as nonblocking, and `naming_debt` groups it by file. Candidate evaluation uses
the same baseline without changing it; the full audit owns cross-file checks and
baseline reduction. Census caches include both the table and baseline.

`unbake.api.naming_debt(root)` is the read-only per-file list consumed by Recipe
Maker name packs. `recipe name next` also publishes it at
`.unbake/recipe/name/symbol-name-debt.json`, including for the type-name pass.
The file reflects the last successful audit; run audit after a landing to update it.

The text scanner reads inactive branches, declarations, parameters, fields,
macros, explicit labels, alias and section attributes. It does not preprocess
source or invent types. Inline asm retains its separate existing source rule.
A source census is not a native compiler or linker proof.

Sources: [GCC assembler labels](https://gcc.gnu.org/onlinedocs/gcc/Asm-Labels.html),
[Microsoft C identifiers](https://learn.microsoft.com/en-us/cpp/c-language/c-identifiers),
[Microsoft PE/COFF specification](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format),
[Apple Mach-O structures](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h),
[Microsoft decorated names](https://learn.microsoft.com/en-us/cpp/build/reference/decorated-names).
