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
reserved global underscore namespace. Compiler/SDK declarations in game code
therefore remain reportable debt; library roots explicitly configured by a project
are excluded by the audit as they are for existing source checks.

`unbake audit --against REV` still ratchets existing source rules and additionally
requires the entire tracked, non-library C/header naming census to pass, including
unchanged files and collisions across files. `source.symbol-name` is unwaivable
and also runs in candidate evaluation. The text scanner reads inactive branches,
declaration heads, parameters, fields, macro names, explicit labels, alias
attributes and section attributes. It does not preprocess source or invent types.
Inline asm instructions retain their separate existing source rule. A source
census is not a native compiler or linker proof.

Sources: [GCC assembler labels](https://gcc.gnu.org/onlinedocs/gcc/Asm-Labels.html),
[Microsoft C identifiers](https://learn.microsoft.com/en-us/cpp/c-language/c-identifiers),
[Microsoft PE/COFF specification](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format),
[Apple Mach-O structures](https://github.com/apple-oss-distributions/xnu/blob/main/EXTERNAL_HEADERS/mach-o/loader.h),
[Microsoft decorated names](https://learn.microsoft.com/en-us/cpp/build/reference/decorated-names).
