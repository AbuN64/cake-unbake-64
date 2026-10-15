# Compiler ownership

The registry owns compiler IDs, versions, families, execution-path kinds, pins,
downloads, finite option spaces, external decompiler/scorer targets, extraction modes
and persisted build-key compatibility. Kind selects a native execution path;
compiler identity selects preprocessing semantics and assembler options.

A family exports `adapter()` implementing `Family`. The protocol covers native
commands and setup probes; preprocessing, analysis and dependency behavior;
object padding, target shape, move idioms and constant selectors; decompiler
register adaptation; allocation/scheduling production, parsing and hints; and
native runtime-helper identity. Unsupported scheduling returns explicit
unavailable evidence. Empty registered external targets mean unsupported.

An additional compiler in an existing family needs only a registry entry. A
new family additionally needs its module and complete adapter. Families sharing
one execution-path kind must agree on native templates, host preprocessing and
padding; each compiler retains its own flag semantics. Driver-template changes
need a distinct execution-path kind, not central caller edits.

IDO native compile commands and setup probes share the VR4300 multiply-hazard
policy `-Wab,-r4300_mul`. The family inserts it before ordered caller codegen
flags; preprocessing keeps its existing language and macro inputs. The shared
template supplies both Make and runner commands, whose argv remains part of the
object cache identity.

The object readers, instruction analysis and O32 transport remain target facts:
big-endian MIPS ELF32 relocations, ISA instructions and the selected target ABI.
Authored C extension syntax and preprocessor option transport are source/provider
contracts, independent of the compiler selected for native code. Family-specific
ABI differences and the host analysis provider's dump syntax live in adapters.
Synthetic decompiler constant sections are distinguished from emitted object
sections and are selected through the family contract.

Runtime recognition currently proves one complete signed integer-to-double
implementation and all of its actual ROM literals. Other implementations and
other helper operations require further family evidence. No generic library
catalogue is implied. Conflicting native addresses are refused. Bindings preserve
ROM bytes; code/data split repair remains the layout owner's operation.

Public compiler headers are described by the family protocol. `unbake recompute
compiler-headers` provisions missing providers without ROM extraction or a type
solve; setup uses the same flow. Compatible existing type owners are imported,
conflicting owners are refused and installed SDK or human providers are retained.
Native selectors and reserved source operators come only from the family adapter.
Standard vararg retrieval captures its cursor lvalue once; native evidence covers
target sizes, alignment holes, floating prefixes and independent list copies.

Scoped recipes use config schema 2. Each canonical `src/path.c` unit selects a
pinned compiler and explicit ordered `preprocess`, `compile`, `assemble`, and
`link` options. The resolver applies compiler defaults, version macros, unit,
proved separate-function scope, then trial options. Exclusive groups replace an
earlier value within their phase. Compiler `-G` never changes assembler `-G`.
Unsupported link options and diagnostic-only build switches refuse before work.
Declared per-unit ABI overrides remain valid; speculative ABI/ISA changes require
compatible target/caller evidence. Both GCC and IDO expose the same option,
capability, semantic preprocessing, and retained observation interfaces.

`migrate-state --plan` reads legacy config before strict loading. Its immutable
plan reports unknown fields, effective argv, verified binary copies, missing
inputs, and evidence disposition. `--apply PLAN` verifies every input, preserves
before-images and historical events, reads back the current config, and copies
verified compiler bytes into digest directories. Used installs are never replaced.
Historical comparisons remain explicitly unverified until compared again.

Ordinary Make and tool preprocessing use the same portable semantic validator.
Native keys include the driver/resolver/contracts and exact binary pins. Diagnostic
captures are explicit, isolated, and accepted as observations only when their
final linked bytes equal the ordinary build; they cannot become build options.
