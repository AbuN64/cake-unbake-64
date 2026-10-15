# Repair imported resident storage blocks

`source.resident-storage` refuses function files containing manufactured
`unbake_rodata_*` storage. Identical linked instructions do not prove that those
definitions own the resident bytes. Original ROM slices supply resident data;
the generated per-unit link keeps the proved text and discards other sections.
Absolute references still need the version's canonical symbol and ROM mappings.

For imported drafts containing the marked, unused storage blocks accepted by
`unbake.layout.resident.deleted`, use one transformation for the whole held group:

```python
from unbake.layout import resident

original = candidate.read_text()
repaired = resident.deleted(original, candidate.stem)
```

Save `repaired` as the candidate in the source-adapter workflow. Keep its code,
imports, externs, symbol aliases, resident mappings, and original ROM data rows.
Stop emitting these redundant blocks when importing future drafts. The existing
`resident` step uses the same transformation for published sources.

The transformation refuses malformed blocks and any manufactured name still
used outside its block, including address-taking. A refusal requires a real
owner/import or literal-pool repair backed by data and relocation proof. Renaming
the definitions or relocating them into a header does not establish ownership.

Run public compare for every actual holding version after transformation, then
use ordinary publication so strict native placement and all-consumer proofs
remain in force. Preserve distinctions between a version-dependent alias,
another function's resident row, and the compiler's own literal pool.

The real RageWars regression fixture records six imported sources and 29 holding
cases. Native compilation, strict placement, and linking before and after this
transformation produced the same ROM instructions and text relocations. Every
active manufactured definition matched existing mapped ROM bytes, while the
linked unit allocated only text. Some definitions copied another function's
resident row; removing the unused copies preserved that row's ownership. One
source has four holding versions, so the proof does not treat the project's five
versions as interchangeable.
