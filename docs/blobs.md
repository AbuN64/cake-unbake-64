# blobs

## Rules
- The blob catalog MUST use exactly five content types (graphics, model, audio, world, text), keep the finer kind in each blob's `format`, and MUST NOT add a catch-all type.
- The blob catalog MUST NOT be one file per asset, and SHOULD be split per type into blobs/<type>.toml with evidence hashes deduplicated once and cross-version pointers derived.
- RSP microcode MUST NOT be a blob; it belongs to the `platform` report row together with boot.
- The tracked ROM-content kind MUST be named `blobs` in code, reports and docs, and MUST NOT be called assets or registry.
- The `data` and `rodata` report rows and the INCLUDE_DATA and INCLUDE_RODATA macros MUST keep those names, following decomp community convention.
