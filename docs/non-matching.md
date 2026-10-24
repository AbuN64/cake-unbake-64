# NON_MATCHING overrides

`unbake check` refuses `source.non_matching_override` when a `NON_MATCHING`
conditional changes a function that the ROM side defines in C. The check also
follows file-local macro aliases, so a host-only constant or ABI name defined
outside a function cannot hide an override. Library modules receive this check
as well. A complete function fallback whose ROM side is assembly is allowed.
The check is structural; ROM byte equality remains the build's responsibility.
Version guards are evaluated separately for every configured version. Unknown
non-version conditions are conservatively retained. The existing tree ownership capture identifies ROM-side C defined by an
include, so replacing that included implementation is also refused. Macros
from other files are not expanded by this source-only rule.

Existing overrides require reviewed project facts in
`config/non-matching-baseline.json`. Its schema is the same shape as the naming
baseline: `schema: 1`, a forty-character `revision`, and `files` mapping paths to
`{reason, lines}` entries copied from the check's findings. Reasons include the
function and conditional token fingerprint. New paths, functions, conditional
contents or extra occurrences fail; moving a block's lines does not. There is
no automatic baseline creation or increase. A successful full check only
shrinks the baseline. A path-scoped or failed check never rewrites it.

`unbake.non_matching.scan(path, text, versions, assembled)` exposes the same
classification for census tooling: A is a C fallback for an assembly function,
B is an override of ROM-side C, and C is other (including declarations and
host-only helpers). `assembled` optionally supplies version-to-symbol names
from the existing tree ownership capture, including assembly in another unit.
One row represents a complete conditional group, not each `elif` arm; nested
groups remain separate rows. A mixed group is B if any member is B, otherwise
A if any member is A. Each member retains its per-version classification.
