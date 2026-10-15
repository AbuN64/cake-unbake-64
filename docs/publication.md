# Publishing a measured group

Publish accepts several draft files. `--compare` runs the public comparison and
records the current source measurement before each native publication gate.
`--events` flushes each successful commit receipt immediately, including its
proved versions and input hashes. A refused sibling does not erase that prefix.

```sh
unbake --config host.toml publish --compare --events --push origin build/work/a/a.c build/work/b/b.c
```

`--push` uses the host's `[publish].branch` and ordinary Git credentials. A
concurrent update triggers fetch/rebase and native reconciliation of changed
function/version inputs. Shared headers are read once per snapshot; unrelated
functions and unchanged versions retain their proof. A mismatch refuses the
push and restores the prior local commit, so a retry must reconcile again.
Conflicting Git edits abort the rebase and retain the local commits. Neither
path force-pushes. A transport refusal names its cause and leaves the commits
available for retry:

```sh
unbake --config host.toml publish --push origin
```

An interrupted command exits 130 with `status: interrupted`, `key: null` and
`retryable: true`. It records no held cause. Publication returns its successful
prefix and the still-ready suffix; its Next command names only that suffix.
A committed prefix remains committed if subsequent maintenance is interrupted.
This classification does not introduce a resumable proof cache or attempt store.

Publication records destination files with the existing journal immediately
before changing them. A rejected or interrupted item restores its own writes,
including newly created providers and deleted headers. It preserves edits that
were present before the invocation. Each accepted Git commit advances the
checkpoint; a later refused sibling or maintenance step restores only the
uncommitted suffix. Byte content, file modes, modification times and symlink
entries are restored without scanning or copying the entire project.

Equal guarded aggregate definitions and typedefs use one installed shared owner.
Comparison stages the same ownership view without editing the draft headers.
Publication reconciles both its initial catalogue and final proposed headers
before proving their existing consumers. Type/header regeneration uses the same
owner plan before caching rendered bytes and after final namespace projection;
its manifest retains canonical imported headers through the next installation.
Different declarations, conflicting
typedef dependencies, conditional uncertainty and import cycles refuse with
provider locations. The native source, ABI, consumer and ROM proof gates remain
required.

Publication needs no separate identity/reconciliation driver. The public compare
checks a draft against its configured holding placements; setup's
`--redo-symbol-matching` reviews correspondence when those placements need
changing. Fresh `publish --compare` measurements replace phase-wide freshness
claims, and native publication proves the exact current input view. The final
Result's landed/failed/ready lists and the `fn.committed` stream replace a separate
publication evidence matrix. Historical raw-source searches remain provenance;
their scores and names cannot substitute for public comparison or publication.

Use a public host configuration to bound cores, workers, parent/worker memory
reservations and cache memory. For aggregate enforcement across commands, run
`unbake resources domain.toml` and select that domain in the host configuration
as described in [resource domains](resource-domain.md). This provides public
command admission without patching configuration loading in a private launcher.

The former publication helpers map to public commands as follows:

| Helper role | Public command |
| --- | --- |
| `bulk_identity.py` publication queue and attempt stamps | `unbake publish --compare --events --push REMOTE FILE...` |
| `run_admitted_tool.py` resource envelope | `unbake --config HOST publish ...`, with `unbake resources DOMAIN` for shared admission |
| `phase_integrity.py` current-input eligibility | `unbake publish --compare ...`; after concurrent changes, `publish --push REMOTE` rechecks affected scopes |
| `stream_reconcile.py` publication aggregation | `unbake publish --events ...` commit stream and final landed/failed/ready lists |
| `MATCHED-GROUP-RETRY.py` fresh compare then same-pass publication/push | `unbake publish --compare --events --push REMOTE FILE...` |

These commands own publication correctness. They do not import historical private
raw-scan evidence into the attempts store or promote an unverified raw identity.

Progress reports use the current split member inventory. Only unguarded C is exact;
startup and original assembly remain code in the denominator. Address aliases count
once. Declared non-code subsegments contribute data, while opaque top-level binary
assets are excluded. Data matching has no owning proof yet: its state is unknown
and its verified numerator is zero.

A retained draft needs its committed receipt, exact source hash, selected compiler,
and all current holding versions. Historical best scores never earn current progress.
Similarity may be null; the objdiff fuzzy scalar includes only known similarity,
so it is a lower bound. The separate `report-state.json` records draft byte/function
coverage, current scores including nulls, source pins and artifact mappings. A missing
receipt requires reviewed source-bound prior evidence and compilation proof for every
holding version. The publication owner performs native reconciliation before a new
publication transition; the report validator refuses a missing record and never
manufactures eligibility from history. Prior receipt evidence is preserved by the
owning offline migration, with unavailable dependency proof kept explicit.

The owning build generator writes `tools/report-verifier.zip`, including the exporter
and its source parser. Both generated CI workflows check its immutable content hash
and run `python3 -m unbake.report.verify` before copying reports. Verification needs no
ROM or compiler executable. It checks strict JSON, receipt agreement, full inventories,
canonical report bytes, README figures and the source/coverage manifest. Each report
artifact retains the established version naming and gains a companion manifest with
the immutable checkout revision, tool payload hash and report hash. Consumers must
compare those identities with the requested revision before accepting numerical repair.

Publication requires a complete current strict proof for every required or already
published holder: identical bounded linked bytes, zero positional and size
changes, resolved literal/relocation placement, no source-rule preconditions,
and the current source/recipe/target/dependency identities. Aligned similarity is
separate diagnostic evidence. Nonexact candidates remain editable history.

The receipt lists the persistent source/header/provider, config, split, symbol,
generated rule/helper, and compiler-pin inputs even when publication did not
change their bytes. The existing publisher stages this bounded closure and reads
it back from the committed tree. Default compiler selections retain unit options.

Immutable producer qualification uses recorded native input, key, CAS, object,
link and bounded ROM identities. Recursive used Make expressions establish recipe
meaning; variable spelling and unrelated inventory are not producing commands.
Reviewed migration can bind a semantic recipe certificate only to byte-verified
original inputs and an existing immutable event. That event is never recaptured
or relabeled. Changed used inputs refuse with a named provenance cause. Resource
storage and execution bindings stay separate; missing historical assembler input
identity remains unavailable, even when an output happens to equal ROM.
