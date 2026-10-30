# module-runs

## Rules
- An original object MUST be one contiguous ROM run per section, so layout MUST create one module per contiguous run, refined to be contiguous in every version that holds it.
- The link script MUST list each module's input sections in ROM order per output segment.
- A PC16 branch into another module's code MUST merge those modules into one original object; layout tooling MUST derive module boundaries from runs and MUST NOT force an interleaved group into one file.
