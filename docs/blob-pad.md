# Pad blobs

The linker pads an object up to the alignment of the next one with zero bytes. Those bytes are no code, data or
content and no source defines them, so they stay `unknown` until a reader proves them. `unbake blobs scan pad` is that
reader: it records the zero runs between aligned objects as `fill` blobs, up to a stated limit.

## The pad rule

Over each unknown gap (the bytes between two accounted objects that no class holds):

- a gap that is one zero run is pad: the run lies between the two objects;
- a gap that starts with at least 16 zero bytes (`blobs.RUN_MIN`, the shortest run the automatic fill accepts inside a
  longer gap) is pad up to its first nonzero byte, when that byte begins at a 4-byte boundary: the run pads the
  previous object up to the alignment of the object that starts there. A shorter leading run is the tail of a word or
  record as likely as padding: it stays unknown and is not listed;
- a run longer than the **reader limit, 512 bytes** (`blob_pad.LIMIT`), is refused, however it lies: past that, a
  zero stretch is as likely to be a zeroed table as padding. The widest pad measured in RageWars is 343 bytes;
- zeros inside a gap, or at its end behind other unknown bytes, are no pad and stay unknown, and so does every
  nonzero byte of the gap. Constant runs of another value (a `0x0a` between strings) are no pad.

Nothing is credited by a threshold or a similarity. The reader rebuilds the range (`length` zero bytes) and the range
closes only when the rebuilt bytes equal the ROM bytes, so the same rule decides in `scan`, in `blobs prove` and in
`blobs verify`. What the rule refuses is listed under `refused` with its reason and stays unknown.

## The row

```toml
[blob."pad.us"]
type = "fill"
format = "pad"
state = "decoded"
limit = 512
stored_size = 347
rom."us" = [[0xde30c, 0xde310], [0x15fca9, 0x15fe00]]
evidence = [{ artifact = "pad-receipt.json", sha256 = "...", pointer = "/us" }]
```

`limit` is part of the row: `unbake blobs --json` lists every pad row with it, and the checks refuse a pad row
without `type = "fill"`, `format = "pad"`, `state = "decoded"` and a `limit`, a range longer than its `limit`, and (in
`verify`) a range whose ROM bytes are not zero. Proven fill that no row records (the automatic B1 rule of
`blobs.fill`) is unchanged.

## Use

```
unbake blobs scan pad --json     # per version: limit, pad ranges, pad_bytes, refused, unknown_before/after
unbake blobs prove REQUEST RECEIPT   # a request row {reader: "pad", region: [a, b], limit, evidence}
```
