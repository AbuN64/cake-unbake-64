# SN page blobs

The builder writes the debug stream of a link in sectors. A sector the later link did not overwrite keeps the last
write of an earlier link: the tail of an object file's code record, the debug records after it, and the head of the
next code record, cut where the sector ends. Those bytes belong to no object and no class, so they stay `unknown`
until `sn-page` proves them. `unbake blobs scan stale` lists the candidate sectors as `sn-page` rows.

## The page rule

A page is one aligned sector of `snpage.SECTOR` (512) bytes. It holds, in this order and nothing else:

- **front**: the payload tail of a code record whose header lies in the sector before. Every word on the 4-byte grid
  that ends at the first record is a canonical VR4300 instruction (`mipsstream.canonical`); at least 8 words and 4
  distinct ones. Up to 3 bytes before the grid are the cut word and are kept as `lead`;
- **records**: at least 8 whole records of the SN object dialect (`snsym.object_record`), chained from the first
  record to the sector end or to a cut code record. At least two thirds are line records (tags 0x32 to 0x3C), and
  the code offsets of patch and line records never go back inside a run between code records and section switches.
  Code read as records comes out mostly as patches with scattered offsets, so it fails these two rules. The first
  record is the lowest start with code before it from which the chain lands;
- **cut**: a code record (tag 0x02) whose u16 length is a whole number of words and runs past the sector end, with
  its header inside the sector and the payload words that are in it canonical instructions.

A sector that is no such page is refused with the rule it breaks and stays unknown.

## The receipt

The receipt reconstructs the page as typed definitions: the front as instruction words with their assembly, the
records as tags and fields, the cut as its header and payload words. It also lists where the front's words occur in
the project's other versions, how many words are equal and the first word that differs. That comparison is a
measurement: the proof is the structure above. A row credits a region inside the page.

## Use

```
unbake blobs scan stale --json         # sn-page rows: {reader: "sn-page", region}
unbake blobs prove REQUEST RECEIPT     # a request row {reader: "sn-page", region: [a, b], evidence}; the page is the sector the region lies in
```

The row is a `leftover` blob with `format = "sn-page"` and `state = "decoded"`.
