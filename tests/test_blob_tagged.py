"""Typed tagged streams preserve every byte and refuse ambiguous or incomplete records."""

import pytest

from unbake import blob_decode


def test_tagged_stream_round_trip_and_rejections():
    layout = {
        "root": "Commands",
        "null_values": [],
        "types": {
            "Commands": {
                "tagged": {
                    "terminator": 0,
                    "alignment": 8,
                    "cases": [
                        {"mask": 128, "value": 128, "type": "Pair"},
                        {"mask": 224, "value": 32, "type": "Pair"},
                        {"mask": 224, "value": 64, "type": "Pair"},
                        {"mask": 224, "value": 96, "type": "Five"},
                    ],
                }
            },
            "Pair": {"fields": "11"},
            "Five": {"fields": "11111"},
        },
    }
    data = bytes([0x81, 1, 0x22, 2, 0x43, 3, 0x64, 4, 5, 6, 7, 0, 0, 0, 0, 0])
    result = blob_decode.Graph(data, layout).finish()
    assert result["sha256"] == result["reemit_sha256"]
    assert [r["type"] for r in result["records"]] == ["Pair", "Pair", "Pair", "Five", "terminator"]
    for bad, message in [
        (b"\x01\0", "exactly one"),
        (b"\x81", "outside extent"),
        (b"\x81\x01", "lacks its terminator"),
        (bytes(7), "unproved padding"),
        (bytes(8) + b"x", "unproved padding"),
    ]:
        with pytest.raises(ValueError, match=message):
            blob_decode.Graph(bad, layout).finish()
    spec = layout["types"]["Commands"]["tagged"]
    spec["cases"].append({"mask": 255, "value": 0x81, "type": "Pair"})
    with pytest.raises(ValueError, match="exactly one"):
        blob_decode.Graph(data, layout).finish()
    spec["cases"].pop()
    layout["types"]["Pair"] = {"fields": "2"}
    with pytest.raises(ValueError, match="begin with a byte"):
        blob_decode.Graph(data, layout).finish()
