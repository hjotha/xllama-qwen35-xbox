import struct

def read_str(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")

def skip_val(f, vtype):
    sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    if vtype in sizes:
        f.seek(sizes[vtype], 1)
    elif vtype == 8:
        read_str(f)
    elif vtype == 9:
        elem_type, count = struct.unpack("<IQ", f.read(12))
        for _ in range(count):
            skip_val(f, elem_type)
    else:
        raise ValueError(f"Unknown type {vtype}")

with open(r"C:\tmp\dflash-q2.gguf", "rb") as f:
    magic = f.read(4)
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    for _ in range(n_kv):
        k = read_str(f)
        vtype = struct.unpack("<I", f.read(4))[0]
        skip_val(f, vtype)

    GGML_TYPES = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
        6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
        14: "Q6_K", 15: "Q8_K"
    }
    for _ in range(n_tensors):
        name = read_str(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
        ttype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
        tname = GGML_TYPES.get(ttype, str(ttype))
        if "blk.0" in name or "fc" in name:
            print(f"{name:<35} {tname:<8} raw={ttype} dims={dims}")
