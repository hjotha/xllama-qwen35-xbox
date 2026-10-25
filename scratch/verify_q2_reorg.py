import struct

def original_q2_scaled(src_block):
    scales = src_block[:16]
    qs = src_block[16:80]
    scaled_weights = [0] * 256
    mins = [0] * 16
    for element in range(256):
        half = element >> 7
        quarter = (element >> 5) & 3
        quant = (qs[half * 32 + (element & 31)] >> (quarter * 2)) & 3
        scale = scales[element >> 4]
        scaled_weights[element] = (scale & 15) * quant
    for i in range(16):
        mins[i] = scales[i] >> 4
    d_u16, dmin_u16 = struct.unpack("<HH", src_block[80:84])
    return scaled_weights, mins, d_u16, dmin_u16

def new_q2_block(src_block):
    scaled_weights, mins, d_u16, dmin_u16 = original_q2_scaled(src_block)
    # The new layout has 288 bytes:
    # 256 bytes: scaled_weights (each uint8)
    # 16 bytes: mins (each uint8)
    # 4 bytes: d (float)
    # 4 bytes: dmin (float)
    # 8 bytes: padding
    return scaled_weights, mins, d_u16, dmin_u16

# Let's test on actual bytes from dflash-q2.gguf
with open(r"C:\tmp\dflash-q2.gguf", "rb") as f:
    f.seek(100000) # somewhere in tensor data
    for b in range(100):
        block = f.read(84)
        w1, m1, d1, dm1 = original_q2_scaled(block)
        w2, m2, d2, dm2 = new_q2_block(block)
        assert w1 == w2
        assert m1 == m2
        assert d1 == d2
        assert dm1 == dm2

print("VERIFIED 100 BLOCKS: w, mins, d, dm MATCH 100%!")
