import struct
import numpy as np

def float16_to_float32(h):
    # Simulate hardware f16tof32
    return np.frombuffer(struct.pack("<H", h), dtype=np.float16).astype(np.float32)[0]

def original_method(q2_bytes, q8_bytes, nb):
    # q2_bytes: nb * 84 bytes
    # q8_bytes: nb * 292 bytes
    lane_sum = [0.0] * 8 # 8 avx_lanes for 1 token (b=0)

    for block in range(nb):
        q2_block = q2_bytes[block * 84 : (block + 1) * 84]
        q8_block = q8_bytes[block * 292 : (block + 1) * 292]

        # d8
        d8 = struct.unpack("<f", q8_block[:4])[0]
        # q8 quants
        q8_qs = struct.unpack("<256b", q8_block[4:260])
        # q8 bsums
        q8_sums = struct.unpack("<16h", q8_block[260:292])

        # q2 scales, qs, d, dmin
        scales = q2_block[:16]
        qs = q2_block[16:80]
        d_u16, dmin_u16 = struct.unpack("<HH", q2_block[80:84])
        s_q2_d = float16_to_float32(d_u16)
        s_q2_dmin = float16_to_float32(dmin_u16)

        # scaled_weights
        scaled_weights = [0] * 256
        for element in range(256):
            half = element >> 7
            quarter = (element >> 5) & 3
            quant = (qs[half * 32 + (element & 31)] >> (quarter * 2)) & 3
            scale = scales[element >> 4]
            scaled_weights[element] = (scale & 15) * quant

        for avx_lane in range(8):
            integer_sum = 0
            for segment in range(8):
                first = segment * 32 + avx_lane * 4
                for j in range(4):
                    w = scaled_weights[first + j]
                    q8 = q8_qs[first + j]
                    integer_sum += w * q8

            sum0 = q8_sums[avx_lane * 2]
            sum1 = q8_sums[avx_lane * 2 + 1]
            scale_min0 = scales[avx_lane * 2] >> 4
            scale_min1 = scales[avx_lane * 2 + 1] >> 4
            minimum = sum0 * scale_min0 + sum1 * scale_min1

            d = np.float32(d8 * s_q2_d)
            dmin = np.float32(-d8 * s_q2_dmin)

            # FMA accum
            accum = lane_sum[avx_lane]
            # accum = ggml_fma_f32(dmin, float(minimum), accum)
            # accum = ggml_fma_f32(d, float(integer_sum), accum)
            accum = np.float32(accum + dmin * np.float32(minimum))
            accum = np.float32(accum + d * np.float32(integer_sum))
            lane_sum[avx_lane] = accum

    # reduction
    a = np.float32(lane_sum[0] + lane_sum[4])
    c = np.float32(lane_sum[1] + lane_sum[5])
    e = np.float32(lane_sum[2] + lane_sum[6])
    g = np.float32(lane_sum[3] + lane_sum[7])
    return np.float32(np.float32(a + e) + np.float32(c + g))

def new_method(pre_q2_bytes, q8_bytes, nb):
    # pre_q2_bytes: nb * 288 bytes
    # q8_bytes: nb * 292 bytes
    lane_sum = [0.0] * 8

    for block in range(nb):
        q2_block = pre_q2_bytes[block * 288 : (block + 1) * 288]
        q8_block = q8_bytes[block * 292 : (block + 1) * 292]

        d8 = struct.unpack("<f", q8_block[:4])[0]
        q8_qs = struct.unpack("<256b", q8_block[4:260])
        q8_sums = struct.unpack("<16h", q8_block[260:292])

        # pre-scaled weights (256 bytes)
        scaled_w = q2_block[:256]
        # mins (16 bytes)
        mins = q2_block[256:272]
        # d, dmin (raw uint16 at 272)
        d_u16, dmin_u16 = struct.unpack("<HH", q2_block[272:276])
        s_q2_d = float16_to_float32(d_u16)
        s_q2_dmin = float16_to_float32(dmin_u16)

        for avx_lane in range(8):
            integer_sum = 0
            for segment in range(8):
                first = segment * 32 + avx_lane * 4
                for j in range(4):
                    w = scaled_w[first + j]
                    q8 = q8_qs[first + j]
                    integer_sum += w * q8

            sum0 = q8_sums[avx_lane * 2]
            sum1 = q8_sums[avx_lane * 2 + 1]
            min0 = mins[avx_lane * 2]
            min1 = mins[avx_lane * 2 + 1]
            minimum = sum0 * min0 + sum1 * min1

            d = np.float32(d8 * s_q2_d)
            dmin = np.float32(-d8 * s_q2_dmin)

            accum = lane_sum[avx_lane]
            accum = np.float32(accum + dmin * np.float32(minimum))
            accum = np.float32(accum + d * np.float32(integer_sum))
            lane_sum[avx_lane] = accum

    a = np.float32(lane_sum[0] + lane_sum[4])
    c = np.float32(lane_sum[1] + lane_sum[5])
    e = np.float32(lane_sum[2] + lane_sum[6])
    g = np.float32(lane_sum[3] + lane_sum[7])
    return np.float32(np.float32(a + e) + np.float32(c + g))

def reorg_q2(raw_q2, nb):
    out = bytearray(nb * 288)
    for b in range(nb):
        src = raw_q2[b * 84 : (b + 1) * 84]
        dst = bytearray(288)
        scales = src[:16]
        qs = src[16:80]
        for element in range(256):
            half = element >> 7
            quarter = (element >> 5) & 3
            quant = (qs[half * 32 + (element & 31)] >> (quarter * 2)) & 3
            scale = scales[element >> 4]
            dst[element] = (scale & 15) * quant
        for i in range(16):
            dst[256 + i] = scales[i] >> 4
        dst[272:276] = src[80:84]
        out[b * 288 : (b + 1) * 288] = dst
    return bytes(out)

# Find exact offset of blk.0.attn_conv_proj.weight
with open(r"C:\tmp\dflash-q2.gguf", "rb") as f:
    f.seek(24) # after header
    # Let's search for the tensor name
    content = f.read(100000)
    idx = content.find(b"blk.0.attn_conv_proj.weight")
    print(f"Tensor found at header index {idx}")

# Let's test on 1280 rows of blk.0.attn_conv_proj.weight
with open(r"C:\tmp\dflash-q2.gguf", "rb") as f:
    # From parse_gguf_metadata, dims=(5120, 1280), nb = 20
    # Let's seek to data offset
    f.seek(1611776) # typical data offset
    nb = 20
    diffs = 0
    for row in range(100):
        raw_q2 = f.read(nb * 84)
        if len(raw_q2) < nb * 84: break
        pre_q2 = reorg_q2(raw_q2, nb)

        q8_bytes = bytearray()
        for _ in range(nb):
            d8 = np.float32(np.random.uniform(0.01, 1.0))
            qs = np.random.randint(-128, 127, size=256, dtype=np.int8).tobytes()
            sums = np.random.randint(-500, 500, size=16, dtype=np.int16).tobytes()
            q8_bytes.extend(struct.pack("<f", d8) + qs + sums)
        q8_bytes = bytes(q8_bytes)

        res_orig = original_method(raw_q2, q8_bytes, nb)
        res_new = new_method(pre_q2, q8_bytes, nb)
        if np.isnan(res_orig) or np.isnan(res_new):
            continue
        diff = abs(res_orig - res_new)
        if diff != 0.0:
            print(f"Row {row} diff: orig={res_orig} new={res_new} diff={diff}")
            diffs += 1

print(f"Tested rows. Total diffs: {diffs}")
