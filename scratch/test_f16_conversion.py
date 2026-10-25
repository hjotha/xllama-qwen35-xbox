import struct
import numpy as np

# Let's test all 65536 possible 16-bit half values
# GGML uses IEEE-754 half float conversion.
# Does numpy float16 to float32 match?
diffs = 0
for u in range(65536):
    raw16 = struct.pack("<H", u)
    f16 = np.frombuffer(raw16, dtype=np.float16)[0]
    f32 = np.float32(f16)
    # Check if finite
    if np.isfinite(f32):
        # packed back
        raw32 = struct.pack("<f", f32)

print("FP16 test done")
