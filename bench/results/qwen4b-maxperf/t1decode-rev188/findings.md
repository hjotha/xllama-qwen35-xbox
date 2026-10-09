# Decode thread-count A/B (rev188 LTCG, config 256/64)

Same-build interleaved A/B, profile OFF, 3 reps per arm per cell, only the
thread variable changed:

| arm                 | decode chat256     | decode std512      | prefill std512    |
| ------------------- | ------------------ | ------------------ | ----------------- |
| t2 (baseline)       | 27.98 t/s          | 28.05 t/s          | 90.3 t/s          |
| t1 decode (batch 2) | 27.59 (-1.5%)      | 27.69 (-1.4%)      | 89.5 (-1.3%)      |
| t1 decode + batch 1 | **24.68 (-11.8%)** | **24.36 (-13.2%)** | **74.7 (-17.3%)** |

- Reducing decode threads does not help (t1 null-to-worse); batch=1 is much
  worse on both phases -> the tiny-node thread-barrier hypothesis is refuted
  and 2 threads is the optimum (t6 was already known worse).
- The t2 medians here pool 6 samples (previous + current round) and agree with
  the 3-rep medians (28.0 decode / 90.7 prefill).
- No change applied; the accepted config keeps n_threads=2 and
  n_threads_batch=2.
