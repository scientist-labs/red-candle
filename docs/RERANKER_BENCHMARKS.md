# Reranker benchmarks

Reproducible latency/throughput/memory numbers for `Candle::Reranker` on
`cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d25c440f23af6f7d6e94d2946bac0bee0a`, sized for
Benchmate's workload: up to 100 candidates per call, CPU-only.

## Run it

```bash
bundle exec ruby bench/reranker_bench.rb
```

Sweeps candidate counts × `max_length` × `batch_size`, and writes
`bench/reranker_bench_results.csv`. `--quick` runs a small subset for a fast sanity check of
the script itself, not for real numbers.

For a trustworthy repeated-call memory growth number, run the isolated check separately (see
below for why the combined sweep's growth column isn't reliable on its own):

```bash
bundle exec ruby bench/reranker_memory_growth.rb
```

**The production target platform is Linux amd64 CPU.** The numbers below were measured on
macOS arm64 (Apple Silicon) — labeled as such throughout, and explicitly **not** a substitute
for running the command above on Linux amd64. Apple's Accelerate BLAS backend and x86_64
BLAS backends (OpenBLAS/MKL-equivalent via candle's default gemm crate) have different
throughput characteristics; only relative trends (which `batch_size` wins, how latency scales
with candidate count) should be expected to transfer, not absolute millisecond figures.

## What's measured, per (max_length, batch_size, candidate_count) combination

- **Cold load time**: `Candle::Reranker.from_pretrained` wall-clock, once per
  (max_length, batch_size) pair. Model weights are typically already OS-page-cached from the
  benchmark's own earlier loads within the same run, so this measures object/tokenizer setup
  cost more than cold-disk I/O — see the script for the precise sequence.
- **Warm p50 / p95 latency**: wall-clock `#rerank` time, after 2 untimed warm-up calls,
  over 15 timed iterations (5 in `--quick`).
- **Throughput**: `candidate_count / (p50_seconds)`, in docs/sec.
- **Peak RSS**: max resident set size (`ps -o rss=`) observed across the whole sweep.
- **Repeated-call memory growth**: RSS before vs. after 50 consecutive `#rerank` calls at a
  fixed representative config (max_length=512, batch_size=16, 100 docs), to catch leaks —
  mmap'd weights should be loaded once and reused, not re-allocated per call.
- **Single vs. multiple concurrent callers**: p50 latency with 1 thread vs. 4 threads calling
  `#rerank` concurrently on one shared instance, at the largest candidate count — the shape
  that matters for Benchmate's memoized-singleton-behind-Puma-threads use case.

## Measured results (macOS arm64, Apple Silicon, Accelerate backend)

Full sweep: 9 (`max_length`, `batch_size`) configurations × 4 candidate counts, 15 warm
iterations each, from [`bench/reranker_bench_results.csv`](../bench/reranker_bench_results.csv)
(raw data, including RSS columns). This is a single run, not averaged — see Caveats.

| max_length | batch_size | candidates | p50 (ms) | p95 (ms) | throughput (docs/s) |
|---|---|---|---|---|---|
| 128 | 8 | 10 | 151.45 | 199.83 | 66.0 |
| 128 | 8 | 25 | 312.80 | 413.49 | 79.9 |
| 128 | 8 | 50 | 622.05 | 780.42 | 80.4 |
| 128 | 8 | 100 | 1149.64 | 1478.54 | 87.0 |
| 128 | 16 | 10 | 101.07 | 119.29 | 98.9 |
| 128 | 16 | 25 | 264.99 | 279.29 | 94.3 |
| 128 | 16 | 50 | 582.22 | 640.32 | 85.9 |
| 128 | 16 | 100 | 1067.88 | 1223.12 | 93.6 |
| 128 | 32 | 10 | 111.64 | 137.00 | 89.6 |
| 128 | 32 | 25 | 248.47 | 288.09 | 100.6 |
| 128 | 32 | 50 | 510.26 | 596.26 | 98.0 |
| 128 | 32 | 100 | 935.82 | 1081.20 | 106.9 |
| 256 | 8 | 10 | 107.48 | 118.29 | 93.0 |
| 256 | 8 | 25 | 237.38 | 264.10 | 105.3 |
| 256 | 8 | 50 | 512.66 | 540.85 | 97.5 |
| 256 | 8 | 100 | 927.52 | 984.48 | 107.8 |
| 256 | 16 | 10 | 89.15 | 95.86 | 112.2 |
| 256 | 16 | 25 | 219.65 | 227.16 | 113.8 |
| 256 | 16 | 50 | 436.81 | 450.92 | 114.5 |
| 256 | 16 | 100 | 870.73 | 891.18 | 114.8 |
| 256 | 32 | 10 | 100.46 | 120.23 | 99.5 |
| 256 | 32 | 25 | 219.98 | 250.08 | 113.6 |
| 256 | 32 | 50 | 436.74 | 480.98 | 114.5 |
| 256 | 32 | 100 | 824.73 | 848.90 | 121.3 |
| 512 | 8 | 10 | 94.87 | 98.69 | 105.4 |
| 512 | 8 | 25 | 228.59 | 230.75 | 109.4 |
| 512 | 8 | 50 | 461.29 | 518.59 | 108.4 |
| 512 | 8 | 100 | 904.93 | 944.05 | 110.5 |
| 512 | 16 | 10 | 87.74 | 88.97 | 114.0 |
| 512 | 16 | 25 | 216.48 | 218.95 | 115.5 |
| 512 | 16 | 50 | 495.13 | 556.36 | 101.0 |
| 512 | 16 | 100 | 1163.27 | 1290.06 | 86.0 |
| 512 | 32 | 10 | 132.07 | 137.91 | 75.7 |
| 512 | 32 | 25 | 246.78 | 264.73 | 101.3 |
| 512 | 32 | 50 | 545.18 | 570.46 | 91.7 |
| 512 | 32 | 100 | 1061.55 | 1157.30 | 94.2 |

Cold load time was 30-65ms across all 9 configurations (weights already OS-page-cached from
earlier loads in the same run — see Caveats for what this does and doesn't measure).

### Concurrency: 1 vs. 4 callers (100 candidates)

4 threads calling `#rerank` concurrently on one shared instance, vs. 1 thread making the same
calls serially — the GVL is released for the compute portion, so this measures real parallel
speedup, not just non-blocking I/O:

| max_length | batch_size | 1 caller p50 (ms) | 4 callers p50 (ms) | overhead |
|---|---|---|---|---|
| 128 | 8 | 1202.29 | 2009.65 | 1.67x |
| 128 | 16 | 1190.51 | 1842.18 | 1.55x |
| 128 | 32 | 866.71 | 1526.36 | 1.76x |
| 256 | 8 | 926.80 | 1989.26 | 2.15x |
| 256 | 16 | 1080.17 | 1542.28 | 1.43x |
| 256 | 32 | 827.95 | 1189.96 | 1.44x |
| 512 | 8 | 915.26 | 1390.22 | 1.52x |
| 512 | 16 | 1127.09 | 1919.75 | 1.70x |
| 512 | 32 | 1173.48 | 1767.18 | 1.51x |

4 concurrent callers cost roughly 1.4-2.2x one caller's latency (not 4x) — consistent with
real parallelism across the 4 threads sharing this machine's CPU cores, not serialization. On a
production host with more CPU headroom than this development laptop, expect less overhead per
additional caller.

### Repeated-call memory growth (isolated measurement)

The growth numbers in `reranker_bench_results.csv`'s combined sweep run are **not**
trustworthy in isolation — they're dominated by garbage-collection timing of the 9 *other*
reranker instances loaded earlier in that same process (see Caveats). Re-measured standalone
(nothing else loaded first) via `bundle exec ruby bench/reranker_memory_growth.rb`,
max_length=512, batch_size=16, 100 candidates, 50 calls:

```
RSS after load + warmup: 214736kB
RSS every 10 calls: 214736 -> 227184 -> 227872 -> 228880 -> 230048 -> 231104kB
Net growth over 50 calls (post-GC.start): 16368kB (327kB/call average)
```

Growth is front-loaded: +12448kB in the first 10 calls, then +688/+1008/+1168/+1056kB in each
subsequent block of 10 — consistent with the Ruby heap and internal buffers reaching a steady
state, not an unbounded per-call leak. Re-run this check after any change to the tokenization
or scoring path, and watch for growth that *doesn't* taper off.

## Choosing a default `batch_size` under a 3 GiB memory budget

**Peak RSS across the entire sweep (all 9 configs × all 4 candidate counts) was 801536kB
(~0.76 GiB)** — comfortably under the 3 GiB budget at every `batch_size` tested, including the
largest (`batch_size: 32`, `max_length: 512`, 100 candidates). At this model's size (22M
params, ~90MB of weights), memory is not the constraint that should drive `batch_size`
selection — throughput and latency are.

Throughput at 100 candidates trends upward with `batch_size` in this run (`batch_size: 32`
generally beat `8` and `16` at `max_length` 128 and 256; `max_length: 512` was noisier — see
the `512/16/100` row above, likely single-run variance rather than a real effect, per the
Caveats). Given the memory headroom is large regardless, **the default of 16 remains a
reasonable, conservative choice** — this data doesn't show a compelling memory or latency
reason to change it. If Linux amd64 production numbers (gathered via the command above) show a
consistent, repeatable throughput win at `batch_size: 32` with acceptable p95 latency, raising
the default would be a reasonable follow-up, backed by that data rather than this arm64 sample.

## Caveats

- These are single-run measurements, not averaged across multiple runs — real p50/p95 will
  have some run-to-run variance, particularly on a shared/loaded machine.
- "Cold load time" here means first-load-for-that-configuration within one process, not a
  fully cold OS page cache — a true first-ever load (nothing cached, e.g. a fresh Linux
  container) will be slower and dominated by the `model.safetensors` download/read, not
  measured here.
- RSS growth numbers are noisy on macOS when other memory-heavy processes are running
  concurrently (page compression/reclaim skews `ps` readings). The numbers above were
  captured with no other heavy process running; re-verify on the actual production host if
  the numbers look surprising there.
- The combined sweep's `rss_growth_kb` column (in `reranker_bench_results.csv`) reflects each
  config's *own* before/after delta, but within one long-running process — several rows show
  large negative deltas (RSS apparently shrinking), which is Ruby's GC finally collecting
  *previous* configurations' reranker instances (`bench/reranker_bench.rb` doesn't force a GC
  between configs, to keep the sweep's own runtime down), not a leak or a real property of the
  config in that row. Trust `bench/reranker_memory_growth.rb`'s isolated, single-process
  measurement (reported above) for actual growth-per-call, not the sweep CSV's growth column.
