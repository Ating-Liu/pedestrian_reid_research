# Performance Tuning Notes

## Current Defaults

The training pipeline is configured to make the GPU the primary bottleneck:

- `--use-amp true`
- `--allow-tf32 true`
- `--cudnn-benchmark true`
- `--channels-last false`
- `--cuda-prefetch true`
- `--fused-optimizer true`
- `--pin-memory true`
- `--persistent-workers true`
- `--prefetch-factor 4`
- `--num-workers 12`

These defaults target the current Windows + RTX 4080 Laptop GPU setup. They should be kept unless a throughput check shows the data pipeline is overloading memory or worker startup becomes unstable. `channels_last` was tested and is disabled by default because it slowed the current ReID model on this machine.

## What Was Optimized

- Data loading now keeps workers alive across epochs by default.
- CPU-to-GPU image transfer uses pinned memory, non-blocking copies, and optional CUDA prefetching.
- `channels_last` remains available as an opt-in flag, but it is disabled by default after profiling showed a large slowdown on the current hardware/model combination.
- Training uses fused Adam when the installed PyTorch/CUDA build supports it.
- Triplet loss no longer loops over samples in Python; hardest positive/negative mining is vectorized on the tensor device.
- Training loss and accuracy statistics are accumulated on GPU and synchronized only for logging, avoiding per-iteration CPU/GPU stalls.
- Evaluation computes Market-1501 and CUHK03-NP distance matrices on GPU by default, falling back to CPU only when the matrix is too large.

## Recommended Training Command

```powershell
py -3.12 scripts\train.py --data-root datasets --dataset-name market1501 --experiment-name market1501_fast_check --device cuda --batch-size 64 --num-instances 4 --num-workers 12 --prefetch-factor 4 --persistent-workers true --pin-memory true --use-amp true --channels-last false --cuda-prefetch true --fused-optimizer true --allow-tf32 true
```

If GPU utilization is still low, test `--num-workers 16` and `--num-workers 24`, but do not assume the highest worker count is fastest. On Windows, too many workers can increase scheduling and memory overhead.

Current smoke-check result on `market1501` local auxiliary residual with `batch_size=64`:

```text
channels_last=false, cuda_prefetch=true, fused_optimizer=true, persistent_workers=true:
722.5 img/s, avg_data_wait_seconds 0.0004, max CUDA memory 3.45 GB
```

The same test showed `channels_last=true` dropping throughput to about `166-168 img/s`, so it should stay off for this project unless a future PyTorch/CUDA build changes the result.

## Reading The Throughput Log

Training now prints per-epoch throughput:

```text
Epoch [1/60] throughput 120.0 img/s (12416 images in 103.5s)
```

It also prints periodic batch-level diagnostics:

- `Img/s`: last printed step throughput.
- `Data`: average time spent waiting for the next batch.
- `Step`: average total iteration wall time.

If `Data` is close to `Step`, the pipeline is data-bound. Increase workers, keep `persistent-workers true`, and avoid heavy CPU-side transforms. If `Data` is much smaller than `Step`, the GPU/model is the main bottleneck, which is the desired state.

## Evaluation

For normal Market-1501 and CUHK03-NP evaluation, keep:

```powershell
--eval-gpu-distance true --eval-gpu-distance-max-elements 200000000
```

For very large datasets such as MSMT17, lower the max element threshold if GPU memory is tight.
