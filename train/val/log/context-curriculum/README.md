# B300 context-curriculum evidence

The three files in `../` are the completed **fast curriculum, seed 0** run. `baseline/` and `gradual/` retain the other two predeclared seed-0 runs. All reached 610 updates and 9,994,240 predictions. The plots contain full 2048-context validation at the ten fixed evaluation points; no checkpoint was selected by validation.

| Run | Final validation CE | B300 training seconds |
| --- | ---: | ---: |
| Baseline, constant 2048 | 4.0255595340448265 | 3506.64 |
| Fast curriculum | 3.9804404553245094 | 4485.72 |
| Gradual curriculum | 3.9829017765381756 | 4183.12 |

```bash
uv run torchrun --standalone --nproc_per_node=8 train/val/run.py \
  --data-dir data --output logs/val-context-fast --seed 0 \
  --context-schedule 1:64,33:128,65:256,97:512,129:1024,161:2048 \
  --eval-every 61
```

For the exact recorded B300 dependency environment and command, see [runtime/README.md](runtime/README.md). The root `uv run` example above uses the normal upstream environment.

Measurements used Python 3.12, PyTorch 2.10.0+cu130 and 8 NVIDIA B300 GPUs. The upstream environment uses cu128 and its record track specifies 8 H100 GPUs. These are B300 measurements for reproduction review, not an official H100 record. Warmup and validation are excluded from training seconds. Complete runtime hardware and allocator-peak metadata remain in the original config/result files.

The completed seed-0 baseline predates the mask-construction hoist needed to compile short contexts. It therefore has a different compiler boundary from the curricula; this comparison does not isolate scheduling from compiler effects. Both curricula use the same compiler path. Auxiliary seed-1/2 baseline/fast pairs use identical mask-hoisted code, initialization/draw seed, and data order within each pair.

The first pair (seed 1) is complete. Both runs reached 610 updates and 9,994,240 predictions with exit 0:

| Seed-1 confirmation | Final validation CE | B300 training seconds |
| --- | ---: | ---: |
| Baseline, constant 2048 | 4.01596332823529 | 3493.98 |
| Frozen fast curriculum | 3.959812460576787 | 4485.82 |

Fast has **1.3982% lower CE** in this same-code pair, while taking longer. This supports an improvement for seed 1 with the compiler boundary held fixed; it does not decompose the original seed-0 difference.

The seed-2 baseline also completed 610 updates and 9,994,240 predictions with exit 0: final CE **4.0188431284006905**, training **3494.89 seconds**, and a separately recorded **31.66-second** warmup excluded from training time. Its paired fast candidate is running; final CE and training time are **PENDING**, so no seed-2 paired gain is asserted.

The auxiliary seeds do not replace the seed-0 submission files in `../`, establish statistical significance, or constitute an H100 record. No timing speedup is claimed.

The default invocation still runs constant 2048 at this preparation stage; use the explicit command above for this result. Source change and logs are prepared locally; no upstream PR has been submitted yet.
