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

Both auxiliary pairs are complete. All four runs reached 610 updates and 9,994,240 predictions with exit 0; the confirmation queue also exited 0:

| Confirmation | Baseline final CE | Fast final CE | Relative CE reduction | Baseline / fast training seconds |
| --- | ---: | ---: | ---: | ---: |
| Seed 1 | 4.01596332823529 | 3.959812460576787 | 1.3982% | 3493.98 / 4485.82 |
| Seed 2 | 4.0188431284006905 | 4.0156356920214265 | 0.0798% | 3494.89 / 4487.06 |

Both same-code pairs favor fast, but the effect varies substantially and the second gain is very small. These results do not establish a stable improvement magnitude or statistical significance, and do not decompose the original seed-0 compiler difference. Fast takes longer in both pairs.

A zero-update, forward-only diagnostic at one frozen checkpoint/site/batch, with four redraws and empty credit memory, found lower relative RMS dispersion (0.659 vs 0.971) and higher directional agreement (0.596 vs 0.353) at context 64, but **higher absolute variance** (1.38226e-4 vs 1.04624e-4). Different conditioning and possible estimator bias prevent a true-gradient or causal explanation of curriculum quality.

The auxiliary seeds do not replace the seed-0 submission files in `../`, establish statistical significance, or constitute an H100 record. No timing speedup is claimed.

The default invocation now selects seed 0, the fast schedule and `eval-every=61`, equivalent to the recorded explicit command above; the remaining recipe is unchanged. For shorter smoke runs ending before update 161, pass `--context-schedule 1:2048`.

[Seed-0 full-validation curves](curves.svg) · [Same-code paired-confirmation curves](confirmation/plots/confirmation.svg)
