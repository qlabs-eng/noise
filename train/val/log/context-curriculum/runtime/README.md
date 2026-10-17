# Exact B300 runtime used for these results

This is the existing Linux x86_64 / Python 3.12 lock deployed for all reported B300 runs. It is supplied as experiment evidence; the repository-root H100 project is unchanged. No new environment was created for this submission.

From the Noise repository root, on B300:

```bash
uv sync --frozen --project train/val/log/context-curriculum/runtime
OMP_NUM_THREADS=8 uv run --no-sync --project train/val/log/context-curriculum/runtime \
  torchrun --standalone --nproc_per_node=8 train/val/run.py \
  --data-dir data --output logs/val-context-fast --seed 0 \
  --context-schedule 1:64,33:128,65:256,97:512,129:1024,161:2048 \
  --eval-every 61
```

The fixed data must first be prepared with the unchanged `utils/prepare_data.py`. The experiment already used its official artifacts and passed the pinned token hashes. The command above is for a fresh reproduction, not a request to repeat completed data preparation or training.

The direct versions are Torch 2.10.0, tokenizers 0.23.1 and datasets 4.8.4. Torch uses cu130 instead of upstream cu128. Other resolved differences from the pinned upstream lock are: aiohttp 3.14.4→3.14.5, filelock 4.0.12→4.2.0, multidict 6.9.1→7.1.0, NumPy 2.5.3→2.5.4, PyArrow 25.0.1→26.0.0, cuda-bindings 12.9.4→13.0.3 and the CUDA libraries selected by the Torch wheel. Triton remains 3.6.0. The lock records complete versions. These are runtime differences, not a claim of an otherwise identical environment.

Eight previously slow wheel downloads use public Aliyun mirror URLs, with their original wheel filenames and locked hashes retained; this changes transport only. Torch's URL remains its official cu130 wheel. No credentials or training data are included.
