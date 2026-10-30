# Noise: Speedrun for pretraining without backpropagation
![Noise baselines](val_loss_animation_30m.gif)

Noise is a pretraining speedrun for learning algorithms that never run a backward pass. We think replacing
backprop with a much better credit assignment algorithm is inevitable. Deep learning's architectures,
optimizers and hardware have co-evolved with backprop, but as compute grows, the
[bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) suggests that more general,
brute-force learning algorithms based on search will win. [Dust](https://qlabs.sh/research/dust) is a big step
forward: it rivals backprop under SGD and is 1,000 to 10,000x more efficient than EGGROLL, the state-of-the-art
weight-space ES, but still trails backprop under Adam. These methods have also had a small fraction of the
research that went into backprop. We made Noise to close the gap: better estimators, and optimizers co-designed
for them instead of for backprop gradients.

The model, the data and the score are fixed: the 37.7M-parameter transformer of Dust, a 30M-token FineWeb
training set, and the cross-entropy on Dust's validation set. What changes is the learning algorithm, which may
only run the model forward; track 3 also frees the architecture. Until October 22, 2026, the training set was
Dust's 10M tokens; those tracks and records are archived in [`archive/10M/`](archive/10M/README.md).

## Tracks

The 10M-token tracks are archived in [`archive/10M/`](archive/10M/README.md) and no longer accept submissions.

1. **Validation loss.** The lowest validation loss after one epoch on one 8xH100 node, with no time limit.
2. **Speedrun.** The shortest training time on one 8xH100 node to reach validation loss **3.5**.
3. **Open architecture.** Track 1 with the architecture free: the model is
   [`train/arch/model.py`](train/arch/model.py), which entries may change.

## Rules

- **No backpropagation.** No autograd (reverse or forward mode) and no hand-written backward pass. Learning
  signal comes from evaluating the model forward: every error signal is measured against a loss by running the
  model forward, and may then take at most one step through a nonlinear part of the model. Steps through
  linear maps don't count, and a signal that has taken a step can't take another, so credit can't be chained
  back through the network block by block. Entries are judged by the spirit of this rule: a method should
  still apply to networks with non-differentiable parts.
- **Fixed model** (tracks 1 and 2). Final weights must load into [`utils/model.py`](utils/model.py) with a
  strict state-dict match. How training computes the same function, and how it initializes it, is free. On
  track 3 the model is [`train/arch/model.py`](train/arch/model.py) instead, a copy of `utils/model.py` that
  entries change in place; it must keep the interface `utils/eval.py` uses (`GPT(GPTConfig())`, `init_weights()`
  and the summed loss from `forward`).
- **Fixed data.** Train only on `data/train.pt` (30M tokens: 14,648 sequences of 2,048 predictions), for at most one
  epoch: each training sequence feeds at most one update. Batch size and order are free.
- **Score.** [`utils/eval.py`](utils/eval.py) on the final weights (on track 3 with `--model
  train/arch/model.py`): mean next-token cross-entropy over the 1,114,112 predictions of `data/val.pt`. No
  checkpoint selection on validation loss.
- **Timing** (speedrun track; reported on the other). From the start of the first update to the end of the
  last, excluding any validation. An untimed warmup on random tokens may compile kernels, as long as training
  then starts from a fresh state (the baselines rebuild everything from the seed). Runs use seed 0.

## Setup

Requires Linux, an NVIDIA GPU with bfloat16 support and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run utils/prepare_data.py
```

`utils/prepare_data.py` streams FineWeb `sample-10BT`, tokenizes it with the fixed BPE-4096 tokenizer in
[`tokenizer/`](tokenizer/tokenizer.json), and writes `data/train.pt` and `data/val.pt`. `val.pt` is the
[Dust repo](https://github.com/qlabs-eng/dust)'s; `train.pt` is Dust's ~20.5M-token training pool followed by the
documents after its validation and test documents, so its first 4,880 sequences are Dust's `train_10M.pt`. Both
are checked against pinned hashes.

## Running the records

```bash
uv run torchrun --standalone --nproc_per_node=8 train/val/run.py         # track 1
uv run torchrun --standalone --nproc_per_node=8 train/speedrun/run.py    # track 2
uv run torchrun --standalone --nproc_per_node=8 train/arch/run.py        # track 3
uv run torchrun --standalone --nproc_per_node=8 train/val/dust.py        # Dust w/ Adam
```

Each run writes `config.json`, `metrics.jsonl`, `final.pt` and `result.json` (validation loss and training time)
to `logs/`. `uv run utils/eval.py logs/val/final.pt` rescores a checkpoint.

## Records

### Track 1: validation loss (30M)

| # | Val loss | Description | Time | Script | Contributors |
| - | - | - | - | - | - |
| 1 | 3.767 | Dust w/ Adam, 24k population, lr cooldown | 5 h 17 min | [Script](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py) | - |
| 2 | 3.409 | 10M record #14: Boost → Cut with own queries, exact values, future-token probes and an end merge | 1 h 43 min | [Script](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/run.py) | [@JaydenTeoh](https://github.com/JaydenTeoh), [@spicylemonade](https://github.com/spicylemonade), [@hoo0681](https://github.com/hoo0681), [@Lazarus-931](https://github.com/Lazarus-931), [@CharlesDDDD](https://github.com/CharlesDDDD), [@NotDesigned](https://github.com/NotDesigned) |
| 3 | 3.338 | A forward readout of cross-token credit along each token's path; a fixed Boost, lower ANVIL II weight decay | 56 min | [Script](https://github.com/qlabs-eng/noise/blob/6f0c450ed52d52dd2fefd2a8ba485d92f2b37e92/train/val/run.py) | [@CharlesDDDD](https://github.com/CharlesDDDD) |
| 4 | 3.300 | #3 with two-hop credit through next-token twins and a transit decode; head transplants, self-distillation, a planted previous-token head, a linear decay | 43 min | [Script](https://github.com/qlabs-eng/noise/blob/baece979a903ed960736476a7fab8ca1df43a146/train/val/run.py) | [@CharlesDDDD](https://github.com/CharlesDDDD) |

### Track 2: speedrun to 3.5 (30M)

| # | Val loss | Description | Time | Script | Contributors |
| - | - | - | - | - | - |
| 1 | 3.409 | 10M record #14: Boost → Cut with own queries, exact values, future-token probes and an end merge | 1 h 43 min | [Script](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/speedrun/run.py) | [@JaydenTeoh](https://github.com/JaydenTeoh), [@spicylemonade](https://github.com/spicylemonade), [@hoo0681](https://github.com/hoo0681), [@Lazarus-931](https://github.com/Lazarus-931), [@CharlesDDDD](https://github.com/CharlesDDDD), [@NotDesigned](https://github.com/NotDesigned) |
| 2 | 3.499 | The identity path first, then Cut-lite at 8 rows per update with faster chain reruns | 8 min 20 s | [Script](https://github.com/qlabs-eng/noise/blob/4586b02e7f77927683c1f5d28a981b3a0342b9ea/train/speedrun/run.py) | [@RobertTLange](https://github.com/RobertTLange) |
| 3 | 3.494 | A shorter identity phase, then Cut with paired MLP roots and a difficulty-balanced row order | 2 min 3 s | [Script](https://github.com/qlabs-eng/noise/blob/8be79219cf53d409060261477cf54ce8f1a03a28/train/speedrun/run.py) | [@conglu1997](https://github.com/conglu1997) |
| 4 | 3.498 | #3's identity phase, then Cut with two single roots, fused Gluon chain kernels and overlapped bf16 collectives | 22.9 s | [Script](https://github.com/qlabs-eng/noise/blob/4f046549915122842092b9c92f7513833455ab3b/train/speedrun/run.py) | [@RobertTLange](https://github.com/RobertTLange) |

### Track 3: open architecture (30M)

New track where any architecture is allowed!

| # | Val loss | Description | Time | Script | Contributors |
| - | - | - | - | - | - |
| 1 | 3.767 | Dust w/ Adam, 24k population, lr cooldown | 5 h 17 min | [Script](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/val/dust.py) | - |
| 2 | 3.409 | 10M record #14: Boost → Cut with own queries, exact values, future-token probes and an end merge | 1 h 43 min | [Script](https://github.com/qlabs-eng/noise/blob/b6405a155a38371f994a8f3915ea5d8a87235315/train/arch/run.py) | [@JaydenTeoh](https://github.com/JaydenTeoh), [@spicylemonade](https://github.com/spicylemonade), [@hoo0681](https://github.com/hoo0681), [@Lazarus-931](https://github.com/Lazarus-931), [@CharlesDDDD](https://github.com/CharlesDDDD), [@NotDesigned](https://github.com/NotDesigned) |
| 3 | 3.378 | A quarter-turn previous-token tap in the embedding; exact future-token probes | 3 h 50 min | [Script](https://github.com/qlabs-eng/noise/blob/d45f830431ed9bebca3f36a464184ec8a3016513/train/arch/run.py) | [@dylantirandaz](https://github.com/dylantirandaz) |
| 4 | 3.358 | 6 blocks with hashed n-gram tables; future and input-key probes by moments | 27 min | [Script](https://github.com/qlabs-eng/noise/blob/e0e1ff2c15fd64e8385c6d761f49ae8e9f35fc39/train/arch/run.py) | [@Euraba](https://github.com/Euraba) |
| 5 | 3.211 | 12 × 1024 GPT with hashed n-gram value embeddings; Cut-lite with input-key probes | 2 h 57 min | [Script](https://github.com/qlabs-eng/noise/blob/561c1f1d54284dfcb344960eba4d26c6d631035b/train/arch/run.py) | [@RobertTLange](https://github.com/RobertTLange) |

## What backprop gets

The same model, data and epoch trained with backprop, scored the same way (seed 0, a few learning rates each):

| Optimizer | Val loss |
| - | - |
| Adam w/ lr cooldown | 3.419 |
| ANVIL II on the block matrices and Adam elsewhere | 3.340 |

The goal is to beat backprop by a large margin.

## Submitting

Open a PR that changes its track's training script in place, [`train/val/run.py`](train/val/run.py),
[`train/speedrun/run.py`](train/speedrun/run.py) or [`train/arch/run.py`](train/arch/run.py) (with
[`train/arch/model.py`](train/arch/model.py) on track 3), with the run's log (`config.json`, `metrics.jsonl` and `result.json` from `logs/`) in the `log/` folder next to it. A record must beat the current one
in its track under the rules above; the training script is the record.

## Citation

```bibtex
@misc{noise_2026,
  title  = {Noise: Speedrun for pretraining without backpropagation},
  author = {G{\"u}lbahar, Serdar and Mandal, Bishwas and Vegesna, Akshay and Dahal, Samip},
  year   = {2026},
  url    = {https://github.com/qlabs-eng/noise}
}
```

**Core contributors:** Serdar Gülbahar and Bishwas Mandal.

## License

Released under the [MIT License](LICENSE).
