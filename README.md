# Depth Laws for the Precision Floor of Trained Neural Networks

Code and result files for the paper *Depth Laws for the Precision Floor of Trained Neural Networks*
(Ahmad S. Tarawneh). The repository reproduces every experiment, figure, table and number in the paper.

## Contents

| Folder | What it contains |
|---|---|
| `data/` | `download.py`: Fashion-MNIST, MNIST, CIFAR-10, WikiText-103 and the pretrained GPT-2 / Pythia checkpoints |
| `mlp/` | MLP experiments (NumPy, CPU): noise floors, amplification, PTQ bit floors, layer map, margins, error decomposition |
| `cnn/`, `vit/` | CNN and Vision Transformer models and training on CIFAR-10 (PyTorch, GPU) |
| `vision/` | shared quantizers, measurement of saved CNN/ViT checkpoints, noise-aware training |
| `llm/` | measurements on pretrained language models (inference only) |
| `analysis/` | builds all figures, tables and `numbers.json` from the result files |
| `results/` | the result files used in the paper |
| `figures/` | output of `analysis/make_figures.py` |

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

The MLP experiments run on a CPU. The CNN, ViT and language-model experiments need a CUDA GPU.

The results in `results/` were produced on one server: 2 × Intel Xeon Gold 6426Y (32 cores), 128 GB RAM,
NVIDIA RTX A4000 (16 GB), Windows Server 2022, Python 3.12, NumPy 2.4, PyTorch 2.5 (CUDA 12.1). Approximate
run times on this machine: full MLP grid 1.7 hours (56 parallel processes); measuring all CNN/ViT checkpoints
and the language models 3.5 GPU-hours; ViT noise-aware training 10 GPU-hours.

## Data

```bash
python -m data.download fashion      # also: mnist, cifar10, wikitext, llms, or all (about 17 GB)
```

Paths are set in `paths.py` and can be changed with the environment variables `DL_DATA` (datasets and
pretrained models, default `data/cache`), `DL_RESULTS` (result files, default `results`) and `DL_MODELS`
(CNN/ViT checkpoints, default `results/models`). All commands are run from the repository root.

## Reproducing the experiments

### MLPs (Fashion-MNIST and MNIST, CPU)

Each script writes one line per run to a text file in `results/mlp/`. The complete grid of the paper
(about 1,000 runs, a few hours on 16 cores) is run in parallel by

```bash
python -m mlp.run_all --jobs 16
```

Single experiments can be run directly, for example

```bash
python -m mlp.noise_floor --arch plain --mode infer --depths 4 8 16 --seeds 0 10
python -m mlp.amplification --arch res2 --depths 8 16 32 64 --seeds 0 10
python -m mlp.quantize --arch res2c --depths 8 16 32 64 --seeds 0 10
python -m mlp.layer_map --depths 4 8 16 24 --seeds 0 5
python -m mlp.margins --mode infer --depths 8 16 --seeds 0 3
python -m mlp.counteraction --arch plain --depths 8 16 32 --seeds 0 3
```

### CNNs and Vision Transformers (CIFAR-10, GPU)

```bash
python -m cnn.train --arch res --depths 4 8 16 32 64 --seeds 0 3          # checkpoints
python -m vit.train --arch pre --depths 2 4 8 16 32 --seeds 0 3
python -m vision.measure --family cnn                                       # results/cnn/measure.txt
python -m vision.measure --family vit                                       # results/vit/measure.txt
python -m vision.qat_noise --family cnn --arch res1 --depths 8 16 32 64     # noise-aware training
python -m cnn.counteraction --bits 6
```

`python -m vision.run_all --list` prints the full grid used in the paper.

### Pretrained language models (GPU, inference only)

```bash
python -m llm.measure gpt2
python -m llm.measure gpt2 --per-channel
python -m llm.measure gpt2-xl --g-batch 1 --g-cpu      # lower memory for the largest model
```

### Figures, tables and numbers

```bash
python -m analysis.make_figures
```

writes every figure (`figures/fig_*.pdf`), every table (`figures/tab_*.tex`) and all numbers quoted in the
paper (`figures/numbers.json`).

## Result files

`results/` contains the raw outputs used in the paper, one line per run, in the format written by the scripts
above:

| File | Produced by |
|---|---|
| `mlp/noise_floor.txt`, `mlp/amplification.txt`, `mlp/quantize.txt`, `mlp/layer_map.txt`, `mlp/margins_*.txt`, `mlp/counteraction.txt` | `mlp/*.py` |
| `cnn/measure.txt`, `vit/measure.txt` | `vision/measure.py` on the trained checkpoints |
| `cnn/qat_noise.txt`, `vit/qat_noise.txt` | an earlier version of `vision/qat_noise.py` with the same models and training (noise drawn from the global generator); rerunning the current script gives statistically equivalent numbers |
| `cnn/counteraction.txt` | `cnn/counteraction.py` |
| `llm/measure.txt` | `llm/measure.py` |

## Reproducibility notes

* The MLP experiments are deterministic for a given NumPy installation. Different BLAS libraries round
  differently; for the deepest plain MLPs (32 layers) this can change individual runs, but not the averages
  over seeds.
* All noise draws in the measurement scripts use seeded generators, so re-measuring a checkpoint gives
  identical numbers.
* GPU training is not bit-wise deterministic. Retraining a CNN or ViT gives accuracies within about one
  percentage point of ours; the depth laws are unaffected.

## License

MIT (see `LICENSE`).
