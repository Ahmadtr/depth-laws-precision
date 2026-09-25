"""Download all datasets and pretrained models used in the paper (from the Hugging Face Hub).

    python -m data.download all            # everything (about 17 GB, mostly the language models)
    python -m data.download cifar10        # or: fashion, mnist, wikitext, llms

Outputs (inside paths.DATA):
    cifar10.pt                         CIFAR-10 as uint8 tensors {"train": (x, y), "test": (x, y)}
    fashion/, mnist/                   IDX files {train,test}-{images,labels}.gz
    hf/wikitext/                       WikiText-103 validation split (parquet)
    hf/models/<name>/                  GPT-2 and Pythia checkpoints (safetensors)
"""
import gzip
import io
import json
import os
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

HUB = "https://huggingface.co"
LLMS = ["openai-community/gpt2", "openai-community/gpt2-medium", "openai-community/gpt2-large",
        "openai-community/gpt2-xl", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b"]


def _parquet_files(repo, prefix=""):
    url = f"{HUB}/api/datasets/{repo}/tree/main?recursive=true"
    tree = json.load(urllib.request.urlopen(url, timeout=60))
    return [f["path"] for f in tree if f["path"].endswith(".parquet") and f["path"].startswith(prefix)]


def _fetch(repo, path, dest):
    import pyarrow.parquet as pq
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(f"{HUB}/datasets/{repo}/resolve/main/{path}", dest + ".part")
        os.replace(dest + ".part", dest)
    return pq.read_table(dest)


def _images(table, mode):
    from PIL import Image
    image_col = [c for c in table.column_names if "im" in c][0]
    label_col = [c for c in table.column_names if "label" in c][0]
    x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert(mode))
                  for d in table.column(image_col).to_pylist()])
    return x.astype(np.uint8), np.asarray(table.column(label_col).to_pylist(), dtype=np.uint8)


def cifar10():
    import torch
    out = {}
    for p in _parquet_files("uoft-cs/cifar10", "plain_text"):
        dest = os.path.join(paths.DATA, "raw", "cifar10", os.path.basename(p))
        x, y = _images(_fetch("uoft-cs/cifar10", p, dest), "RGB")
        out["train" if "train" in p else "test"] = (torch.tensor(x).permute(0, 3, 1, 2).contiguous(),
                                                    torch.tensor(y, dtype=torch.long))
    torch.save(out, os.path.join(paths.DATA, "cifar10.pt"))
    print("CIFAR-10:", {k: tuple(v[0].shape) for k, v in out.items()})


def _idx(name, repo):
    d = os.path.join(paths.DATA, name)
    os.makedirs(d, exist_ok=True)
    for p in _parquet_files(repo):
        split = "train" if "train" in p else "test"
        x, y = _images(_fetch(repo, p, os.path.join(paths.DATA, "raw", name, os.path.basename(p))), "L")
        with gzip.open(os.path.join(d, f"{split}-images.gz"), "wb") as f:
            f.write(struct.pack(">IIII", 2051, len(x), 28, 28) + x.tobytes())
        with gzip.open(os.path.join(d, f"{split}-labels.gz"), "wb") as f:
            f.write(struct.pack(">II", 2049, len(y)) + y.tobytes())
        print(f"{name} {split}: {x.shape}")


def fashion():
    _idx("fashion", "zalando-datasets/fashion_mnist")


def mnist():
    _idx("mnist", "ylecun/mnist")


def wikitext():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
                        repo_type="dataset", local_dir=os.path.join(paths.HF, "wikitext"))
    print("WikiText-103:", p)


def llms():
    from huggingface_hub import list_repo_files, snapshot_download
    for repo in LLMS:
        files = list_repo_files(repo)
        weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["pytorch_model*.bin"]
        snapshot_download(repo, local_dir=os.path.join(paths.HF, "models", repo.split("/")[1]),
                          allow_patterns=["*.json", "tokenizer*", "vocab*", "merges*"] + weights)
        print("downloaded", repo)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("cifar10", cifar10), ("fashion", fashion), ("mnist", mnist), ("wikitext", wikitext),
                     ("llms", llms)):
        if what in (name, "all"):
            fn()
