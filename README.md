<div align="center">

# 🍐 PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation

[![ACL 2026](https://img.shields.io/badge/ACL-2026-2f6fdd)](https://2026.aclweb.org/)
[![ACL Anthology](https://img.shields.io/badge/ACL%20Anthology-paper-b31b1b)](https://aclanthology.org/2026.acl-long.1953/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-ffcc4d)](https://huggingface.co/collections/Prosho/pear)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.12-blue)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

</div>

This repository provides the inference toolkit for 🍐**PEAR**: **P**airwise **E**valuation for **A**utomatic **R**elative Scoring in Machine Translation.

PEAR reframes reference-free machine translation evaluation as a graded pairwise comparison: given a source segment and two candidate translations, it predicts both the **direction** and **magnitude** of their quality difference. The toolkit also supports reference-anchored PEAR scoring and PEAR-based minimum Bayes risk (MBR) decoding.

* 📄 Paper: [ACL Anthology](https://aclanthology.org/2026.acl-long.1953/)
* 🤗 Models and resources: [PEAR Hugging Face collection](https://huggingface.co/collections/Prosho/pear)

## Installation

Install from a local clone:

```bash
git clone https://github.com/prosho-97/pear.git
cd pear
pip install .
```

For local development, install the development extra too:

```bash
pip install -e ".[dev]"
```

## Loading a model

```python
import pear

metric = pear.load_metric("pear")       # resolves to Prosho/pear on Hugging Face
metric_xl = pear.load_metric("pear-xl") # resolves to Prosho/pear-xl on Hugging Face
```

You can also pass a Hugging Face repository ID or a local PEAR model checkpoint path:

```python
metric = pear.load_metric("Prosho/pear")
metric = pear.load_metric("/path/to/checkpoints/model.ckpt")
```

## Pairwise QE scoring

```python
scores = pear.score_pairwise(
    metric,
    sources=["The cat is on the mat."],
    translations_a=["El gato está en la alfombra."],
    translations_b=["El gato está en el mapa."],
)
```

Positive scores mean `translations_a` is preferred over `translations_b`.

To score both candidate orders:

```python
scores = pear.score_pairwise(
    metric,
    sources=["The cat is on the mat."],
    translations_a=["El gato está en la alfombra."],
    translations_b=["El gato está en el mapa."],
    mode="both",
)
# {"forward": [...], "reverse": [...]}
```

## Reference-anchored scoring

Reference-anchored PEAR uses the human reference (or any other list of candidate translations) as the second candidate/anchor.

```python
scores = pear.score_reference_anchored(
    metric,
    sources=["The cat is on the mat."],
    translations=["El gato está en la alfombra."],
    references=["El gato está sobre la alfombra."],
)
```

As with pairwise QE scoring, set `mode="both"` to score both candidate orders for reference-anchored inputs.

## PEAR for MBR decoding

```python
from pear.mbr import pear_utility_matrix, select_mbr_hypothesis

hyps = ["This is a good translation.", "This is a very good translation.", "This is a bad translation."]
utility = pear_utility_matrix(metric, "Questa è una traduzione molto buona.", hyps, mode="half")
index, expected_utility = select_mbr_hypothesis(utility)
best = hyps[index]
# best -> "This is a very good translation."
```

Use `mode="full"` to score all off-diagonal ordered pairs (N<sup>2</sup> - N comparisons). Use `mode="half"` to score one triangular half and fill the opposite direction by PEAR antisymmetry.

## Batching, devices, and progress bars

Python and CLI inference are batched. Use `batch_size` / `--batch-size` to control the number of pairwise examples per PEAR forward batch.

By default, PEAR uses:

```python
gpus="auto"
```

This selects one available CUDA/MPS accelerator when present. Pass `gpus=0` or `--gpus 0` to force CPU. Progress bars are shown by default; in Python calls, control them with the `progress_bar` argument (`progress_bar=True` by default, or `progress_bar=False` to disable them).

## CLI

### Score a TSV file

Pairwise input TSV must contain `src`, `mt_0`, and `mt_1` columns:

```bash
pear score --model pear --input pairs.tsv --output scored.tsv --batch-size 16
pear score --hf-model Prosho/pear --input pairs.tsv --output scored.tsv --batch-size 16
pear score --checkpoint /path/to/model.ckpt --input pairs.tsv --output scored.tsv --batch-size 16
```

Reference-anchored input TSV must contain `src`, `mt`, and `ref` columns:

```bash
pear score --model pear --mode reference --input refs.tsv --output scored.tsv --batch-size 16
```

Add `--both` to score both candidate orders.

### Run MBR

Input JSONL rows must contain `src` and `hypotheses`:

```json
{"src": "source text", "hypotheses": ["hyp A", "hyp B"]}
```

Run half- or full-matrix PEAR MBR:

```bash
pear mbr --model pear --input nbest.jsonl --output selected.jsonl --utility half --batch-size 16
pear mbr --hf-model Prosho/pear-xl --input nbest.jsonl --output selected.jsonl --utility full --batch-size 16
pear mbr --checkpoint /path/to/model.ckpt --input nbest.jsonl --output selected.jsonl --batch-size 16
```

## Development

Install the development extra before running formatter checks:

```bash
pip install -e ".[dev]"
```

Then run:

```bash
python -m black --check pear
python -m compileall pear
```

## Citation

PEAR has been published at ACL 2026 (Main Conference). If you use it, please consider citing our paper as follows:

```bibtex
@inproceedings{proietti-etal-2026-pear,
    title = "{PEAR}: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation",
    author = "Proietti, Lorenzo  and
      Grundkiewicz, Roman  and
      Post, Matt",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1953/",
    doi = "10.18653/v1/2026.acl-long.1953",
    pages = "42189--42207",
    ISBN = "979-8-89176-390-6"
}
```

## License

This project is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See [`LICENSE`](LICENSE) for the full license text.
