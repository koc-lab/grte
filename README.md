# Graph Receptive Transformer Encoder for Text Classification

[![GitHub License](https://img.shields.io/github/license/koc-lab/grte)](https://github.com/koc-lab/grte/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org)

This repository contains the code for the ["Graph Receptive Transformer Encoder for Text Classification"](https://doi.org/10.1109/TSIPN.2024.3380362) paper published in _IEEE Transactions on Signal and Information Processing over Networks_. Please cite the following paper if you use this code in your research:

```bibtex
@article{grte2024,
  author   = {Aras, Arda Can and Alikaşifoğlu, Tuna and Koç, Aykut},
  journal  = {IEEE Transactions on Signal and Information Processing over Networks},
  title    = {Graph Receptive Transformer Encoder for Text Classification},
  year     = {2024},
  volume   = {},
  number   = {},
  pages    = {1-13},
  doi      = {10.1109/TSIPN.2024.3380362}
}
```

## Installation

This codebase utilizes [`Poetry`](https://python-poetry.org) for package management. To install the dependencies, in the root directory run:

```sh
poetry install
```

or if you do not want to use `Poetry`, or use the exact pinned versions, you can install the dependencies provided in [`requirements.txt`](requirements.txt) using `pip` or `conda`, e,g.,

```sh
pip install -r requirements.txt
```
