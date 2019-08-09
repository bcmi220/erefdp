# EREFDP
This repository implements the dependency parser described in the paper [Effective Representation for Easy-First Dependency Parsing](https://arxiv.org/abs/1811.03511)

## Prerequisite
[Dynet Library](http://dynet.readthedocs.io/en/latest/)

## Usage (by examples)
### Train
We use embedding pre-trained by [GloVe](https://nlp.stanford.edu/projects/glove/) (Wikipedia 2014 + Gigaword 5, 6B tokens, 100d)

```
  python run.py --e train -c config.yaml [--dynet-gpu]
```
### Test
```
  python run.py --e test -c config.yaml [--dynet-gpu]
```
