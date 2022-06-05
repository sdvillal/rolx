#!/bin/zsh
source ~/.zshrc
conda-dev-on
source activate jpegs-benchmark-noturbo
python -u benchmark.py benchmark | tee jpegs-benchmark-noturbo.log