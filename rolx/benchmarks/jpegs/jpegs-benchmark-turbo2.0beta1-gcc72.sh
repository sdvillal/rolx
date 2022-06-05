#!/bin/zsh
source ~/.zshrc
conda-dev-on
source activate jpegs-benchmark-turbo2.0beta1-gcc72
python -u benchmark.py benchmark | tee jpegs-benchmark-turbo2.0beta1-gcc72.log