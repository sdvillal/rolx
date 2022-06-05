#!/bin/zsh
source ~/.zshrc
conda-dev-on
source activate jpegs-benchmark-turbo1.5.3-gcc48
python -u benchmark.py benchmark | tee jpegs-benchmark-turbo1.5.3-gcc48.log