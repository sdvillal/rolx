# rolx: counterfeit watches for timing and benchmarking

rolx is a timing and monitoring package for python. It includes
luxury watches that make your code look good while tic-tocing
around statements you want to know how long they take and how
many resources they use.

rolx comes with batteries:

- *Iteratively updatable watches*

```python
import time
import numpy as np
from rolx.watches import Rusager, Unit

watch = Rusager(Unit('zzz', 'zzzs', 'zzz'))

with watch:
    time.sleep(1)

with watch.tictoc(3):
    time.sleep(1)
    time.sleep(3)
    time.sleep(2)

print(watch.wall_time, watch.user_time, watch.system_time)
```

- *Watch grouping*

- *Tools to help fair benchmarking*

- ...

rolx sponsors certain modern day activities

- *Images and video I/O benchmarking*

- ...

## Install

Well, we need to define deps & friends, will do in a environment.yaml.
Bit for the time being, on top of our conda environment:

```
# Let's do in a disposable environment, just in case.
conda create --name loopy-benchmark --clone loopy-gpu

# Install/update
conda env update -n rolx -f loopy-supplement.yaml

# Only if you want to run the benchmarks, use this dev branch
pip install 'git+ssh://git@github.com/loopbio/python-loopb.git@thread_count#egg=loopb-bview' --upgrade --no-deps

# Install this...
pip install -e .
```


## Inspirations

[Our motto](http://www.gotfuturama.com/Multimedia/EpisodeSounds/3ACV20/04.mp3):

> Hey universe, check out the dude with the rolx!

is inspired by [Bender](https://www.youtube.com/watch?v=y1nuZrqULkM) Bending [Rodríguez](https://www.youtube.com/watch?v=_2wxNtYTypU).

## Authors and contact

Self-plagiarized and updated by @sdvillal @[loopbio](http://www.loopbio.com).
