# coding=utf-8
"""I/O workload simulations."""
from __future__ import print_function
from itertools import product
from functools import partial

import numpy as np

import whatami

import hashlib

from rolx.sponsors.video.atloopbio import OpenCVReader
from rolx.sponsors.video.checks import Barcoder
from rolx.utils import resize, sanitize_start_stop, ensure_cold, ensure_warm


class Workload(object):

    def __init__(self, name, config, callme):
        super(Workload, self).__init__()
        self.name = name
        self.config = config
        self.callme = callme

    @property
    def name_hash(self):
        return hashlib.sha256(self.name).hexdigest()

    def __call__(self, *args, **kwargs):
        return self.callme(*args, **kwargs)


# --- Write workload

# Width, height, color-depth
BENCHMARK_IMAGE_RESOLUTIONS = (
    (960, 540, 3),
    (480, 270, 3),
    (1920, 1080, 3),
)


def write_workload(original_path, writer, dest_path, start=0, stop=None, step=1, batch_size=1,
                   original_reader=OpenCVReader, watermarker=Barcoder(), new_shape=(960, 540, 3),
                   seed=0):
    # instantiate the reader for the original video
    original_reader = original_reader(original_path)
    # start stop and frame numbers
    if stop is None:
        stop = len(original_reader)
    if start is None:
        rng = np.random.RandomState(seed)
        start = rng.randint(stop // 2)
    start, stop = sanitize_start_stop(original_reader, start, stop)
    frame_nums = np.arange(start, stop, step)
    # read in batches?
    batches = np.array_split(frame_nums, np.arange(batch_size, len(frame_nums), batch_size))
    # instantiate writer, taking into account variants
    try:
        # pffff, take into account writers that need the shape beforehand (i.e. StoreWriter)
        writer = writer(path=dest_path, image_shape=new_shape)
    except TypeError:
        writer = writer(path=dest_path)
    # do writing (including resizing and watermarking)
    written_frame_num = 0
    with original_reader:
        with writer:
            for batch in batches:
                images = original_reader.get(batch)
                for image in images:
                    image = resize(image, new_width=new_shape[0], new_height=new_shape[1])
                    image = watermarker.encode(image, written_frame_num)
                    writer.add(image)
                    written_frame_num += 1


BENCHMARK_WRITE_WORKLOADS = []

for resolution in BENCHMARK_IMAGE_RESOLUTIONS:
    config = dict(new_shape=resolution)
    name = whatami.What('write', conf=config).id()
    BENCHMARK_WRITE_WORKLOADS.append(Workload(
        name=name,
        config=config,
        callme=partial(write_workload, **config)
    ))

# --- Read workload

# Sequential or random access workload?
BENCHMARK_READ_SEQUENTIALS = True, False

# Read step
# How well do readers allow to skip consecutive frames
BENCHMARK_READ_STEPS = 1, 2, 4, 8
BENCHMARK_READ_STEPS = 1,

# Read batch size
BENCHMARK_READ_BATCH_SIZES = 1, 32,  # 16, 64, 128

# Seek policies
# (note that commented out tend to be slow for random access; "seek vs find")
BENCHMARK_READ_SEEKS = 'always',  # 'never', 5, 50

# Num threads (checks must be done to ensure this is supported)
BENCHMARK_READ_NUM_THREADSS = tuple(sorted({1, 2, 4}))  # 0, cpu_count()

# Essentially, how often do we need to hit disk (cold vs warm file system caches)
# Ideally we just need to compute two values (never, always) and then interpolate
# Obviouly, in real life this is not so simple, evict rates depend on system load,
# it is LRU and so a store can just be partially warm, partially cold.
# oh well... useful models (?)...
BENCHMARK_READ_EVICT_RATES = 0, 1, None
# TODO: we should add None + cold_start, makes sense as evict_rate=1 is very artificial
#       specially for video, where we force cold caches where we would otherwise be
#       having mostly warm (as page fetches would bring a lot of data from the
#       next frame... oh well...)

# How much data
# N.B. warm will put as much data as possible (but obviously some might evict due to RAM limits)
BENCHMARK_READ_FS_STATE = None,  # 'cold', 'warm'

# Sensible combinations for FS cache states
# These tuples are: initial_fs_state, evict_rate
BENCHMARK_CACHE_POLICIES = (
    # Always (hopefully) warm
    (None, 0),
    # Always cold
    (None, 1),
    # Start cold, let it evolve (specially relevant for full sequential)
    ('cold', None),
)


def read_workload(reader, path, sequential=True,
                  start=0, stop=None, step=1,
                  max_num_frames_to_read=256,
                  min_num_frames_to_read=10,
                  stop_fps_threshold=20,
                  batch_size=1,
                  seek='always',
                  num_threads=1,
                  evict_rate=1, initial_fs_state=None, warm_always=False,
                  seed=0):

    # perhaps here check that reader uses these / make a difference and if not, fail inmediately
    reader = reader(path=path, num_threads=num_threads, seek=seek)
    # initial file system cache state (evict_rate == 0 overrides it, document)
    is_warm = False
    if evict_rate == 0 or initial_fs_state == 'warm':
        ensure_warm(reader.path)
        is_warm = True
    elif initial_fs_state == 'cold':
        ensure_cold(reader.path)
    # generate batches to read
    start, stop = sanitize_start_stop(reader, start, stop)
    rng = np.random.RandomState(seed)
    frame_nums = np.arange(start, stop, step)
    batches = np.array_split(frame_nums, np.arange(batch_size, len(frame_nums), batch_size))
    if not sequential:
        batches = rng.permutation(batches)
    # how many batches to read?
    # warning: this should be enough to have a decent sample size
    num_batches = max_num_frames_to_read // batch_size
    batches = batches[:num_batches]
    # use the watch to finish early (here we could use proper benchmark calib)
    watch = reader.callbacks[-1].watch  # Need to give a name here, although guarantee to be the last one
    # loop-it
    with reader:
        for batch in batches:
            # warm or cold read?
            if evict_rate is not None:
                if 0 < rng.uniform() <= evict_rate:
                    ensure_cold(reader.path)
                    is_warm = False
                elif not is_warm or warm_always:
                    ensure_warm(reader.path)
                    is_warm = True

            # do the read
            reader.get(batch)

            # Early stop?
            # (of course this gets tricked by large batch sizes, so slow workloads can still take too long)
            # whatever we can do about it, it would be more or less nasty:
            #  - do this same check in Reader.get
            #  - allow callbacks to kick out reading at any time
            # maybe at another time
            if watch.num_units >= min_num_frames_to_read:
                if watch.ups_wall < stop_fps_threshold:
                    return


BENCHMARK_READ_WORKLOADS = []
for (sequential,
     read_step, batch_size, seek,
     num_threads,
     (initial_fs_state, evict_rate)) in product(
        BENCHMARK_READ_SEQUENTIALS,
        BENCHMARK_READ_STEPS,
        BENCHMARK_READ_BATCH_SIZES,
        BENCHMARK_READ_SEEKS,
        BENCHMARK_READ_NUM_THREADSS,
        BENCHMARK_CACHE_POLICIES):
    config = dict(
        sequential=sequential,
        step=read_step,
        batch_size=batch_size,
        num_threads=num_threads,
        evict_rate=evict_rate,
        initial_fs_state=initial_fs_state,
        # seek here is a bit contrived,
        # and will make for repeated work on exploded Stores (which obviously do not use seek)
        seek=seek,
        # TODO:
        # the same for num_threads, need to extract these concerns
        # to another benchmark dimension: reader
        #  - then we can check for readers that do not support certain things
        #    (like stores and seeking, of MJPEG + thread_count (apparently))
        # anyway, it is late...
    )
    name = whatami.What('read', conf=config).id()
    BENCHMARK_READ_WORKLOADS.append(Workload(
        name=name,
        config=config,
        callme=partial(read_workload, **config)
    ))


# --- Put it all together

BENCHMARK_WORKLOADS = [(wwl, BENCHMARK_READ_WORKLOADS)
                       for wwl in BENCHMARK_WRITE_WORKLOADS]


if __name__ == '__main__':
    print('Number of read workloads:', sum(len(rwl)for _, rwl in BENCHMARK_WORKLOADS))
