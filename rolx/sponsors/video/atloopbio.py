# coding=utf-8
"""Image providers on top of loopbio, with strong support to use video files."""
# Trying to collect here everything that depends on internal loopbio libraries
# See also notes on checks.py and codecs.py
from __future__ import print_function, division
from future.utils import string_types

import os.path as op
import shutil
from functools import partial
from itertools import chain

import numpy as np
from joblib import Parallel, delayed
from ruamel import yaml

from bview.capture.opencv import OpenCVCapture
from loopb.imgstore import new_for_filename, STORE_MD_FILENAME, new_for_format
from loopb.record import RecorderFFMPEG
from rolx.utils import ensure_writable_dir
from rolx.sponsors.video.videocodecs import VideoEncoderConfig, VEC
from rolx.sponsors.video.io import Writer, Reader, WatchCallback
from rolx.watches import PSUtilRusager


# --- Writers

class FFMPEGWriter(Writer):
    """Thin wrapper over RecorderFFMPEG."""

    def __init__(self, path, callbacks=(),
                 fn='video',
                 pix_fmt_i='bgr24',
                 codec=VideoEncoderConfig(),
                 fps=25,
                 image_size=None,
                 verbose=False, capture_output=True,
                 num_threads=None,
                 close_timeout=2.0,
                 ffmpeg_path='ffmpeg'):
        super(FFMPEGWriter, self).__init__(path, codec=codec, callbacks=callbacks)
        self.fn = fn
        self.pix_fmt_i = pix_fmt_i
        self.fps = fps
        self.image_size = image_size
        self.verbose = verbose
        self.capture_output = capture_output
        self.num_threads = num_threads
        self.close_timeout = close_timeout
        self.ffmpeg_path = ffmpeg_path
        self._rec = None

    def is_open(self):
        return self._rec is not None

    def open(self):
        self.close()
        if self.image_size is not None:
            self._rec = RecorderFFMPEG(
                fn=op.join(ensure_writable_dir(self.path), self.fn),
                codec=self.codec.to_recorder_dict(self.pix_fmt_i),
                fps=self.fps,
                image_size=self.image_size,
                add_extension=True,
                capture_output=self.capture_output,
                verbose=self.verbose,
                num_threads=self.num_threads,
                close_timeout=self.close_timeout,
                ffmpeg_path=self.ffmpeg_path,
            )
        return self

    def _write_meta(self):
        with open(op.join(self.path, 'info.yaml'), 'wt') as writer:
            meta_dict = {
                'type': 'video',
                'video': {
                    'file': self.codec.path(self.fn)
                }
            }
            yaml.safe_dump(meta_dict, writer)

    def _close_hook(self):
        if self.is_open():
            self._rec.close()
            self._rec = None

    def add(self, data):
        # so we do not miss first insertion in callbacks...
        if self.image_size is None:
            self.image_size = data.shape[1], data.shape[0]
            self.open()
        elif not self.is_open():
            self.open()
        super(FFMPEGWriter, self).add(data)

    def _add_hook(self, data):
        # N.B. at the moment RecorderFFMPEG always overwrites the file
        # We should make it able to append instead
        self._rec.process(data)

    @property
    def process(self):
        if self.is_open():
            # noinspection PyProtectedMember
            return self._rec._proc  # PFFFF
        return None


class StoreWriter(Writer):

    NPY = VEC(name='exploded-npy', container='npy', codec='npy', params=(),
              is_lossless=True, is_intra_only=False)
    JPG = VEC(name='exploded-jpg', container='jpg', codec='jpg', params=(),
              is_lossless=True, is_intra_only=True)
    PNG = VEC(name='exploded-png', container='png', codec='png', params=(),
              is_lossless=True, is_intra_only=True)

    def __init__(self, path, codec=None, store_factory=None, image_shape=None, callbacks=(), wipe=False):
        super(StoreWriter, self).__init__(path, codec=codec if codec else self.NPY, callbacks=callbacks)
        self._store_path = op.join(self.path, 'store')
        if wipe:
            shutil.rmtree(self.store_path, ignore_errors=True)

        if store_factory is None:
            store_factory = partial(
                new_for_format,
                fmt=codec.container,
                imgshape=image_shape,  # (width, height, channels),
                imgdtype=np.uint8,
                chunksize=1000,
                format=codec.container
            )
        self._store_factory = store_factory

        self._store = None

    @property
    def store_path(self):
        return self._store_path

    def is_open(self):
        return self._store is not None

    def open(self):
        if self.is_open():
            raise Exception('To reopen, please close first')
        self._store = self._store_factory(basedir=self.store_path)

    def _write_meta(self):
        with open(op.join(self.path, 'info.yaml'), 'wt') as writer:
            meta_dict = {
                'type': 'store',
                'store': {
                    'path': self.store_path
                }
            }
            yaml.safe_dump(meta_dict, writer)

    def _close_hook(self):
        if self.is_open():
            self._store.close()
            self._store = None

    def _add_hook(self, data):
        if not self.is_open():
            self.open()
        self._store.add_image(data, self._store.frame_count, np.nan)


# --- Readers

class OpenCVReader(Reader):
    """
    A thin wrapper over bview.OpenCVCapture objects
    (probably could work for other Capture impl.,
    but need to clearly define the meaning of "path" and "meta").
    """

    def __init__(self,
                 path,
                 callbacks=(),
                 factory=partial(OpenCVCapture,
                                 is_file=True,
                                 fail_if_unsupported_num_threads=True),
                 num_threads=None,
                 seek=None):

        super(OpenCVReader, self).__init__(path, callbacks=callbacks)

        # mmmm something feels really wrong here; but this is the easiest way
        # to allow parameterization for benchmark
        extra_factory_params = {}
        if num_threads is not None:
            extra_factory_params['num_threads'] = num_threads
        if seek is not None:
            extra_factory_params['seek'] = seek
        if extra_factory_params:
            factory = partial(factory, **extra_factory_params)

        self._factory = factory
        self._cap = None
        self._video_path = None

    def is_open(self):
        return self._cap is not None

    def open(self):
        if not self.is_open():
            try:
                self._cap = self._factory(self.path)
                self._video_path = self.path
            except IOError:
                self._cap = self._factory(op.join(self.path, self.meta['video']['file']))
                self._video_path = op.join(self.path, self.meta['video']['file'])

    @property
    def video_path(self):
        self.open()
        return self._video_path

    @property
    def cap(self):
        if not self.is_open():
            self.open()
        return self._cap

    def close(self):
        if self.is_open():
            self._cap.close()
            self._cap = None

    def _get_hook(self, frame_nums):
        # Not worrying here about the frameindex / framenumber complexity
        images = []
        for frame_num in frame_nums:
            self.cap.seek_index(frame_num)
            images.append(self.cap.grab_next_frame())
        return images

    @property
    def num_frames(self):
        return self.cap.frame_count

    @property
    def shape(self):
        return self.cap.image_shape


def balanced_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_images_from_store(store, frame_nums):
    store_disposable = False
    if isinstance(store, string_types):
        # dumb, store should allow parallel itself or at least be pickable
        # alternative, instantiate workers that poll a queue
        store = new_for_filename(op.join(store, STORE_MD_FILENAME), mode='r')
        store_disposable = True
    images = []
    for frame_num in frame_nums:
        # noinspection PyUnusedLocal
        image, (frame_num, frame_t) = store.get_image(frame_num)
        images.append((frame_num, image))
    if store_disposable:
        store.close()
    return images


class StoreReader(Reader):
    """
    A thin wrapper over bview.OpenCVCapture objects
    (probably could work for other Capture impl.,
    but need to clearly define the meaning of "path" and "meta").
    """

    # We probably could just reuse CaptureReader...

    def __init__(self, path, num_threads=1, min_frame_nums_for_parallel=0, callbacks=(), seek='unused'):
        super(StoreReader, self).__init__(path, callbacks=callbacks)
        self.num_threads = num_threads
        self.min_frame_nums_for_parallel = min_frame_nums_for_parallel
        self._store_path = op.join(self.path, 'store')
        self._store = None

    @property
    def store_path(self):
        return self._store_path

    def is_open(self):
        return self._store is not None

    def open(self):
        if not self.is_open():
            self._store = new_for_filename(op.join(self.store_path, STORE_MD_FILENAME), mode='r')

    def close(self):
        if self.is_open():
            self._store.close()
            self._store = None

    @property
    def store(self):
        if not self.is_open():
            self.open()
        return self._store

    def _get_hook(self, frame_nums):
        # quick and dirty "query optimization" + slow parallelism; missing jagged
        sorted_frame_nums = np.array(sorted(set(frame_nums)))
        if self.num_threads == 1 or len(sorted_frame_nums) < self.min_frame_nums_for_parallel:
            images = get_images_from_store(self.store, sorted_frame_nums)
        else:
            # slowest parallel implementation ever
            worker_frame_nums = [wfn for wfn in np.array_split(sorted_frame_nums, self.num_threads)
                                 if len(wfn)]
            images = Parallel(n_jobs=len(worker_frame_nums), backend='threading')(
                delayed(get_images_from_store)(self.store_path, wfn)
                for wfn in worker_frame_nums
            )
            images = list(chain.from_iterable(images))
        fn2i = {frame_num: i for i, frame_num in enumerate(frame_nums)}
        return [images[fn2i[frame_num]][1] for frame_num in frame_nums]

    @property
    def num_frames(self):
        return self.store.frame_count

    @property
    def shape(self):
        return self.store.image_shape


# --- Callbacks

# noinspection PyAbstractClass
class FFMPEGWatchCallback(WatchCallback):

    def __init__(self):
        super(FFMPEGWatchCallback, self).__init__()
        self.ffmpeg_watch = PSUtilRusager()

    def __call__(self, event, context):
        super(FFMPEGWatchCallback, self).__call__(event, context)
        # WTF²
        # This will miss the first frame if the FFMPEGWriter was not yet initialized... oh well...
        # A pity RUSAGE_CHILDREN does not work here
        if event in (Writer.EVENT_ADD_IN,):
            if self.ffmpeg_watch.process is None:
                caller = context['caller']
                if isinstance(caller, FFMPEGWriter) and caller.is_open():
                    self.ffmpeg_watch.process = caller.process
            if self.ffmpeg_watch.process is not None:
                self.ffmpeg_watch.tic()
        elif event in (Writer.EVENT_ADD_OUT,) and self.ffmpeg_watch.process is not None:
            self.ffmpeg_watch.toc(count=self._count(context))

    # TODO: we would need to fix other measurements (e.g. means, stdevs)... do time allowing
    #       anyway this is just used for RecorderFFMPEG writer, so not very interesting

    @property
    def user_time(self):
        return self.watch.user_time + self.ffmpeg_watch.ru_utime.total

    @property
    def system_time(self):
        return self.watch.system_time + self.ffmpeg_watch.ru_stime.total

    @property
    def num_units(self):
        return self.watch.num_units

    @property
    def ups_wall(self):
        return self.watch.ups_wall

    @property
    def ups_ru(self):
        return self.num_units / (self.user_time + self.system_time)
