# coding=utf-8
"""Generic image providers, with strong support to use video files."""
from __future__ import print_function, division

import os.path as op

import numpy as np
from ruamel import yaml

from rolx.sponsors.video.checks import Barcoder
from rolx.utils import du, sync, drop_caches
from rolx.watches import Rusager, timer


# --- Writers

class Writer(object):

    EVENT_ADD_IN = 'event-add-in'
    EVENT_ADD_OUT = 'event-add-out'
    EVENT_CLOSE = 'event-close-in'

    def __init__(self, path, codec=None, callbacks=()):
        super(Writer, self).__init__()
        self._path = path
        self._codec = codec
        self._callbacks = callbacks

    @property
    def codec(self):
        return self._codec

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def path(self):
        return self._path

    def is_open(self):
        raise NotImplementedError()

    def open(self):
        raise NotImplementedError()

    def close(self):
        if self.is_open():
            self._write_meta()
            self._close_hook()
            context = {'caller': self}
            for callback in self._callbacks:
                callback(event=self.EVENT_CLOSE, context=context)

    def _write_meta(self):
        raise NotImplementedError()

    def _close_hook(self):
        raise NotImplementedError()

    def add(self, data):
        context = {'caller': self, 'data': data}
        for callback in self._callbacks:
            callback(event=self.EVENT_ADD_IN, context=context)
        self._add_hook(data)
        for callback in self._callbacks[::-1]:
            callback(event=self.EVENT_ADD_OUT, context=context)

    def _add_hook(self, data):
        raise NotImplementedError()

    @property
    def du(self):
        return du(self.path)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()


# --- Readers


class Reader(object):

    EVENT_GET_IN = 'event-get-in'
    EVENT_GET_OUT = 'event-get-out'

    def __init__(self, path, callbacks=()):
        super(Reader, self).__init__()
        self._path = path
        self._callbacks = callbacks

    @property
    def path(self):
        return self._path

    @property
    def callbacks(self):
        return self._callbacks

    def read_meta(self):
        with open(op.join(self.path, 'info.yaml'), 'rt') as reader:
            return yaml.safe_load(reader)

    @property
    def meta(self):
        return self.read_meta()

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def get(self, frame_nums):
        if isinstance(frame_nums, int):
            frame_nums = [frame_nums]
        context = {'caller': self, 'frame_nums': frame_nums}
        for callback in self._callbacks:
            callback(event=self.EVENT_GET_IN, context=context)
        context['images'] = self._get_hook(frame_nums)
        for callback in self._callbacks[::-1]:
            callback(event=self.EVENT_GET_OUT, context=context)
        return context['images']

    def _get_hook(self, frame_nums):
        raise NotImplementedError()

    @property
    def num_frames(self):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def du(self):
        return du(self.path)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError('Only single frame indexing allowed at the moment')
        return self.get(item)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def __iter__(self):
        for i in range(len(self)):
            return i, self[i][0]
        # Also add:
        #   Custom range & subsampling iters
        #   Batched iters


# --- Useful callbacks for timing and benchmarking


class BarcodeCallback(object):

    # Assumes we are expecting barcode == frame_num is store

    def __init__(self, barcoder=Barcoder()):
        super(BarcodeCallback, self).__init__()
        self.barcoder = barcoder
        self._total = 0
        self._correct = 0

    def __call__(self, event, context):
        if event in (Reader.EVENT_GET_OUT,):
            for frame_num, image in zip(context['frame_nums'], context['images']):
                self._total += 1
                if frame_num == self.barcoder.decode(image):
                    self._correct += 1

    @property
    def total(self):
        return self._total

    @property
    def correct(self):
        return self._correct

    @property
    def pct_correct(self):
        return 100 * self.correct / self.total


class WatchCallback(object):

    def __init__(self, watch=None):
        super(WatchCallback, self).__init__()
        self.watch = Rusager() if watch is None else watch

    def __call__(self, event, context):
        if event in (Reader.EVENT_GET_IN, Writer.EVENT_ADD_IN, Writer.EVENT_CLOSE):
            self.watch.tic()
        elif event in (Reader.EVENT_GET_OUT, Writer.EVENT_ADD_OUT):
            self.watch.toc(count=self._count(context))

    @staticmethod
    def _count(context):
        frame_count = len(context.get('frame_nums', []))
        frame_count = 1 if not frame_count else frame_count
        return frame_count


class FSCachePurgerCallback(object):

    def __init__(self,
                 seed=0,
                 purge_chance=0.0,
                 maybe_purge_each=1,
                 min_frame_distance=0,
                 max_purges_per_event=1):
        super(FSCachePurgerCallback, self).__init__()
        self.seed = seed
        self.purge_chance = purge_chance
        self.maybe_purge_each = maybe_purge_each
        self.purge_timer = timer('purges', units='op')
        self.min_frame_distance = min_frame_distance
        self.max_purges_per_event = max_purges_per_event
        self.num_purges_in_event = False
        self._last_frame_num = None
        self._purge_countdown = maybe_purge_each
        self._rng = np.random.RandomState(seed)

    def __call__(self, event, context):
        if event in (Reader.EVENT_GET_IN,):
            self.num_purges_in_event = 0
            # Allowed by countdown?
            for frame_num in context['frame_nums']:
                self._purge_countdown -= 1
                if self._purge_countdown <= 0:
                    self._purge_countdown = self.maybe_purge_each
                    # Allowed by frame distance?
                    can_purge = True
                    if self._last_frame_num is not None:
                        can_purge = abs(frame_num - self._last_frame_num) > self.min_frame_distance
                    # Allowed by chance?
                    if can_purge:
                        if self._rng.uniform() < self.purge_chance:
                            with self.purge_timer:
                                drop_caches(context['caller'].path)
                            self.num_purges_in_event += 1
                if self.num_purges_in_event >= self.max_purges_per_event:
                    self._last_frame_num = context['frame_nums'][-1]
                    break
                else:
                    self._last_frame_num = frame_num


class SyncAfterClose(object):

    # Probably this could be a top level sync_on_close=False option of writer?
    # Timings here can be misleading (we do not know what extra work sync is doing)
    # To use rusage here, we would need to waitpid for the PID of the sync command and poll at the end

    def __init__(self):
        super(SyncAfterClose, self).__init__()
        self.watch = timer('sync', 'sync')

    def __call__(self, event, context):
        if event in (Writer.EVENT_CLOSE,):
            with self.watch:
                sync()
