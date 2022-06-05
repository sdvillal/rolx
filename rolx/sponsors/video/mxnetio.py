# coding=utf-8
"""
Video reader writer on top of mxnet RecordIO.

Some mxnet docs & sources:
  https://mxnet.incubator.apache.org/architecture/note_data_loading.html
  https://mxnet.incubator.apache.org/faq/recordio.html
  https://mxnet.incubator.apache.org/api/python/io/io.html
  https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.cc
  https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py

This is one of the inspirations of GulpIO (which we won't try until we migrate to py3):
  https://github.com/TwentyBN/GulpIO
  https://github.com/TwentyBN/GulpIO-benchmarks
  https://medium.com/twentybn/introducing-gulpio-f97b07f1da58
  https://www.reddit.com/r/MachineLearning/comments/6yv7sz/n_gulpio_an_opensource_io_framework_for_faster/
"""
import mxnet as mx

from rolx.sponsors.video.io import Writer

mx.recordio.pack_img()


class RecordIOWriter(Writer):

    def __init__(self, path, codec=None, callbacks=()):
        super(RecordIOWriter, self).__init__(path, codec, callbacks)

    def is_open(self):
        pass

    def open(self):
        pass

    def _write_meta(self):
        pass

    def _close_hook(self):
        pass

    def _add_hook(self, data):
        pass
