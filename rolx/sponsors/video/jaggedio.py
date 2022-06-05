# coding=utf-8
import os.path as op
from functools import partial

import cv2
import numpy as np
import pandas as pd
from jagged.blosc_backend import JaggedByBlosc

from rolx.sponsors.video.videocodecs import VideoEncoderConfig
from rolx.sponsors.video.io import Writer, Reader
from rolx.utils import ensure_dir


# --- Trying to fit this into the FFMPEGRecorder + Store design is quite complex
# Need to change a bit base clases to make them more general and with better separation of concerns.

class DummyCompressor(object):

    def __init__(self, codec='npy'):
        super(DummyCompressor, self).__init__()
        self.codec = codec if not codec.startswith('.') else codec[1:]

    # noinspection PyMethodMayBeStatic
    def compress(self, data):
        return data

    # noinspection PyMethodMayBeStatic
    def decompress(self, data):
        return data


class OpenCVCompressor(DummyCompressor):

    # https://docs.opencv.org/3.4.1/d4/da8/group__imgcodecs.html

    def __init__(self, codec='jpg', jpeg_quality=95):
        super(OpenCVCompressor, self).__init__()

        # TODO: just accept "codec" that will provide both ext and options
        # But for that we need to stop using a base Codec class influenced by both ffmpeg and RecorderFFMPEG

        # Many others are exposed, see:
        #   https://docs.opencv.org/3.4.1/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac
        # But we should just start using turbo API directly
        self.codec = codec if not codec.startswith('.') else codec[1:]
        self.jpeg_quality = jpeg_quality

        self._imread_mode = cv2.IMREAD_COLOR

    def compress(self, data):

        # Remember, imencode allows to specify output buffer, which if well implemented,
        # can be quite interesting.
        #
        #     imencode(ext, img[, params]) -> retval, buf
        # .   @brief Encodes an image into a memory buffer.
        # .
        # .   The function imencode compresses the image and stores it in the memory buffer that is resized to fit the
        # .   result. See cv::imwrite for the list of supported formats and flags description.
        # .
        # .   @param ext File extension that defines the output format.
        # .   @param img Image to be written.
        # .   @param buf Output buffer resized to fit the compressed image.
        # .   @param params Format-specific parameters. See cv::imwrite and cv::ImwriteFlags.

        success, compressed = cv2.imencode('.' + self.codec, data, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not success:
            raise ValueError('cannot compress the image')
        return compressed

    def decompress(self, data):

        #     imdecode(buf, flags) -> retval
        # .   @brief Reads an image from a buffer in memory.
        # .
        # .   The function imdecode reads an image from the specified buffer in the memory.
        # .   If the buffer is too short or contains invalid data, the function returns
        # .   an empty matrix ( Mat::data==NULL ).
        # .
        # .   See cv::imread for the list of supported formats and flags description.
        # .
        # .   @note In the case of color images, the decoded images will have the channels stored in **B G R** order.
        # .   @param buf Input array or vector of bytes.
        # .   @param flags The same flags as in cv::imread, see cv::ImreadModes.

        uncompressed = cv2.imdecode(np.frombuffer(data, np.uint8), self._imread_mode)
        return uncompressed


class JaggedByCompressor(JaggedByBlosc):

    def __init__(self, path=None, journal=None, compressor=DummyCompressor):
        super(JaggedByCompressor, self).__init__(path, journal, compressor)
        # Need to finish that branch to support arbitrary ndims asap
        self._real_shape = None

    def _compressor(self):
        if not isinstance(self.compressor, DummyCompressor):
            self.compressor = self.compressor()
        return self.compressor

    def _write_real_shape(self, data):
        shape_dir = ensure_dir(op.join(self.path_or_fail(), 'meta', 'real_shape'))
        shape_path = op.join(shape_dir, 'shape.pkl')
        pd.to_pickle(data.shape, shape_path)

    def video_shape(self):
        if self._real_shape is None:
            shape_dir = op.join(self.path_or_fail(), 'meta', 'real_shape')
            shape_path = op.join(shape_dir, 'shape.pkl')
            if op.isfile(shape_path):
                self._real_shape = pd.read_pickle(shape_path)
        return self._real_shape

    @property
    def ndims(self):
        return 3  # Yeah, always include color depth

    @property
    def shape(self):
        raw_shape = super(JaggedByCompressor, self).shape
        n_images = np.prod(raw_shape) // np.prod(self.video_shape())
        return (n_images,) + self.video_shape()

    def append(self, data):
        if self.video_shape() is None:
            self._write_real_shape(data)
        data = data.reshape(-1, 1)
        return super(JaggedByCompressor, self).append(data)

    def _append_hook(self, data):
        compressor = self._compressor()
        compressed = compressor.compress(data.reshape(self.video_shape()))
        self._mm.write(compressed)
        self._bytes_journal().append(compressed)


# Keep this alias as some results were computed when it was the name of the class
JaggedByOpenCV = JaggedByCompressor


class JaggedWriter(Writer):

    def __init__(self,
                 path,
                 codec=None,
                 num_threads=1,
                 callbacks=()):
        if codec is None:
            print(codec)
            codec = jagged_jpeg_95
        super(JaggedWriter, self).__init__(path=path, codec=codec, callbacks=callbacks)
        self._jagged = self.codec.jagged_factory(path)
        if num_threads is not None and num_threads != 1:
            raise ValueError('Multithreaded Jagged is not implemented yet')
        self._num_threads = num_threads

    @property
    def num_threads(self):
        return self.num_threads

    def is_open(self):
        return self._jagged.is_open

    def open(self):
        pass

    def _write_meta(self):
        pass

    def _close_hook(self):
        self._jagged.close()

    def _add_hook(self, data):
        self._jagged.append(data)


class JaggedReader(Reader):

    # N.B. seek and num_threads, due to tense design... will disappear ASAP
    # noinspection PyUnusedLocal
    def __init__(self, path, codec=None, callbacks=(), num_threads=1, seek=None):
        super(JaggedReader, self).__init__(path, callbacks)
        if codec is None:
            codec = jagged_jpeg_95
        self._codec = codec
        self._jagged = self.codec.jagged_factory(path)
        if num_threads is not None and num_threads != 1:
            raise ValueError('Multithreaded Jagged is not implemented yet')
        self._num_threads = num_threads

    @property
    def num_threads(self):
        return self.num_threads

    @property
    def codec(self):
        return self._codec

    def open(self):
        pass

    def close(self):
        self._jagged.close()

    def _get_hook(self, frame_nums):
        return self._jagged.get(frame_nums)

    @property
    def num_frames(self):
        return len(self._jagged)

    @property
    def shape(self):
        return self._jagged.shape


class JaggedCodec(VideoEncoderConfig):

    def __init__(self,
                 name,
                 container,
                 codec,
                 params,
                 jagged_factory,
                 is_lossless=False, is_intra_only=True,
                 **meta):
        self._python_params = {param: value for param, value in zip(params[::2], params[1::2])}
        self._jagged_factory = jagged_factory
        super(JaggedCodec, self).__init__(name,
                                          container,
                                          codec,
                                          params,
                                          is_lossless,
                                          is_intra_only,
                                          writer=JaggedWriter,
                                          reader=JaggedReader,
                                          **meta)

    @property
    def jagged_factory(self):
        return self._jagged_factory

    def writer(self, **options):
        return partial(self.meta['writer'], codec=self, **options)

    def reader(self, **options):
        return partial(self.meta['reader'], codec=self, **options)


jagged_dummy = JaggedCodec(
    name='jagged_dummy',
    container='jagged',
    codec='npy-jagged',
    params=(),
    jagged_factory=partial(JaggedByCompressor, compressor=DummyCompressor),
    is_lossless=True, is_intra_only=True
)

jagged_jpeg_95 = JaggedCodec(
    name='jagged_jpeg_95',
    container='jagged',
    codec='jpg-jagged',
    params=('codec', 'jpg', 'jpeg_quality', 95),
    jagged_factory=partial(JaggedByCompressor, compressor=partial(OpenCVCompressor,
                                                                  codec='jpg',
                                                                  jpeg_quality=95)),
    is_lossless=False, is_intra_only=True
)
