from __future__ import print_function, division

import cv2
import numpy as np
from whatami import What

from rolx.benchmarks.jpegs.turbojpeg import TurboJPEG, TJPF, TJSAMP


class DummyCodec(object):

    def __init__(self):
        super(DummyCodec, self).__init__()

    # noinspection PyMethodMayBeStatic
    def encode(self, data):
        return data

    # noinspection PyMethodMayBeStatic
    def decode(self, data):
        return data

    def what_encoder(self):
        return What(name='encoder', conf=self.encoder_params())

    def encoder_params(self):
        params = {'name': self.__class__.__name__}
        params.update(self._encoder_params_hook())
        return params

    # noinspection PyMethodMayBeStatic
    def _encoder_params_hook(self):
        return {}

    def what_decoder(self):
        return What(name='decoder', conf=self.decoder_params())

    def decoder_params(self):
        params = {'name': self.__class__.__name__}
        params.update(self._decoder_params_hook())
        return params

    # noinspection PyMethodMayBeStatic
    def _decoder_params_hook(self):
        return {}


class JPEGOpenCVCodec(DummyCodec):

    # See:
    #   https://docs.opencv.org/3.4.1/d4/da8/group__imgcodecs.html
    #   https://docs.opencv.org/3.4.1/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac

    # Remember, imdecode and imencode imdecode allows to specify output buffer,
    # which if well implemented, can be quite interesting.

    def __init__(self,
                 encode_quality=95,
                 encode_progressive=False,
                 encode_optimize=False,
                 encode_restart_interval=0):
        super(JPEGOpenCVCodec, self).__init__()

        # Encoding params
        self.encode_quality = encode_quality
        self.encode_progressive = encode_progressive
        self.encode_optimize = encode_optimize
        self.encode_restart_interval = encode_restart_interval

        # Decoding params
        self.decode_mode = cv2.IMREAD_COLOR

    def encode(self, data):

        success, compressed = cv2.imencode('.jpg', data, [
            cv2.IMWRITE_JPEG_QUALITY, self.encode_quality,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0 if not self.encode_progressive else 1,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0 if not self.encode_optimize else 1,
            cv2.IMWRITE_JPEG_RST_INTERVAL, self.encode_restart_interval,
            # These are marked as "don't use", and if used, destroy the image...
            # cv2.IMWRITE_JPEG_LUMA_QUALITY, self.encode_luma_quality,
            # cv2.IMWRITE_JPEG_CHROMA_QUALITY, self.encode_chroma_quality,
        ])
        if not success:
            raise ValueError('cannot compress the image')
        return compressed.data

    def _encoder_params_hook(self):
        return dict(
            encode_quality=self.encode_quality,
            encode_progressive=self.encode_progressive,
            encode_optimize=self.encode_optimize,
            encode_restart_interval=self.encode_restart_interval,
        )

    def decode(self, data):
        return cv2.imdecode(np.frombuffer(data, np.uint8), self.decode_mode)

    def _decoder_params_hook(self):
        return dict(decode_mode=self.decode_mode)


class JPEGTurboCodec(DummyCodec):

    def __init__(self,
                 turbo=None,
                 encode_quality=95,
                 encode_pixel_format=TJPF.BGR,
                 encode_jpeg_subsample=TJSAMP.YUV422,
                 encode_progressive=False,
                 encode_fast_dct=True,
                 encode_accurate_dct=False,
                 decode_pixel_format=TJPF.BGR,
                 decode_scaling_factor=None,
                 decode_fast_upsample=False,
                 decode_fast_dct=False,
                 decode_accurate_dct=False):
        super(JPEGTurboCodec, self).__init__()

        self.turbo = turbo if turbo is not None else TurboJPEG()

        # Encoding params
        self.encode_quality = encode_quality
        self.encode_pixel_format = encode_pixel_format
        self.encode_jpeg_subsample = encode_jpeg_subsample
        self.encode_progressive = encode_progressive
        self.encode_fast_dct = encode_fast_dct
        self.encode_accurate_dct = encode_accurate_dct

        # Decoding params
        # Note, some combinations might make no sense
        self.decode_pixel_format = decode_pixel_format
        self.decode_scaling_factor = decode_scaling_factor
        self.decode_fast_upsample = decode_fast_upsample
        self.decode_fast_dct = decode_fast_dct
        self.decode_accurate_dct = decode_accurate_dct

    def encode(self, data):
        return self.turbo.encode(data,
                                 quality=self.encode_quality,
                                 pixel_format=self.encode_pixel_format,
                                 jpeg_subsample=self.encode_jpeg_subsample,
                                 progressive=self.encode_progressive,
                                 fast_dct=self.encode_fast_dct,
                                 accurate_dct=self.encode_accurate_dct)

    def decode(self, data):
        return self.turbo.decode(data,
                                 pixel_format=self.decode_pixel_format,
                                 scaling_factor=self.decode_scaling_factor,
                                 fast_upsample=self.decode_fast_upsample,
                                 fast_dct=self.decode_fast_dct,
                                 accurate_dct=self.decode_accurate_dct)

    def _encoder_params_hook(self):
        return dict(
            encode_quality=self.encode_quality,
            encode_pixel_format=self.encode_pixel_format,
            encode_jpeg_subsample=self.encode_jpeg_subsample,
            encode_progressive=self.encode_progressive,
            encode_fast_dct=self.encode_fast_dct,
            encode_accurate_dct=self.encode_accurate_dct,
        )

    def _decoder_params_hook(self):
        return dict(
            decode_pixel_format=self.decode_pixel_format,
            decode_scaling_factor=self.decode_scaling_factor,
            decode_fast_upsample=self.decode_fast_upsample,
            decode_fast_dct=self.decode_fast_dct,
            decode_accurate_dct=self.decode_accurate_dct,
        )

#
# FIXME: I liked more the tentative design in videobenchmark, with encoder and decoder separated
#        Rework
#
# FIXME: as usual, make the API take a collection of images to compress / decompress
#
# FIXME: eliminate as much as possible python & calling overhead, that can dominate for smaller images
#        e.g. no context managers in TurboJPEG bindings
#
# FIXME: allow custom dest & origin memory addresses (+ pin memory)
#
# TODO: wrap tensorflow jpeg bindings into a Codec
#
