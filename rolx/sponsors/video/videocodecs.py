# coding=utf-8
"""Video encoding configurations."""
from __future__ import print_function, division

from functools import partial

import os.path as op
from subprocess import check_output

from rolx.sponsors.video.ffmpeg import FFMPEGQ, normalize_params


class VideoEncoderConfig(object):
    """Helper to manipulate video encoding configurations."""

    def __init__(self,
                 name='libx264-maxcompat',
                 container='mp4',
                 codec='libx264',
                 params=('-preset', 'medium', '-crf', '20', '-profile:v', 'baseline',
                         '-level', '3.0', '-pix_fmt', 'yuv420p'),
                 is_lossless=False, is_intra_only=False,
                 writer=None, reader=None,
                 **meta):

        super(VideoEncoderConfig, self).__init__()

        self._name = name
        self._container = container if not container.startswith('.') else container[1:]
        self._codec = codec
        self._params = normalize_params(params, sort=False)

        if is_lossless is None:
            is_lossless = FFMPEGQ.ffq().is_lossless(self.codec, self.params)
        self._is_lossless = is_lossless

        if is_intra_only is None:
            is_intra_only = FFMPEGQ.ffq().is_intra(self.codec, self.params)
        self._is_intra_only = is_intra_only

        self.meta = meta
        self.meta['writer'] = writer
        self.meta['reader'] = reader

    @property
    def name(self):
        return self._name

    @property
    def container(self):
        return self._container

    def writer(self, **options):
        if self.meta['writer'] is None:
            return None
        return partial(self.meta['writer'], codec=self, **options)

    def reader(self, **options):
        if self.meta['reader'] is None:
            return None
        return partial(self.meta['reader'], **options)

    @property
    def codec(self):
        return self._codec

    @property
    def params(self):
        return self._params

    @property
    def is_lossless(self):
        return self._is_lossless

    @property
    def is_intra_only(self):
        return self._is_intra_only

    @property
    def pix_fmt(self):
        return self.meta.get('pix_fmt', self.param_value('pix_fmt'))

    @property
    def param_tuples(self):
        return normalize_params(self.params, as_tuples=True)

    def param_value(self, param_name):
        if not param_name.startswith('-'):
            param_name = '-' + param_name
        for name, value in self.param_tuples:
            if name == param_name:
                return value
        return None

    # --- Interaction with RecorderFFMPEG

    @staticmethod
    def check_recorder_codec_assumptions():
        from loopb.record import RecorderFFMPEG
        for name, config in RecorderFFMPEG.DEFAULT_CODECS.items():
            assert len(config['codec_params']) % 2 == 0
            assert len(config) == 4

    def to_recorder_dict(self, pix_fmt_i='bgr24'):
        return {
            self.name: {
                'pix_fmt_i': pix_fmt_i,
                'container': '.' + self.container,
                'videocodec': self.codec,
                'codec_params': list(self.params)
            }
        }

    @classmethod
    def from_recorder_dict(cls, recorder_config, name=None, is_intra=None, is_lossless=None, **further_params):
        if len(recorder_config) == 1:
            name, recorder_config = recorder_config.items()[0]
        if name is None:
            raise ValueError('Unspecified encoder configuration name')
        return cls(
            name=name,
            container=recorder_config['container'],
            codec=recorder_config['videocodec'],
            params=recorder_config['codec_params'],
            is_lossless=is_lossless,
            is_intra=is_intra,
            **further_params
        )

    # --- Misc.

    @staticmethod
    def encoder_options(codec='mjpeg', ffmpeg='ffmpeg'):
        # See also:
        #  https://ffmpeg.org/ffmpeg.html#Main-options
        #  https://ffmpeg.org/ffmpeg.html#AVOptions
        return check_output([ffmpeg, '-h', 'encoder=%s' % codec])

    @staticmethod
    def color_transform_is_lossless(pix_fmt_i, pix_fmt):
        if pix_fmt is None:
            raise Exception('Unknown dest pix_fmt')
        if pix_fmt == pix_fmt_i:
            return True
        if pix_fmt.contains('yuv420p'):
            return pix_fmt_i == 'yuv420p'
        # A lot to complete here...
        return None

    def path(self, stem, dirname=None):
        fn = stem + '.' + self.container  # assume extension is the same as container id
        return op.join(dirname, fn) if dirname else fn

    # TODO: bring back this
    #       we would need clever defaults for things like "is_lossless"
    # @staticmethod
    # def conf_variants_product(variants, namer=None):
    #     """
    #     Iterate over the cross-product of configuration variants.
    #     This allows for simple declaration of configuration variants, where all parameters can be combined freely.
    #     Further filtering for invalid configurations is the best way to deselect invalid combinations.
    #     Useful for brute-force exploring the codec space, interesting variants should be manually selected and named.
    #
    #     Parameters
    #     ----------
    #     variants : dictionary
    #       The variants to try.
    #       For encoder configurations, usually these need to contain at least container, pix_fmt and codec.
    #
    #     namer : function (config-dict: name) or None
    #       A function
    #
    #     Yields
    #     ------
    #     A `VideoEncoderConfig` for each configuration variant.
    #
    #     Examples
    #     --------
    #
    #     >>> # ffmpeg -h encoder=ffv1
    #     >>> ffv1_variants = dict(
    #     ...   container='avi',
    #     ...   pix_fmt='yuv420p',
    #     ...   codec='ffv1',
    #     ...   slicecrc=[False, True],    # None would be auto
    #     ...   coder=['rice', 'range_def', 'range_tab', 'ac'],
    #     ...   context=[False, True]
    #     ... )
    #     >>> codecs = VideoEncoderConfig.conf_variants_product(ffv1_variants)
    #
    #     >>> # ffmpeg -h encoder=ffvhuff
    #     >>> ffvhuff_variants = dict(
    #     ...   container='.avi',
    #     ...   pix_fmt=['yuv420p', 'rgb24'],     # Note that we can also add pix_fmt variants
    #     ...   codec='ffvhuff',
    #     ...   non_deterministic=[True, False],
    #     ...   pred=['left', 'plane', 'median'],
    #     ...   context=[False, True]
    #     ... )
    #     >>> codecs = VideoEncoderConfig(ffvhuff_variants)
    #     """
    #
    #     if namer is None:
    #         def namer(config):
    #             return config['name'] + '_' + config['pix_fmt'] + 'FIXME'
    #
    #     param_names = sorted(variants)
    #     param_values = [variants[param_name] for param_name in param_names]
    #     param_values = [[values] if isinstance(values, string_types) else values for values in param_values]
    #
    #     for config in product(*param_values):
    #         zip(param_names, config)
    #         yield VideoEncoderConfig(name=namer())


VEC = VideoEncoderConfig
