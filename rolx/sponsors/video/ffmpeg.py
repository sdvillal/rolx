# coding=utf-8
"""Benchmarking image providers."""
from __future__ import print_function, division
from future.utils import string_types

from itertools import chain

import os
from subprocess import check_output

import json

import pandas as pd


class FFMPEGQ(object):

    _FFMPEGQS = {}

    def __init__(self, ffmpeg='ffmpeg'):
        super(FFMPEGQ, self).__init__()
        self.ffmpeg = ffmpeg
        self._codecs = None

    @staticmethod
    def ffq(ffmpeg='ffmpeg', reread=False):
        """
        :rtype : FFMPEGQ
        """
        if reread or ffmpeg not in FFMPEGQ._FFMPEGQS:
            FFMPEGQ._FFMPEGQS[ffmpeg] = FFMPEGQ(ffmpeg=ffmpeg)
        return FFMPEGQ._FFMPEGQS[ffmpeg]

    # --- codecs

    @staticmethod
    def available_codecs(ffmpeg='ffmpeg', as_pandas=True):
        # Codecs:
        #  D..... = Decoding supported
        # .E.... = Encoding supported
        # ..V... = Video codec
        # ..A... = Audio codec
        # ..S... = Subtitle codec
        # ...I.. = Intra frame-only codec
        # ....L. = Lossy compression
        # .....S = Lossless compression
        # -------
        #  D.VI.S 012v                 Uncompressed 4:2:2 10-bit

        codecs = []
        with open(os.devnull, 'wb') as DEVNULL:
            ffmpeg_output = check_output([ffmpeg, '-codecs'], stderr=DEVNULL)
            ffmpeg_output = ffmpeg_output.partition('-------\n')[2]
        for codec in ffmpeg_output.splitlines():
            codec_info, codec_name, codec_desc = codec.strip().split(' ', 2)
            codecs.append({
                'name': codec_name,
                'type': 'video' if 'V' in codec_info else 'audio' if 'A' in codec_info else 'subtitle',
                'can_decode': 'D' in codec_info,
                'can_encode': 'E' in codec_info,
                'intra_only': 'I' in codec_info,
                'can_lossy': 'L' in codec_info,
                'can_lossless': 'S' in codec_info,
                'description': codec_desc.strip()
            })

        names = [codec['name'] for codec in codecs]
        if len(set(names)) != len(names):
            raise Exception('There are repeated codec names')

        if not as_pandas:
            return {codec['name']: codec for codec in codecs}

        # noinspection PyTypeChecker
        codecs = pd.DataFrame(codecs)
        columns = ['name', 'type',
                   'can_decode', 'can_encode',
                   'intra_only',  'can_lossy', 'can_lossless',
                   'description']
        columns += [col for col in codecs.columns if col not in columns]
        return codecs[columns]

    @property
    def codecs(self):
        if self._codecs is None:
            self._codecs = self.available_codecs(ffmpeg=self.ffmpeg, as_pandas=False)
        return self._codecs

    def codec_info(self, codec, fail_if_unknown=True):
        try:
            return self.codecs[codec]
        except KeyError:
            if fail_if_unknown:
                raise KeyError('No info for codec %r' % codec)
            return None

    def codec_prop(self, codec, prop, fail_if_unknown=False):
        codec_info = self.codec_info(codec, fail_if_unknown=fail_if_unknown)
        if codec_info:
            return codec_info[prop]
        return None

    def is_video(self, codec):
        return self.codec_prop(codec, 'type') == 'video'

    def is_audio(self, codec):
        return self.codec_prop(codec, 'type') == 'audio'

    def is_subtitle(self, codec):
        return self.codec_prop(codec, 'type') == 'subtitle'

    def is_intra_only(self, codec):
        return self.codec_prop(codec, 'intra_only')

    # noinspection PyUnusedLocal
    def is_intra(self, codec, params=None):
        """
        Examples
        --------
        >>> ffq = FFMPEGQ.ffq()
        When codecs are not intra-only, we would need to further look at their
        parameterization:
        >>> assert not ffq.is_intra_only('ffv1')
        >>> assert ffq.is_intra_only('ffv1', params=None)
        >>> assert not ffq.is_intra_only('ffv1', params=['-g', 1])
        >>> assert not ffq.is_intra_only('ffv1', params=['-g', 2])

        Obviously, we can know about intra-only codecs without looking at their params
        >>> assert ffq.is_intra_only('ffvhuff')
        >>> assert ffq.is_intra_only('ffvhuff', params=None)
        >>> assert ffq.is_intra_only('mjpeg')
        >>> assert ffq.is_intra_only('mjpeg', params=None)

        We still need to implement quite a bit of logic, do if useful...
        >>> assert not ffq.is_intra_only('h264')        >>> assert ffq.is_intra_only('h264', params=None) is None

        """
        if self.is_intra_only(codec):
            return True
        if codec == 'ffv1':
            if params is not None:
                gop_size = get_param(params, '-g', 1)
                return gop_size == 1
            return True  # ffv1 default is Intra Only
        # None -> "uncertain"
        # Here we would need to complete using params
        return None

    def can_lossy(self, codec):
        return self.codec_prop(codec, 'can_lossy')

    def can_lossless(self, codec):
        return self.codec_prop(codec, 'can_lossless')

    def lossy_only(self, codec):
        return self.can_lossy(codec) and not self.can_lossless(codec)

    def lossles_only(self, codec):
        return not self.can_lossy(codec) and self.can_lossless(codec)

    # noinspection PyUnusedLocal
    def is_lossless(self, codec, params=None):
        """
        Examples
        --------
        >>> ffq = FFMPEGQ.ffq()
        # Some codecs can be used in both lossless and lossy regime => we get None
        >>> assert ffq.is_lossless('h264') is None
        # These are always lossless
        >>> assert ffq.is_lossless('ffv1')
        >>> assert ffq.is_lossless('ffvhuff')
        # These are always lossy
        >>> assert not ffq.is_lossless('mjpeg')
        """
        if self.lossles_only(codec):
            return True
        if self.lossy_only(codec):
            return False
        # None -> "uncertain"
        # Here we would need to infer using params
        return None

    # --- formats

    @staticmethod
    def available_formats(ffmpeg='ffmpeg'):
        return check_output([ffmpeg, '-formats'], stderr=None)


# --- (Encoder) Param list normalization

def normalize_param_name(param_name):
    param_name = param_name.strip()
    if param_name.startswith('-'):
        return param_name
    return '-' + param_name


def normalize_param_value(value):
    if value is True:
        return '1'
    if value is False:
        return '0'
    return value


def normalize_params(params, sort=True, as_tuples=False):
    normalized_params = []
    if isinstance(params, dict):
        params = list(chain.from_iterable(params.items()))
    if len(params) % 2 != 0:
        raise ValueError('codec params must be a list-like [param_name, param_value]* or a dictionary')
    params = list(zip(params[0::2], params[1::2]))
    for param_name, param_value in params:
        if not isinstance(param_name, string_types):
            raise ValueError('param_name %r is not a string' % param_name)
        normalized_params += [(normalize_param_name(param_name),
                               normalize_param_value(param_value))]
    params = normalized_params
    if sort:
        params = sorted(params)
    if as_tuples:
        return tuple(params)
    return tuple(chain.from_iterable(params))


def get_param(params, param_name, default=None):
    return dict(normalize_params(params, as_tuples=True)).get(param_name, default)


# --- Poll video information

def read_frames_info(video_path, ffprobe='ffprobe', only_first_s=None, as_dataframe=True):

    #
    # man ffprobe
    #   -show_format
    #   -show_streams
    #   -show_packets
    #   -show_frames
    # alternatively, directly with ffmpeg
    #  ffmpeg -i input.mp4 -vf select='eq(n,334)',showinfo -f null -
    #

    command = [
        ffprobe,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_frames'
    ]

    if only_first_s is not None:
        command += ['-read_intervals', '%%+%d' % only_first_s]

    command += [video_path]

    output = check_output(command)

    if not output:
        raise Exception('No frames info for video %s' % video_path)

    frames = json.loads(output)['frames']

    if as_dataframe:
        return pd.DataFrame(frames)
    return frames


def infer_gops(video_path_or_frames_info, only_first_s=5):
    # random nice reads:
    #   https://en.wikipedia.org/wiki/Group_of_pictures
    #   https://www.hhi.fraunhofer.de/en/departments/vca/research-groups/image-video-coding/research-topics/hierarchical-prediction-structures.html
    #   http://www.tiliam.com/Blog/2015/07/06/effective-use-long-gop-video-codecs
    #   https://stackoverflow.com/questions/18085458/checking-keyframe-interval
    #   https://superuser.com/questions/604858/ffmpeg-extract-intra-frames-i-p-b-frames
    # do not use this seriously...
    # should also check for things like open vs closed gop
    #   https://stackoverflow.com/questions/32782447/gop-structure-via-ffmpeg

    if isinstance(video_path_or_frames_info, string_types):
        frames_info = read_frames_info(video_path_or_frames_info, only_first_s=only_first_s)
    else:
        frames_info = video_path_or_frames_info

    if isinstance(frames_info, pd.DataFrame):
        frame_pict_types = ''.join(frames_info['pict_type'])
    else:
        frame_pict_types = ''.join(frame_info['pict_type'] for frame_info in frames_info)

    return frame_pict_types, ['I' + ni + 'I' for ni in frame_pict_types.split('I') if len(ni)]


def seems_intra(video_path_or_frames_info, only_first_s=5):
    if isinstance(video_path_or_frames_info, string_types):
        frames_info = read_frames_info(video_path_or_frames_info, only_first_s=only_first_s)
    else:
        frames_info = video_path_or_frames_info

    frame_pict_types, _ = infer_gops(frames_info)

    if frame_pict_types == ('?' * len(frame_pict_types)):
        return None

    return ('I' * len(frame_pict_types)) == frame_pict_types
