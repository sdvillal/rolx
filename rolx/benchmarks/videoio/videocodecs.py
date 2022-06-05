"""Benchmarking image providers."""
# Do not call this "codecs", it is one of these names that makes python gasps sometimes
from __future__ import print_function, division

from itertools import product

from rolx.sponsors.video.atloopbio import StoreWriter, StoreReader, FFMPEGWriter, OpenCVReader
from rolx.sponsors.video.jaggedio import jagged_jpeg_95, jagged_dummy
from rolx.sponsors.video.videocodecs import VEC

# --- Lossless (but beware of color space transformation)

LOSSLESS_CODECS = {

    # Some more could be added. UT and Lagarith (with "null" frames) are probably the most interesting ones?
    # Lagarith encoder is only windows though.

    # --- FFV1
    # FFV1 is better known for its encoding speed and good compression,
    # but its decoding speed is not really that competitive. So it is mainly
    # a good format for archival purposes.

    'ffv1_rice': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffv1',
        'codec_params': ['-level', '1', '-coder', 'rice', '-context', '1'],
        'container': '.avi'},

    'ffv13_rice_s4': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffv1',
        'codec_params': ['-level', '3', '-slices', '4', '-coder', 'rice', '-context', '1'],
        'container': '.avi'},

    'ffv13_rice_s12': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffv1',
        'codec_params': ['-level', '3', '-slices', '12', '-coder', 'rice', '-context', '1'],
        'container': '.avi'},

    'ffv1_rangedef': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffv1',
        'codec_params': ['-level', '1', '-coder', 'range_def', '-context', '1'],
        'container': '.avi'},

    'ffv13_rangedef_s4': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffv1',
        'codec_params': ['-level', '3', '-slices', '4', '-coder', 'range_def', '-context', '1'],
        'container': '.avi'},

    'ffv13_rangedef_s12': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffv1',
        'codec_params': ['-level', '3', '-slices', '12', '-coder', 'range_def', '-context', '1'],
        'container': '.avi'},

    # --- LIBX264 lossless

    'libx264-ll': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'libx264',
        'codec_params': ['-qp', '0'],
        'container': '.mp4'},

    'libx264-ll-uf': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'libx264',
        'codec_params': ['-preset', 'ultrafast', '-qp', '0'],
        'container': '.mp4'},

    'libx264-ll-sf': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'libx264',
        'codec_params': ['-preset', 'superfast', '-qp', '0'],
        'container': '.mp4'},

    'libx264-ll-medium': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'libx264',
        'codec_params': ['-preset', 'medium', '-qp', '0'],
        'container': '.mp4'},

    'libx264-ll-fastdecode': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'libx264',
        'codec_params': ['-qp', '0', '-tune', 'fastdecode'],
        'container': '.mp4'},

    'libx264-ll-uf-fastdecode': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'libx264',
        'codec_params': ['-preset', 'ultrafast', '-qp', '0', '-tune', 'fastdecode'],
        'container': '.mp4'},

    'libx264-ll-sf-fastdecode': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'libx264',
        'codec_params': ['-preset', 'superfast', '-qp', '0', '-tune', 'fastdecode'],
        'container': '.mp4'},

    'libx264-ll-medium-fastdecode': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'libx264',
        'codec_params': ['-preset', 'medium', '-qp', '0', '-tune', 'fastdecode'],
        'container': '.mp4'},

    # --- FFVHUFF

    'ffvhuff_plane_0_nd': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffvhuff',
        'codec_params': ['-pred', 'plane', '-context', '0', '-non_deterministic', '1'],
        'container': '.avi'},

    'ffvhuff_left_0_nd': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffvhuff',
        'codec_params': ['-pred', 'left', '-context', '0', '-non_deterministic', '1'],
        'container': '.avi'},

    'ffvhuff_median_0_nd': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffvhuff',
        'codec_params': ['-pred', 'median', '-context', '0', '-non_deterministic', '1'],
        'container': '.avi'},

    'ffvhuff_plane_1_nd': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffvhuff',
        'codec_params': ['-pred', 'plane', '-context', '1', '-non_deterministic', '1'],
        'container': '.avi'},

    'ffvhuff_left_1_nd': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffvhuff',
        'codec_params': ['-pred', 'left', '-context', '1', '-non_deterministic', '1'],
        'container': '.avi'},

    'ffvhuff_median_1_nd': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'ffvhuff',
        'codec_params': ['-pred', 'median', '-context', '1', '-non_deterministic', '1'],
        'container': '.avi'},

    # --- HUFFYUV

    # Let's represent HUFFYUV with ffvhuff
    # In our tests, the former is usually just better (faster and slightly higher compression ratio)
    #
    # 'huffyuv_plane': {
    #     'pix_fmt_i': 'bgr24',
    #     'videocodec': 'huffyuv',
    #     'codec_params': ['-pred', 'plane', '-context', '0'],
    #     'container': '.avi'},
    # 'huffyuv_left': {
    #     'pix_fmt_i': 'bgr24',
    #     'videocodec': 'huffyuv',
    #     'codec_params': ['-pred', 'left', '-context', '0'],
    #     'container': '.avi'}
    #

    # Preliminary tests show it does not seem really competitive,
    # we should use some other currently widely used codecs
    # (thinking of VP9). Do not bother with H.265. AV1 would be
    # cool when it comes (it is not that hard to use even
    # right now).

    # --- UTVideo

    'utvideo_none': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'utvideo',
        'codec_params': ['-pred', 'none'],
        'container': '.avi'},

    'utvideo_left': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'utvideo',
        'codec_params': ['-pred', 'left'],
        'container': '.avi'},

    'utvideo_gradient': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'utvideo',
        'codec_params': ['-pred', 'gradient'],
        'container': '.avi'},

    'utvideo_median': {
        'pix_fmt_i': 'bgr24',
        'videocodec': 'utvideo',
        'codec_params': ['-pred', 'median'],
        'container': '.avi'},
}

# Of course, not lossless
LOSSLESS_PIX_FMTS = 'yuv420p',

LOSSLESS_CONFIGS = []
for pix_fmt in LOSSLESS_PIX_FMTS:
    for name, config in LOSSLESS_CODECS.items():
        LOSSLESS_CONFIGS.append(VEC(name=name + '_%s' % pix_fmt,
                                    container=config['container'],
                                    codec=config['videocodec'],
                                    params=config['codec_params'] + ['-pix_fmt', pix_fmt],
                                    is_lossless=True,
                                    is_intra_only=None,
                                    writer=FFMPEGWriter, reader=OpenCVReader))

#
# --- MJPEG
#
# Talking about a bunch of JPGs, the obvious choice is to use MJPEG
# ffmpeg -h encoder=mjpeg
#
# https://en.wikipedia.org/wiki/Motion_JPEG
# https://aeon.nervanasys.com/index.html/provider_video.html
# https://stackoverflow.com/questions/32147805/ffmpeg-generate-higher-quality-images-for-mjpeg-encoding
#
# N.B. check which decoder we use when we go via opencv. Force FFMPEG or force internal (uses turbo?)
# N.B. at the moment we are a bit struggling witht the quality of these clips, so MJPEG deserves
# a good day research... (probably it is not a good idea to look at these with VLC)
#

VALID_MJPEG_PIX_FMTs = 'yuvj420p', 'yuvj422p', 'yuvj444p'


def mjpeg_encoder_config(quality=2, optimal_huffman=False, pix_fmt='yuvj420p',
                         writer=FFMPEGWriter, reader=OpenCVReader):

    # this can easily be inferred using ffmpeg CL
    if pix_fmt not in VALID_MJPEG_PIX_FMTs:
        raise ValueError('pix_fmt %r is not valid; options are %r' % (pix_fmt, VALID_MJPEG_PIX_FMTs))

    name = ['mjpeg', str(quality)]
    if optimal_huffman:
        name .append('h')
    name.append(pix_fmt)
    name = '_'.join(name)

    params = ['-q:v', str(quality),
              '-huffman', 'huffman' if optimal_huffman else 'default',
              '-pix_fmt', pix_fmt]

    return VEC(name=name, container='avi', codec='mjpeg', params=params,
               is_lossless=False, is_intra_only=True,
               writer=writer, reader=reader)


# 1 => low loss mjpeg; see e.g.
# https://superuser.com/questions/347433/how-to-create-an-uncompressed-avi-from-a-series-of-1000s-of-png-images-using-ff
mjpeg_qualities = 1, 2, 10       # the lower the better quality, fractions allowed
mjpeg_pixex_fmts = 'yuvj420p',   # VALID_MJPEG_PIX_FMTs
mjpeg_optimal_huffmans = False,  # , True

MJPEG_CONFIGS = []
for q, h, pf in product(mjpeg_qualities, mjpeg_optimal_huffmans, mjpeg_pixex_fmts):
    config = mjpeg_encoder_config(q, h, pf)
    assert not config.is_lossless
    assert config.is_intra_only
    MJPEG_CONFIGS.append(config)


# --- Store configs

STORE_CONFIGS = [
    VEC(name='exploded-jpg', container='jpg', codec='jpg', params=(),
        pix_fmt='yuv420',
        is_lossless=True, is_intra_only=True,
        writer=StoreWriter, reader=StoreReader),
    VEC(name='exploded-npy', container='npy', codec='npy', params=(),
        pix_fmt='bgr24',
        is_lossless=True, is_intra_only=False,
        writer=StoreWriter, reader=StoreReader),
    # VEC(name='exploded-png', container='png', codec='png', params=(),
    #     pix_fmt='rgb24',
    #     is_lossless=True, is_intra_only=True,
    #     writer=StoreWriter, reader=StoreReader),
]


# --- Jagged configs

JAGGED_CONFIGS = [jagged_dummy, jagged_jpeg_95]

# --- All together

BENCHMARK_CODECS = tuple(STORE_CONFIGS + LOSSLESS_CONFIGS)  # + MJPEG_CONFIGS + JAGGED_CONFIGS


if __name__ == '__main__':
    print('There are %d configs' % len(BENCHMARK_CODECS))

    # Which chroma subsampling does opencv use by default?
    #  http://pillow.readthedocs.io/en/4.0.x/PIL.html#subsampling

    import cv2
    import numpy as np
    from PIL import JpegImagePlugin, Image
    import os

    cv2.imwrite('/tmp/image.jpg', np.zeros((100, 100, 3), dtype=np.uint8))
    print(JpegImagePlugin.get_sampling(Image.open('/tmp/image.jpg')))  # => 2
    os.system('identify -verbose /tmp/image.jpg | grep samp')          # => jpeg:sampling-factor: 2x2,1x1,1x1
    # conclusion: it is 4:2:0 (YUV420)
