from __future__ import print_function, division
import os
import socket

import delegator
import os.path as op
import glob

from itertools import product

import time

import numpy as np
import pandas as pd
import cv2

from rolx.benchmarks.jpegs.imgcodecs import JPEGOpenCVCodec, JPEGTurboCodec
from rolx.benchmarks.jpegs.envs import create_environments, env_name
from rolx.benchmarks.jpegs.imgutils import psnr, phash_compare
from rolx.benchmarks.jpegs.turbojpeg import TurboJPEG, TJPF, TJSAMP
from rolx.utils import resize, ensure_dir, ensure_writable_dir, timestr


# --- Quick utils

def honor_area_constraint(width, height, area_constraint):
    aspect_ratio = (width / height)
    height_new = int(np.sqrt(area_constraint / aspect_ratio))
    width_new = int(height_new * aspect_ratio)
    return width_new, height_new


class TimedIterator(object):

    def __init__(self, iterator_or_num_repetitions=5, name=None):
        super(TimedIterator, self).__init__()
        self.name = name
        self._times = []
        if isinstance(iterator_or_num_repetitions, int):
            iterator_or_num_repetitions = range(iterator_or_num_repetitions)
        self._iterator = iterator_or_num_repetitions

    def __iter__(self):
        for element in self._iterator:
            start = time.time()
            yield element
            self._times.append(time.time() - start)

    @property
    def times(self):
        return np.array(self._times)


# --- Benchmark loop

def benchmark(results_dir=None,
              image_files=None,
              repetitions=(0, 1),
              num_compressions=10,
              num_decompressions=10,
              save_images=False):

    if results_dir is None:
        results_dir = ensure_writable_dir(op.dirname(__file__), 'results')

    if image_files is None:
        image_files = sorted(glob.glob(op.join(op.dirname(__file__), 'images', '*')))

    result = {
        'host': socket.gethostname(),
        'env': os.environ.get('CONDA_DEFAULT_ENV'),
        'turbo_path': TurboJPEG().lib_path,
        'turbo_version': TurboJPEG().turbo_version,
    }

    # Packages provenance keeping
    packages = delegator.run('conda list --explicit')
    env_dir = ensure_dir(results_dir,
                         'host=%r' % result['host'],
                         'env=%r' % result['env'])
    with open(op.join(env_dir, 'environment-explicit.yaml'), 'wt') as writer:
        writer.write(packages.out)

    # --- Codec/decoder pairs

    encode_qualities = [80, 95, 99]  # , 97, 100
    optimizes = False, True

    opencv_encoders = [JPEGOpenCVCodec(
        encode_quality=encode_quality,
        encode_optimize=optimize,
        # Here we could play with several other params, for later
    ) for encode_quality, optimize in product(encode_qualities, optimizes)]
    opencv_decoders = [JPEGOpenCVCodec()]
    opencv_codecs = [(encoder, opencv_decoders) for encoder in opencv_encoders]

    subsamples = TJSAMP.YUV420,  # , TJSAMP.YUV422, TJSAMP.YUV440, TJSAMP.YUV444  TJSAMP.GRAY, maybe later
    pixel_formats = TJPF.BGR, TJPF.RGB
    fast_dcts = False, True
    fast_upsamples = False, True

    turbo_encoders = [JPEGTurboCodec(
        encode_quality=encode_quality,
        encode_pixel_format=TJPF.BGR,
        encode_jpeg_subsample=subsample,
        encode_fast_dct=fast_dct,
        encode_accurate_dct=False
    ) for encode_quality, subsample, fast_dct in product(encode_qualities, subsamples, fast_dcts)]
    turbo_decoders = [JPEGTurboCodec(
        decode_pixel_format=pixel_format,
        decode_fast_dct=fast_dct,
        decode_fast_upsample=fast_upsample,
    ) for pixel_format, fast_dct, fast_upsample in product(pixel_formats, fast_dcts, fast_upsamples)]
    turbo_codecs = [(encoder, turbo_decoders) for encoder in turbo_encoders]

    # noinspection PyTypeChecker
    all_codecs = opencv_codecs + turbo_codecs

    num_pixelss = 480 * 270, 960 * 540, 1920 * 1080  # , None => original image size

    for repetition, image_file in product(repetitions, image_files):

        original_image = cv2.imread(image_file)
        print('Original image: %s, repetition: %d' % (image_file, repetition))
        if original_image is None:
            print('Not an image: %r' % image_file)
            continue

        rep_result = result.copy()
        rep_result['original_image_file'] = op.basename(image_file)
        rep_result['original_image_shape'] = original_image.shape
        rep_result['repetition'] = repetition

        for num_pixels in num_pixelss:

            if num_pixels is None:
                image = original_image.copy()
            else:
                width, height = honor_area_constraint(original_image.shape[1],
                                                      original_image.shape[0],
                                                      num_pixels)
                image = resize(original_image, new_width=width, new_height=height, force_copy=True)

            np_result = rep_result.copy()
            np_result['num_pixels'] = num_pixels
            np_result['image_shape'] = image.shape
            np_result['uncompressed_size_bytes'] = len(image.data)

            # Run codecs in random order
            rng = np.random.RandomState(repetition)
            all_codecs_shuffled = list(all_codecs)
            rng.shuffle(all_codecs_shuffled)

            for encoder, decoders in all_codecs_shuffled:
                encoded, encoder_dir, write_result = benchmark_encoder(encoder, image,
                                                                       num_compressions,
                                                                       np_result, results_dir,
                                                                       save_images=save_images)
                for decoder in decoders:
                    benchmark_decoder(decoder, encoded, image,
                                      num_decompressions,
                                      write_result, encoder_dir,
                                      save_images=save_images)


def benchmark_encoder(encoder, image, num_compressions, result, results_dir, save_images=False):
    result = result.copy()
    result['encoder'] = encoder.what_encoder()
    # Compress
    encoded = encoder.encode(image)
    result['compressed_size_bytes'] = len(encoded)
    # Time compress?
    encoder_dir = ensure_dir(results_dir,
                             'host=%r' % result['host'],
                             'env=%r' % result['env'],
                             'image=%r' % result['original_image_file'],
                             'num_pixels=%r' % result['num_pixels'],
                             'repetition=%d' % result['repetition'],
                             result['encoder'].id())
    if save_images:
        cv2.imwrite(op.join(encoder_dir, 'original.png'), image)
    encoder_pickle = op.join(encoder_dir, 'encoder_result.pkl')
    if not op.isfile(encoder_pickle):
        # Time compress
        timer = TimedIterator(num_compressions)
        for _ in timer:
            encoder.encode(image)
        result['encode_time_s'] = timer.times  # can remove the first ones if too different
        result['encode_time_mean_s'] = timer.times.mean()
        result['encode_time_std_s'] = timer.times.std()
        result['encode_date'] = timestr()
        pd.to_pickle(result, encoder_pickle)
        # Ensure we can read back
        pd.read_pickle(encoder_pickle)
    else:
        # Read result
        print('Already done %r' % encoder_pickle)
        result = pd.read_pickle(encoder_pickle)
    return encoded, encoder_dir, result


def benchmark_decoder(decoder, encoded, image, num_decompressions, write_result, encoder_result_dir, save_images=False):
    # Decoder coordinates
    result = write_result.copy()
    result['decoder'] = decoder.what_decoder()
    # Already done?
    decoder_dir = ensure_writable_dir(encoder_result_dir, result['decoder'].id())
    decoder_pickle = op.join(decoder_dir, 'decoder_result.pkl')
    if op.isfile(decoder_pickle):
        print('Already done: %s' % decoder_pickle)
        return
    # Store buffer info
    result['encoded_info'] = TurboJPEG().info(encoded)
    # Decompress
    roundtripped = decoder.decode(encoded)
    if save_images:
        cv2.imwrite(op.join(decoder_dir, 'roundtripped.png'), roundtripped)
    assert roundtripped.shape[:2] == image.shape[:2]
    result['psnr'] = psnr(image, roundtripped)
    result['phash'] = phash_compare(image, roundtripped)
    # Time decompress
    timer = TimedIterator(num_decompressions)
    for _ in timer:
        decoder.decode(encoded)
    result['decode_time_s'] = timer.times  # can remove the first ones if too different
    result['decode_time_mean_s'] = timer.times.mean()
    result['decode_time_std_s'] = timer.times.std()
    result['decode_date'] = timestr()
    # Save
    pd.to_pickle(result, decoder_pickle)
    # Ensure we can read back
    pd.read_pickle(decoder_pickle)


# --- Conda mayhem

def benchmark_environments():
    return sorted(glob.glob(op.join(op.dirname(__file__), 'environments', '*.yaml')))


def create_all_environments(update='force'):
    environment_yamls = benchmark_environments()
    scripts = []
    for env_yaml in environment_yamls:
        create_environments([env_yaml], copy_from=None, update=update)
        name = env_name(env_yaml)
        scripts.append(benchmark_in_environment(name))
    runall_path = op.join(op.dirname(__file__), 'runall.sh')
    with open(runall_path, 'wt') as writer:
        writer.write('#!/bin/zsh\n' + '\n'.join('./%s' % op.basename(script) for script in scripts))
    os.chmod(runall_path, 0o770)


def benchmark_in_environment(name):
    script_path = op.join(op.dirname(__file__), name + '.sh')
    with open(op.join(op.dirname(__file__), name + '.sh'), 'wt') as writer:
        writer.write('#!/bin/zsh\n'
                     'source ~/.zshrc\n'
                     'conda-dev-on\n'
                     'source activate {env}\n'
                     'python -u benchmark.py benchmark | tee {env}.log'.format(env=name))
    os.chmod(script_path, 0o770)
    return script_path

    # Unfortunately it is not enough anymore to use the environment python executable
    # to run somethin under the environment...
    # And damn conda made me waste some hours already...
    # Will just run by hand


if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([create_all_environments, benchmark])
    parser.dispatch()
    print('Done')

# --- Notes
#
# Note that memory caches will be warm, but oh well

# TODO: some names need to be more consistent (like how we call turbo and pixel_format vs color_space)
