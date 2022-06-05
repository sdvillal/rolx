# coding=utf-8
"""Benchmarking image providers."""
from __future__ import print_function, division

import logging
import os
import os.path as op
import shutil
import tempfile
import traceback
from functools import partial
from itertools import product

import humanize
import pandas as pd
import tqdm

from rolx.benchmarks.videoio.hardware import BENCHMARK_DATA_LOCATIONS
from rolx.benchmarks.videoio.videocodecs import BENCHMARK_CODECS
from rolx.benchmarks.videoio.videos import BENCHMARK_VIDEOS
from rolx.benchmarks.videoio.workloads import BENCHMARK_WORKLOADS
from rolx.sponsors.video.atloopbio import (FFMPEGWatchCallback)
from rolx.sponsors.video.io import SyncAfterClose, WatchCallback, BarcodeCallback
from rolx.utils import ensure_writable_dir, du, timestr, force_new_line
from rolx.watches import Rusager

_log = logging.getLogger('rolx')

BENCHMARKS_ROOT = op.expanduser('~/videoio-benchmarks')
BENCHMARKS_ROOT_DEBUG = op.expanduser('~/videoio-benchmarks-debug')


class BenchmarkSuite(object):

    def __init__(self,
                 root=BENCHMARKS_ROOT,
                 videos=BENCHMARK_VIDEOS,
                 codecs=BENCHMARK_CODECS,
                 locations=BENCHMARK_DATA_LOCATIONS,
                 workloads=BENCHMARK_WORKLOADS,
                 repetitions=(0, 1, 2),
                 log=None):
        super(BenchmarkSuite, self).__init__()

        # Logging
        self.log = log if log is not None else _log

        # Data & results location
        self.root = ensure_writable_dir(root)
        self.source_videos_root = ensure_writable_dir(self.root, 'videos')
        self.results_root = ensure_writable_dir(self.root, 'results')

        # Benchmark suite
        self.videos = videos
        self.codecs = codecs
        self.locations = locations
        self.workloads = workloads
        self.repetitions = repetitions

    def run_all(self, force=False, debug=False):
        all_jobs = list(product(self.repetitions,
                                self.videos,
                                self.locations,
                                self.workloads,
                                self.codecs))
        for repetition, video, location, workload, codec in tqdm.tqdm(all_jobs, unit='experiment'):
            force_new_line()
            self.run_one(video=video,
                         codec=codec,
                         location=location,
                         workload=workload,
                         repetition=repetition,
                         force=force,
                         debug=debug)

    def run_one(self, video, codec, location, workload, repetition, force=False, debug=False):

        # Check if we can find the location
        if not location.is_available():
            self.log.info('location %s not available, skipping' % location.name)
            return
        self.log.info('location: %s' % location.name)

        # Check if we have the video
        try:
            video.cache_video_to(self.source_videos_root)
        except Exception as ex:
            self.log.info('video %r not available (%s), skipping' % (video, str(ex)))
            return
        self.log.info('original video: %s' % video.name)

        # Codec reader and writers
        writer, reader = codec.meta['writer'], codec.meta['reader']

        # Result template
        result = {
            'video': video,
            'codec': codec,
            'location': location,
            'repetition': repetition
        }

        # Results directory
        result_dir = ensure_writable_dir(
            self.results_root,
            'clip=%s' % video.name,
            'codec=%s' % codec.name,
            'location=%s' % location.name,
            'repetition=%d' % repetition
        )

        # Place to write and read data from
        workspace_dir = tempfile.mkdtemp(prefix='video-benchmark',
                                         dir=location.ensure_under('videoio-benchmarks-deleteme'))
        self.log.info('Working on temporary directory %s' % workspace_dir)

        try:
            # Many read workloads are associated with a write workload
            write_workload, read_workloads = workload

            # Write workload
            write_result_dir, fixed_result_values = self.run_write_workload(workload=write_workload,
                                                                            writer=writer,
                                                                            video=video, codec=codec,
                                                                            repetition=repetition,
                                                                            workspace_dir=workspace_dir,
                                                                            result=result, result_dir=result_dir,
                                                                            force=force)

            if write_result_dir:

                # Now this is fixed, we still want to have it...
                result.update(fixed_result_values)

                # Read workloads
                for read_workload in tqdm.tqdm(read_workloads, unit='workload'):
                    force_new_line()
                    self.run_read_workload(workload=read_workload,
                                           reader=reader,
                                           repetition=repetition,
                                           workspace_dir=workspace_dir,
                                           result=result, result_dir=write_result_dir,
                                           force=force)
        finally:
            # Wipe temporary workspace dir
            if not debug:
                shutil.rmtree(workspace_dir)

    def run_write_workload(self,
                           workload,
                           writer,
                           video, codec,
                           repetition,
                           workspace_dir,
                           result, result_dir,
                           force):

        result = result.copy()

        result['total_watch'] = Rusager()
        result['total_watch'].tic()

        result_dir = ensure_writable_dir(
            result_dir,
            'write_workload=%s' % workload.name,
        )

        self.log.info('WRITING %s' % result_dir)

        if force:
            shutil.rmtree(result_dir, ignore_errors=True)
            ensure_writable_dir(result_dir)

        watch, syncer = callbacks = (FFMPEGWatchCallback(), SyncAfterClose())
        writer = partial(writer, codec=codec, callbacks=callbacks)
        result['type'] = 'write'
        result['success'] = False
        result['workload'] = workload
        result['watch'] = watch
        result['syncer'] = syncer

        # Do the job
        # noinspection PyUnusedLocal
        try:
            workload(original_path=video.cache_video_to(self.source_videos_root),
                     writer=writer,
                     dest_path=workspace_dir,
                     start=video.start,
                     stop=video.start + video.num_frames * video.step,
                     step=video.step,
                     seed=repetition)
            result['disk_usage_bytes'] = du(workspace_dir)
            result['success'] = True
        except Exception as ex:
            tb = traceback.format_exc()
            self.log.error(tb)
            with open(op.join(result_dir, 'stacktrace.txt'), 'wt') as writer:
                writer.write(tb)

        # This needs to be done before pickling, obviously; the rest is peanuts anyway
        result['total_watch'].toc()

        # Pickle the result - hurry is a devil invention
        pd.to_pickle(result, op.join(result_dir, 'result.pkl'))
        # ensure we can reread
        pd.read_pickle(op.join(result_dir, 'result.pkl'))

        # Wrap-up
        if not result['success']:
            # We have failed, get rid of success marker
            if op.isfile(op.join(result_dir, 'DONE.txt')):
                os.unlink(op.join(result_dir, 'DONE.txt'))
            self.log.info('Failed %s, will not run read workloads' % result_dir)
            return None, None
        else:
            # We have succeeded, so get rid of error messages
            if op.isfile(op.join(result_dir, 'stacktrace.txt')):
                os.unlink(op.join(result_dir, 'stacktrace.txt'))

            with open(op.join(result_dir, 'DONE.txt'), 'wt') as writer:
                writer.write(timestr())

            self.log.info('DONE writing %s' % result_dir)
            self.log.info('Disk space: %s' % humanize.naturalsize(result['disk_usage_bytes']))
            self.log.info('Writer utime: %g' % watch.user_time)
            self.log.info('Writer stime: %g' % watch.system_time)
            self.log.info('Writer wtime: %g' % watch.watch.wall_time)
            self.log.info('Sync step time: %g' % syncer.watch.total)
            self.log.info('Total time: %g' % result['total_watch'].wall_time)

            values_for_reader = {
                'clip_actual_num_frames': watch.watch.num_units,
                'clip_new_shape': workload.config['new_shape'],
                'disk_usage_bytes': result['disk_usage_bytes'],
            }

            return result_dir, values_for_reader

    @staticmethod
    def must_ignore_result(result, min_batch_size_for_non_sequential=0):

        # Ignore logic for read workloads
        if result['type'] == 'read':
            workload = result['workload']
            codec = result['codec']
            sequential = workload.config['sequential']
            batch_size = workload.config['batch_size']

            if not sequential and batch_size < min_batch_size_for_non_sequential:
                # We know that seeking is usually very slow for video readers
                # unless amortized by sequential reading; by default we still
                # let them run for batch_size 0 (so we estimate how slow is to seek).
                # But we could simply just not let, for example, not-sequential
                # and batch_size=2.
                return True, 'We do know random and small batch size is slow for video codecs'

            if (codec.name in ['exploded-npy', 'exploded-png', 'exploded-jpg'] and
                    workload.config['num_threads'] > 1):
                #
                # Multithreading as implemented for stores is unfortunately slow.
                # The reason is probably that:
                #   - we are opening a new store on each worker
                #   - we are creating the workers on each call to get
                # It won't represent how a proper implementation would scale
                # (which we probably need for proper comparison).
                # Either bring fisherman multithreaded directory reader or, better,
                # use something smarter (not exploded files ala TFRecord / GulpIO / RecordIO / jagged).
                # It could have been really interesting to see how this fares when saturating I/O
                # or when things are in disk.
                #
                return True, 'Multithreading is poorly implemented for stores'

            if codec.codec in ['mjpeg'] and workload.config['num_threads'] > 1:
                #
                # MJPEG experiments are a bit broken too, in that not only setting
                # the number of threads to something other than 1 is ignored,
                # if we would let it run it would just fail with some nasty
                # errors (let it pass with ignore setting and you will get
                # the error). In any case, we need to get tighter control
                # over the MJPEG decoder (e.g. ensure we either use ffmpeg
                # or opencv internal impl.).
                #
                return True, 'MJPEG and num_threads > 1 is not working at the moment'

            if codec.container in ['jagged'] and workload.config['num_threads'] > 1:
                # It should be simple to compress / decompress on parallel, but no time this afternoon
                return True, 'jagged container and num_threads > 1 is not working at the moment'

        return False, None

    def run_read_workload(self, workload, reader, repetition, workspace_dir, result, result_dir, force):

        result = result.copy()

        result['total_watch'] = Rusager()
        result['total_watch'].tic()

        # Path to store the result
        result_dir = ensure_writable_dir(
            result_dir,
            'read_workload=%s' % workload.name,
        )

        self.log.info('READING %s' % result_dir)

        # Do we need to recompute?
        if force:
            shutil.rmtree(result_dir, ignore_errors=True)
            ensure_writable_dir(result_dir)
        if op.isfile(op.join(result_dir, 'DONE.txt')):
            self.log.info('%s already done, skipping...' % result_dir)
            return

        # Prepare watches
        checker, watch = callbacks = BarcodeCallback(), WatchCallback()
        reader = partial(reader, callbacks=callbacks)

        # Update result dict
        result['success'] = False
        result['type'] = 'read'
        result['workload'] = workload
        result['checker'] = checker
        result['watch'] = watch

        # Is this something we decide not too run?
        # (e.g. because we do know it is very slow)
        must_ignore, ignore_reason = self.must_ignore_result(result)
        if must_ignore:
            self.log.info('forcefully ignored %s' % result_dir)
            self.log.info('reason: %s' % ignore_reason)
            shutil.rmtree(result_dir, ignore_errors=True)
            return

        # Do the job
        # noinspection PyUnusedLocal
        try:
            workload(reader=reader, path=workspace_dir, seed=repetition)
            result['success'] = True
        except Exception as ex:
            tb = traceback.format_exc()
            self.log.error(tb)
            with open(op.join(result_dir, 'stacktrace.txt'), 'wt') as writer:
                writer.write(tb)

        # This needs to be done before pickling, obviously; the rest is peanuts anyway
        result['total_watch'].toc()

        # Pickle the result - hurry is a devil invention
        pd.to_pickle(result, op.join(result_dir, 'result.pkl'))
        # ensure we can reread
        pd.read_pickle(op.join(result_dir, 'result.pkl'))

        # Wrap-up
        if not result['success']:
            # We have failed, get rid of success marker
            if op.isfile(op.join(result_dir, 'DONE.txt')):
                os.unlink(op.join(result_dir, 'DONE.txt'))
                self.log.info('Failed %s' % result_dir)
        else:
            # We have succeeded, so get rid of error messages
            if op.isfile(op.join(result_dir, 'stacktrace.txt')):
                os.unlink(op.join(result_dir, 'stacktrace.txt'))

            with open(op.join(result_dir, 'DONE.txt'), 'wt') as writer:
                writer.write(timestr())

            watch = watch.watch
            self.log.info('Reader utime: %g' % watch.user_time)
            self.log.info('Reader stime: %g' % watch.system_time)
            self.log.info('Reader wtime: %g' % watch.wall_time)
            self.log.info('Reader FPU: %g' % watch.ups_ru)
            self.log.info('Reader FPS: %g' % watch.ups_wall)
            self.log.info('Disk space: %s' % humanize.naturalsize(result['disk_usage_bytes']))
            self.log.info('PCT Correct:  %g' % checker.pct_correct)
            self.log.info('Total time: %g' % result['total_watch'].wall_time)
            self.log.info('DONE reading %s' % result_dir)
            # log.info(purger.purge_timer.n)


def run_benchmark(debug=False, repetition=0):
    if debug:
        bs = BenchmarkSuite(root=BENCHMARKS_ROOT_DEBUG, repetitions=(repetition,))
        bs.run_all(force=True, debug=True)
    else:
        bs = BenchmarkSuite(root=BENCHMARKS_ROOT, repetitions=(repetition,))
        bs.run_all(force=False, debug=False)
    _log.info('ALL DONE, CONGRATS!')


if __name__ == '__main__':
    import argh
    logging.basicConfig(level=logging.INFO)
    argh.dispatch_command(run_benchmark)


# TODO: create Result, WriteResult, ReadResult classes
# TODO: split Clip and Video
# TODO: save results in a more durable format
# TODO: proper calibration of tests + finish early if obviously too slow or test more if too fast
#       actually it could be great to limit each result computation to x seconds, so bring back
#       Timeout from sandbox

# ---- Notes

# Are we using the fastest codepath for outputting BGR?
#   https://github.com/libjpeg-turbo/libjpeg-turbo/issues/65
# That is, is cv2.imread asking the underlying jpeg library to directly output BGR?

#
# yuv420p exact format:
#  https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV420p_(and_Y%E2%80%B2V12_or_YV12)_to_RGB888_conversion
#

#
# Alternatively to all these tictocs, do not use callback / poll continuously.
# Instead, just call rusage at the end (possibly of a process
# spawned only for the task at hand). Something like:
#   watch.tic()
#   do_all_the_work()
#   watch.toc()
#   if isinstance(writer, FFMPEGWriter):
#       total += writer.process.cpu_times().user
# This have advantages and disadvantages. From the continuous polling I like the most
# that we really localize measurements, I like the least the amount of extra work and
# therefore slowness and higher measurement error we introduce.
#

#
# From sandbox/image_read_benchmark.py
# --------------------------
# So what takes longer is jpeg decoding...
# Very manually (change code & run!) and unscientifically, quick timings
#
# WITH PNGS
#   read_decompress: 81.28 (8.13 +/- 0.06 s/op, 10 operations)
#   read_decompress_4threads: 24.31 (2.43 +/- 0.05 s/op, 10 operations)
#   Size: ~1.5GiB (I guess using RLE without interlacing? default from imagemagick)
#
# WITH libjpeg 9 + Multiprocessing
#   read_decompress: 15.17 (1.52 +/- 0.05 s/op, 10 operations)
#   read_decompress_4threads: 10.25 (1.03 +/- 0.04 s/op, 10 operations)
#   Size: ~179MiB (what are the compression settings?)
#
# With TurboJPEG + Multiprocessing
#   read_decompress: 7.48 (0.75 +/- 0.02 s/op, 10 operations)
#   read_decompress_4threads: 9.07 (0.91 +/- 0.01 s/op, 10 operations)
# With TurboJPEG + Threading
#   read_decompress: 7.24 (0.72 +/- 0.02 s/op, 10 operations)
#   read_decompress_4threads: 2.94 (0.29 +/- 0.06 s/op, 10 operations)
#   Size: ~179MiB (what are the compression settings?)
#
# Very Naive H5 File + h5py uncompressed + threading (multiprocessing is just useless)
#   read_decompress: 1.29 (0.13 +/- 0.03 s/op, 10 operations)
#   read_decompress_4threads: 2.38 (0.24 +/- 0.01 s/op, 10 operations)
#   Size: ~8GiB (much larger than plain png with, I guess, RLE-interlacing)
#   So dumb fast compression is just going to make it smaller and faster
#
# This round winner: vanilla HDF5
#
# Next contenders:
#  Jagged + lossless compression
#  Jagged + segments together + clever arrangement + compression
#    (e.g. same pixel local across small batch of frames + fast l4hc compression)
#  Jagged + saved jpeg data
#  Regular Video Compression
#  Aeon
#
# Note: I tried a little bit jagged like stuff and it was not as fast as promising
#       Apparently I have lost these tests (probably it was just a 10 lines thing)
#
# --------------------------
#
# On a second round, I checked how codecs fared in a "real workload" scenario.
# That is, I let trained with a variety of codecs and measured real read speed
# (plus ensured everything was correct).
# It is in that scenario that ffvhuff worked best *with Simon data*.
# Most likely a combination of proper I/O and decoder complexities in the concrete
# hardware and dataset size settings. An important note, the tests were always
# performed at full-res. It is a long time we do not resize online, so these
# new tests are most relevant. In that benchmark there was not bunch of mjpeg
# of any kind other than exploded jpegs, and it is very likely I was using
# jpeg 9 instead of turbo.
# --------------------------
#
# Another thing we checked, some quick sequential vs minibatching (+OCR ans seekability)
# From sandbox/random_mp4_access (small seekability test over standard MP4 Simon was recoding)
#
# Sequential speed with pims + conda-forge ffmpeg: 246.76 fps
# Minibatch speed with pims + conda-forge ffmpeg: 85.90 fps
# OCR speed: 53.41 fps
# OCR accuracy (random frame access): 0.99
#
