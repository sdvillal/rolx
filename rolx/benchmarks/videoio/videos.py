# coding=utf-8
"""Videos for the benchmark suite."""
from __future__ import print_function

import os
import os.path as op
import shutil
from subprocess import check_call

from ruamel import yaml

try:
    from pytube import YouTube
except ImportError:
    YouTube = None

from rolx.sponsors.video.atloopbio import OpenCVReader


# noinspection PyClassHasNoInit
class RESOLUTIONS:
    # https://en.wikipedia.org/wiki/720p
    HD = 1280, 720
    # 1080P (https://en.wikipedia.org/wiki/1080p)
    FHD = 1920, 1080
    # 4Ks (https://en.wikipedia.org/wiki/4K_resolution)
    UHD1 = 3840, 2160
    UW4K = 3840, 1600
    DCI4K = 4096, 2160


# noinspection PyClassHasNoInit
class LICENSES:
    YOUTUBE_STANDARD = 'youtube-standard-license'
    CREATIVE_COMMONS = 'creative-commons-license'
    WITH_PERMISSION = 'with-owner-permission'


class Video(object):

    # TODO: split this in two:
    #    Video
    #    Clip (start, stop, step subset of frames of video)

    def __init__(self,
                 name, url,
                 motion='high',
                 constant_bg=False,
                 resolution=RESOLUTIONS.FHD,
                 license=LICENSES.CREATIVE_COMMONS,
                 reader=OpenCVReader,
                 start=0, step=1, num_frames=1500,
                 container='mp4',
                 **kwargs):
        super(Video, self).__init__()
        self.video_id = name
        self.name = name + '-%d-%d-%d' % (start, step, num_frames)
        self.url = url
        self.motion = motion
        self.constant_bg = constant_bg
        self.resolution = resolution
        self.license = license
        self.reader = reader
        # Write step
        # How important is that there is frame continuity, that is, that there is temporal structure?
        self.start, self.step, self.num_frames = start, step, num_frames
        self.container = container
        self.meta = kwargs

    def cache_video_to(self, dest_dir, symlink_if_possible=True, overwrite=False, use_pytube=False):

        dest_dir = op.join(dest_dir, self.video_id)
        fn = self.video_id + '.' + self.container
        dest_file = op.join(dest_dir, fn)

        if op.exists(dest_file):
            if overwrite:
                os.unlink(dest_file)
            else:
                return dest_file

        if self.in_youtube():
            # Unfortunately this already fails with some of our videos
            if use_pytube and YouTube is not None:
                (YouTube(self.url).
                 streams.
                 filter(adaptive=True, file_extension=self.container).
                 first().
                 download(output_path=ensure_dir(dest_dir), self=self.video_id))
            else:
                check_call([
                    'youtube-dl',
                    '-f', 'bestvideo[ext=%s]' % self.container,
                    '-o', dest_file,
                    self.url
                ])
        elif op.isfile(self.url):
            ensure_dir(dest_dir)
            if symlink_if_possible:
                os.symlink(self.url, dest_file)
            else:
                shutil.copy(dest_dir, dest_file)
        else:
            raise Exception('Cannot find video %s (should be at %r)' % (self.video_id, self.url))

        with open(op.join(dest_dir, 'info.yaml'), 'wt') as writer:
            info = self.__dict__.copy()
            del info['reader']
            del info['start']
            del info['step']
            del info['num_frames']
            del info['name']
            yaml.safe_dump(info, writer)

        return dest_file

    def in_youtube(self):
        return self.url.startswith('https://www.youtube.com')


# --- Drones

drone_niger = Video(name='drone-niger',
                    url='https://www.youtube.com/watch?v=amUqnEk4DuQ',
                    motion='high',
                    constant_bg=False,
                    resolution=RESOLUTIONS.FHD,
                    start=0, step=2, num_frames=1200)

nz_sheeps = Video(name='nz-sheeps',
                  url='https://www.youtube.com/watch?v=D8mXL2JapWM',
                  motion='normal',
                  constant_bg=False,
                  license=LICENSES.YOUTUBE_STANDARD,
                  resolution=RESOLUTIONS.UHD1,
                  start=400, step=3, num_frames=1200)

# --- Worms

c_elegans = Video(name='c-elegans',
                  url='https://www.youtube.com/watch?v=GgZHziFWR7M',
                  motion='low',
                  constant_bg=True,
                  license=LICENSES.YOUTUBE_STANDARD,
                  resolution=RESOLUTIONS.HD,
                  start=2000, step=5, num_frames=1200)  # 19614 frames originally

# --- Under water

fish_mirror = Video(name='fish-mirror',
                    url='https://www.youtube.com/watch?v=AefrFSdvAb0',
                    motion='high',
                    constant_bg=True,
                    resolution=RESOLUTIONS.FHD,
                    start=0, step=1, num_frames=1000)

red_sea = Video(name='red-sea',
                url='https://www.youtube.com/watch?v=bNucJgetMjE',
                motion='normal',
                constant_bg=False,  # well, sometimes
                license=LICENSES.YOUTUBE_STANDARD,
                resolution=RESOLUTIONS.HD,
                start=0, step=4, num_frames=1200)  # 9815 frames originally


# --- Manakins

manakins = Video(name='manakins',
                 url='https://www.youtube.com/watch?v=8q9_QvviVUY',
                 motion='low',
                 constant_bg=True,
                 license=LICENSES.YOUTUBE_STANDARD,
                 resolution=RESOLUTIONS.HD,
                 start=0, step=1, num_frames=600)  # we overshoot, this will only read 435 frames


# --- Some pretty 4K CC mix

nature = Video(name='nature',
               url='https://www.youtube.com/watch?v=6xwozDohDUs',
               motion='normal',
               constant_bg=False,
               license=LICENSES.CREATIVE_COMMONS,
               resolution=RESOLUTIONS.UHD1,
               start=1000, step=4, num_frames=600)


BENCHMARK_VIDEOS = (
    drone_niger,
    # nz_sheeps,
    c_elegans,
    fish_mirror,
    # red_sea,
    manakins,
    nature,
)

if __name__ == '__main__':

    from rolx.utils import ensure_dir, sanitize_start_stop, timestr
    from rolx.benchmarks.videoio.workloads import BENCHMARK_WRITE_WORKLOADS
    from rolx.benchmarks.videoio.benchmark import BENCHMARKS_ROOT
    from rolx.benchmarks.videoio.videocodecs import BENCHMARK_CODECS
    from rolx.sponsors.video.ffmpeg import seems_intra, read_frames_info, infer_gops

    for clip in BENCHMARK_VIDEOS:
        # Download video if needed
        video_path = clip.cache_video_to(op.join(BENCHMARKS_ROOT, 'videos'))
        # Check actual length of clip (FIXME: clip id should be informed by this)
        with OpenCVReader(video_path) as reader:
            start, stop = sanitize_start_stop(reader,
                                              clip.start,
                                              clip.start + clip.step * clip.num_frames)
        print('%s actually has %d frames' % (clip.name, len(range(start, stop, clip.step))))
        # Select a smaller clip to write and get frame statistics from
        frames_to_write = 100, 300
        start = start + 100
        num_frames_to_write = 200
        stop = range(start, stop, clip.step)[:num_frames_to_write][-1] + 1
        # Write the thing
        redo = False
        for codec in BENCHMARK_CODECS:
            for write_workload in BENCHMARK_WRITE_WORKLOADS:
                if codec.codec != 'mjpeg' and codec.container not in ['mp4']:
                    continue
                if write_workload.config['new_shape'] != (480, 270, 3):
                    continue
                dest_dir = op.join(op.dirname(video_path), 'clips', codec.name, write_workload.name)
                if op.isfile(op.join(dest_dir, 'DONE.txt')) and not redo:
                    continue
                # print('\t', write_workload.name, codec.name)
                write_workload(
                    original_path=video_path,
                    writer=codec.writer(),
                    dest_path=dest_dir,
                    start=start,
                    stop=stop,
                    step=clip.step,
                    new_shape=write_workload.config['new_shape']
                )
                if codec.container in ['mp4', 'avi']:
                    reader = OpenCVReader(dest_dir)
                    frames_info = read_frames_info(reader.video_path)
                    frame_pict_types, gops = infer_gops(frames_info)
                    all_frames_are_intra = seems_intra(frames_info)
                    print('\t', codec.name, len(gops), all_frames_are_intra, frame_pict_types)
                with open(op.join(dest_dir, 'DONE.txt'), 'wt') as writer:
                    writer.write(timestr())
