# coding=utf-8
"""Consolidation and analysis of the benchmark results."""
from __future__ import print_function, division

import logging
import os
from itertools import product

import humanize
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata

from rolx.benchmarks.videoio.benchmark import BENCHMARKS_ROOT
from rolx.sponsors.video.atloopbio import FFMPEGWatchCallback
from rolx.utils import ensure_dir
from rolx.watches import Rusager

_log = logging.getLogger('rolx')


DEFAULT_RESULTS_DIR = op.join(BENCHMARKS_ROOT, 'results')


# --- Utils

def compression_ratio(result):
    # https://en.wikipedia.org/wiki/Data_compression_ratio
    uncompressed_size = np.prod(result['clip_new_shape']) * result['clip_actual_num_frames']
    compressed_size = result['du_bytes']
    return uncompressed_size / compressed_size


def normalize_metrics(df,
                      groupers=('video_name', 'clip_resolution', 'location_name'),
                      metrics=('fps_wall', 'fps_ru', 'space_savings'),
                      prefix='normalized_',
                      straw_man_query='codec_codec == "jpg"',
                      ignore_no_straw_man=False,
                      center_to_0=False):

    # Pretty sure there is a more pandas way to do this...
    normalized_dfs = []
    for group_id, normalized_df in df.groupby(groupers):
        normalized_df = normalized_df.copy()
        straw_man = normalized_df.query(straw_man_query)
        if len(straw_man) != 1:
            if len(straw_man) == 0:
                if ignore_no_straw_man:
                    continue
                raise Exception('No straw man %r found for group %r' % (straw_man_query, group_id))
            else:
                raise Exception('Multiple straw man %r for group %r; please add more groupers' %
                                (straw_man_query, group_id))
        for metric in metrics:
            normalized_df[prefix + metric] = normalized_df[metric]
            if center_to_0:
                normalized_df[prefix + metric] -= straw_man.iloc[0][metric]
            normalized_df[prefix + metric] /= straw_man.iloc[0][metric]
        normalized_dfs.append(normalized_df)

    normalized_df = pd.concat(normalized_dfs)
    if not ignore_no_straw_man:
        assert len(normalized_df) == len(df)

    # Now speed is normalized against the straw man and we can proceed to aggregate
    # print(normalized_df.groupby('codec_name')[[speed_metric, disk_metric]].agg([np.mean, np.std]))

    return normalized_df


def rank_group_metrics(df,
                       groupers=('video_name', 'clip_resolution', 'location_name'),
                       metrics=('fps_wall', 'fps_ru', 'space_savings'),
                       smaller_ranks_higher=False,
                       prefix='ranked_',
                       method='average'):

    ranked_dfs = []

    if isinstance(smaller_ranks_higher, bool):
        smaller_ranks_higher = [smaller_ranks_higher] * len(metrics)
    elif isinstance(smaller_ranks_higher, dict):
        smaller_ranks_higher = [smaller_ranks_higher[metric] for metric in metrics]
    if not len(smaller_ranks_higher) == len(metrics):
        raise ValueError('does not know how to rank all metrics, '
                         'please provide a complete `smaller_ranks_higher` spec')
    smaller_ranks_higher = [1 if srh else -1 for srh in smaller_ranks_higher]

    for _, ranked_df in df.groupby(groupers):
        ranked_df = ranked_df.copy()
        for srh, metric in zip(smaller_ranks_higher, metrics):
            ranked_df[prefix + metric] = rankdata(srh * ranked_df[metric], method=method)
        ranked_dfs.append(ranked_df)

    ranked_df = pd.concat(ranked_dfs)
    assert len(ranked_df) == len(df)

    return ranked_df


# --- Some column and value semantics

# What identifies a clip
CLIP_ID = ['clip_name', 'clip_new_shape']

RESOLUTIONS = {
    'small': (480, 270, 3),
    'medium': (960, 540, 3),
    'high': (1920, 1080, 3),
}

# What identifies a location
LOCATION_ID = ['location_name']

# What identifies a codec (N.B. codec_name includes the pixel format)
CODEC_ID = ['codec_name']

# What identifies a workload
WORKLOAD_ID = [
    # Something like:
    # "read(batch_size=1,evict_rate=0,initial_fs_state=None,num_threads=1,seek='always',sequential=False,step=1)"
    'workload_name'
]

WORKLOAD_ID_ATTRIBUTES = [
    # True => Access fully sequentially; False => Access sequentially in a batch (with random batch order)
    'workload_sequential',
    # Distance between different frames in a batch (1 ATM)
    'workload_step',
    # Batch size (1, 32 ATM); how many consecutive frames to read for each read operation
    'workload_batch_size',
    # 'warm': all has been recently touched, 'cold': all has been evicted, None: do nothing
    'workload_initial_fs_state',
    # probability to evict or touch all files; 1 => always cold, 0 => always warm, None => do not touch the cache
    'workload_evict_rate',
]

# What identifies resources used

NUM_THREADS = [
    # number of threads used
    'workload_num_threads',
]

# What identifies a repetition (usually we would need to add a mean to this)
REPETITION_ID = [
    'repetition',
]


# --- Write workloads

WRITE_RESULT_OBJECT_COLUMNS = ('result', 'video', 'location', 'codec', 'workload', 'watch', 'syncer')

WRITE_COLUMNS = [
    'result',
    'result_type',
    'success',
    'total_time',
    'total_time_human',
    # original video (should also collect codec info and put it here); see Video class
    'video',
    'video_name',
    'video_motion',
    'video_constant_bg',
    'video_original_resolution',
    'video_original_container',
    # clip (segment of the original video); FIXME: should split out of Video class
    'clip_name',
    'clip_start',
    'clip_num_frames',
    'clip_step',
    'clip_new_shape',
    'clip_resolution',
    'clip_actual_num_frames',
    # location (could also bring: path, comments, theoretical_peak_read, meta)
    'location',
    'location_name',
    'location_host',
    'location_type',
    # codec (akin to "dest video" characteristics)
    'codec',
    'codec_name',
    'codec_codec',
    'codec_pixfmt',
    'codec_params',
    'codec_container',
    'codec_lossless',
    'codec_intra_only',
    # workload
    'workload',
    'workload_name',
    'workload_resolution',
    # in workload, but actually a writer configuration
    'workload_new_shape',
    'workload_num_threads',
    # repetition
    'repetition',
    # disk usage
    'du_bytes',
    'du_human',
    'compression_ratio',
    'space_savings',
    # watch (if we get bored, we can extract many other interesting measurements here)
    'watch',
    'wall_time',  # N.B., only of get operations
    'fps_wall',
    'fps_ru',
    'num_written_frames',
    # read sanity checker
    'syncer',
    'sync_wall_time',
]


def load_write_result_to_dict(result_path,
                              columns_to_remove=WRITE_RESULT_OBJECT_COLUMNS):

    result = pd.read_pickle(op.join(result_path, 'result.pkl'))

    if result['type'] != 'write':
        raise ValueError('the result at %s is not a reading result' % result_path)

    # We can decide to add more stuff from this to the analysis
    watch = result['watch']  # type: FFMPEGWatchCallback

    result = dict(
        # result itself
        result=result,
        result_type=result['type'],
        success=result['success'],
        total_time=result['total_watch'].wall_time,
        total_time_human=humanize.naturaltime(result['total_watch'].wall_time),
        # original video (should also collect codec info and put it here); see Video class
        video=result['video'],
        video_name=result['video'].video_id,
        video_motion=result['video'].motion,
        video_constant_bg=result['video'].constant_bg,
        video_original_resolution=result['video'].resolution,
        video_original_container=result['video'].container,
        # clip (segment of the original video); FIXME: should split out of Video class
        clip_name=result['video'].name,
        clip_start=result['video'].start,
        clip_num_frames=result['video'].num_frames,
        clip_step=result['video'].step,
        clip_new_shape=result['workload'].config['new_shape'],
        clip_actual_num_frames=watch.watch.num_units,
        # location (could also bring: path, comments, theoretical_peak_read, meta)
        location=result['location'],
        location_name=result['location'].name,
        location_host=result['location'].host,
        location_type=result['location'].type,
        # codec (akin to "dest video" characteristics)
        codec=result['codec'].codec,
        codec_name=result['codec'].name,
        codec_codec=result['codec'].codec,  # ugly!
        codec_pixfmt=result['codec'].pix_fmt,
        codec_params=result['codec'].param_tuples,
        codec_container=result['codec'].container,
        codec_lossless=result['codec'].is_lossless,
        codec_intra_only=result['codec'].is_intra_only,
        # repetition
        repetition=result['repetition'],
        # workload
        workload=result['workload'],
        workload_name=result['workload'].name,
        # in workload, but actually a writer configuration;
        workload_new_shape=result['workload'].config['new_shape'],
        workload_num_threads=None,  # we used whatever default from FFMPEGReader
        # disk usage
        du_bytes=result['disk_usage_bytes'],
        du_human=humanize.naturalsize(result['disk_usage_bytes']),
        # watch (if we get bored, we can extract many other interesting measurements here)
        watch=watch,
        wall_time=watch.watch.wall_time,  # N.B., only of get operations
        fps_wall=watch.ups_wall,
        fps_ru=watch.ups_ru,
        num_written_frames=watch.watch.num_units,
        # sync callback
        syncer=result['syncer'],
        sync_wall_time=result['syncer'].watch.total
    )

    # Add compression ratio and space savings
    result['compression_ratio'] = compression_ratio(result)
    result['space_savings'] = 100 * (1 - 1 / compression_ratio(result))
    # Make a resolution column
    result['clip_resolution'] = {r: rname for rname, r in RESOLUTIONS.items()}[result['clip_new_shape']]

    if columns_to_remove:
        for column in columns_to_remove:
            del result[column]

    return result


# --- Read workloads

READ_RESULT_OBJECT_COLUMNS = ('result', 'video', 'location', 'codec', 'workload', 'watch', 'checker')

READ_COLUMNS = [
    'result',
    'result_type',
    'success',
    'total_time',
    'total_time_human',
    # original video (should also collect codec info and put it here); see Video class
    'video',
    'video_name',
    'video_motion',
    'video_constant_bg',
    'video_original_resolution',
    'video_original_container',
    # clip (segment of the original video); FIXME: should split out of Video class
    'clip_name',
    'clip_start',
    'clip_num_frames',
    'clip_step',
    'clip_new_shape',
    'clip_resolution',
    'clip_actual_num_frames',
    # location (could also bring: path, comments, theoretical_peak_read, meta)
    'location',
    'location_name',
    'location_host',
    'location_type',
    # codec (akin to "dest video" characteristics)
    'codec',
    'codec_name',
    'codec_codec',
    'codec_pixfmt',
    'codec_params',
    'codec_container',
    'codec_lossless',
    'codec_intra_only',
    # workload
    'workload',
    'workload_name',
    'workload_sequential',
    'workload_step',
    'workload_batch_size',
    'workload_evict_rate',
    'workload_initial_fs_state',
    'workload_fs_cache',
    # in workload, but actually a reader configuration
    'workload_seek',
    'workload_num_threads',
    # repetition
    'repetition',
    # disk usage
    'du_bytes',
    'du_human',
    'compression_ratio',
    'space_savings',
    # watch (if we get bored, we can extract many other interesting measurements here)
    'watch',
    'wall_time',  # N.B., only of get operations
    'fps_wall',
    'fps_ru',
    'num_read_frames',
    # read sanity checker
    'checker',
    'pct_correct',
]


def load_read_result_to_dict(result_path,
                             columns_to_remove=READ_RESULT_OBJECT_COLUMNS):

    result = pd.read_pickle(op.join(result_path, 'result.pkl'))

    if result['type'] != 'read':
        raise ValueError('the result at %s is not a reading result' % result_path)

    # We can decide to add more stuff from this to the analysis
    watch = result['watch'].watch  # type: Rusager

    result = dict(
        # result itself
        result=result,
        result_type=result['type'],
        success=result['success'],
        total_time=result['total_watch'].wall_time,
        total_time_human=humanize.naturaltime(result['total_watch'].wall_time),
        # original video (should also collect codec info and put it here); see Video class
        video=result['video'],
        video_name=result['video'].video_id,
        video_motion=result['video'].motion,
        video_constant_bg=result['video'].constant_bg,
        video_original_resolution=result['video'].resolution,
        video_original_container=result['video'].container,
        # clip (segment of the original video); FIXME: should split out of Video class
        clip_name=result['video'].name,
        clip_start=result['video'].start,
        clip_num_frames=result['video'].num_frames,
        clip_step=result['video'].step,
        clip_new_shape=result['clip_new_shape'],
        clip_actual_num_frames=result['clip_actual_num_frames'],
        # location (could also bring: path, comments, theoretical_peak_read, meta)
        location=result['location'],
        location_name=result['location'].name,
        location_host=result['location'].host,
        location_type=result['location'].type,
        # codec (akin to "dest video" characteristics)
        codec=result['codec'].codec,
        codec_name=result['codec'].name,
        codec_codec=result['codec'].codec,  # ugly!
        codec_pixfmt=result['codec'].pix_fmt,
        codec_params=result['codec'].param_tuples,
        codec_container=result['codec'].container,
        codec_lossless=result['codec'].is_lossless,
        codec_intra_only=result['codec'].is_intra_only,
        # repetition
        repetition=result['repetition'],
        # workload
        workload=result['workload'],
        workload_name=result['workload'].name,
        workload_sequential=result['workload'].config['sequential'],
        workload_step=result['workload'].config['step'],
        workload_batch_size=result['workload'].config['batch_size'],
        workload_evict_rate=result['workload'].config['evict_rate'],
        workload_initial_fs_state=result['workload'].config['initial_fs_state'],
        # in workload, but actually a reader configuration
        workload_seek=result['workload'].config['seek'],
        workload_num_threads=result['workload'].config['num_threads'],
        # disk usage
        du_bytes=result['disk_usage_bytes'],
        du_human=humanize.naturalsize(result['disk_usage_bytes']),
        # watch (if we get bored, we can extract many other interesting measurements here)
        watch=watch,
        wall_time=watch.wall_time,  # N.B., only of get operations
        fps_wall=watch.ups_wall,
        fps_ru=watch.ups_ru,
        num_read_frames=watch.num_units,
        # read sanity checker
        checker=result['checker'],
        pct_correct=result['checker'].pct_correct,
    )

    # Add compression ratio and space savings
    result['compression_ratio'] = compression_ratio(result)
    result['space_savings'] = 100 * (1 - 1 / compression_ratio(result))
    # Make a resolution column
    result['clip_resolution'] = {r: rname for rname, r in RESOLUTIONS.items()}[result['clip_new_shape']]
    # Make missing fs_state a string "whatever", si we can query using numexpr
    if not result['workload_initial_fs_state']:
        result['workload_initial_fs_state'] = 'whatever'

    # Create a fs_cache_state column
    def fs_cache_state(row):
        if row['workload_initial_fs_state'] == 'whatever' and row['workload_evict_rate'] == 0:
            return 'always-warm'
        elif row['workload_initial_fs_state'] == 'whatever' and row['workload_evict_rate'] == 1:
            return 'always-cold'
        elif row['workload_initial_fs_state'] == 'cold' and row['workload_evict_rate'] is None:
            return 'start-cold'
        raise ValueError('Unknown (inital_fs_state, evict_rate) combination')
    result['workload_fs_cache'] = fs_cache_state(result)

    if columns_to_remove:
        for column in columns_to_remove:
            del result[column]

    return result


# --- To dataframe

def load_all_results(results_dir=op.join(BENCHMARKS_ROOT, 'results'),
                     read_results=True,
                     reread=False):

    cache_path = op.join(results_dir, 'read-results.pkl' if read_results else 'write-results.pkl')

    if not reread and op.isfile(cache_path):
        return pd.read_pickle(cache_path)

    factory = load_read_result_to_dict if read_results else load_write_result_to_dict
    results = []
    for dirpath, dirnames, filenames in os.walk(results_dir):
        if 'result.pkl' in filenames and 'DONE.txt' in filenames:
            try:
                results.append(factory(dirpath))
            except ValueError:
                pass

    # noinspection PyTypeChecker
    df = pd.DataFrame(results)

    # Tidy column order
    columns = list(READ_COLUMNS if read_results else WRITE_COLUMNS)
    columns = [column for column in columns if column in df.columns]
    columns += [column for column in df.columns if column not in columns]

    df = df[columns]

    # Sane row order
    order_by = ['clip_name',
                'clip_new_shape',
                'location_name',
                'codec_name',
                'workload_name',
                'repetition',
                'fps_wall']
    df = df.sort_values(order_by).reset_index(drop=True)

    # Categoricals

    # Cache
    pd.to_pickle(df, cache_path)

    return df

# --- Example analysis


def warm_start_vs_evict_rate(df):
    """
    This was a little analysis with the first version of the data.
    Looking at the resulting dataframes, I decided that:

    - I would quickly try to repeat the experiments without fs-status-start
      The effect would be seen only via evict_rate.

    - I would reduce the amount of tried codecs (e.g. no mjpeg pixel_fmt or png
      yet).

    This allows to see this table:

            1      0
    warm   cold   warm
    cold   cold   ?*

    ?* => getting warm as reads happen, but since we do not read repeatedly ATM,
    the warming effect can only be seen on real video files for which data that
    will be read later is cached if contiguous data is requested on a fetch.
    """

    # df = df.query('video_name != "c-elegans" and codec_container in ["jpg", "npy", "png"]')
    # df = df.query('video_name != "c-elegans" and codec_codec in ["mjpeg"]')

    df = df.set_index([
        'video_name',
        'clip_new_shape',
        'location_name',
        'codec_name',
        'workload_sequential',
        'workload_batch_size',
        'workload_seek',
        'workload_num_threads',
    ])

    warm_1 = df.query('workload_initial_fs_state == "warm" and workload_evict_rate == 1')[['du_human', 'fps_wall']]
    cold_1 = df.query('workload_initial_fs_state == "cold" and workload_evict_rate == 1')[['fps_wall']]
    warm_0 = df.query('workload_initial_fs_state == "warm" and workload_evict_rate == 0')[['fps_wall']]
    cold_0 = df.query('workload_initial_fs_state == "cold" and workload_evict_rate == 0')[['fps_wall']]

    df1 = pd.merge(warm_1, cold_1, left_index=True, right_index=True, suffixes=('_warm1', '_cold1'))
    df0 = pd.merge(warm_0, cold_0, left_index=True, right_index=True, suffixes=('_warm0', '_cold0'))
    df = pd.merge(df1, df0, left_index=True, right_index=True)

    df = df.reset_index(drop=False).sort_values(['video_name',
                                                 'clip_new_shape',
                                                 'workload_sequential',
                                                 'workload_batch_size',
                                                 'workload_seek',
                                                 'location_name',
                                                 'workload_num_threads',
                                                 'codec_name'])

    print(df)
    df.to_html(op.expanduser('~/warm0-warm1-cold0-cold1.html'))


def evict_rate_vs(df):

    df = df.set_index([
        'video_name',
        'clip_new_shape',
        'location_name',
        'codec_name',
        'workload_sequential',
        'workload_batch_size',
        'workload_seek',
        'workload_num_threads',
    ])

    cold = df.query('workload_evict_rate == 1')[['du_human', 'space_savings', 'fps_wall', 'fps_ru']]
    warm = df.query('workload_evict_rate == 0')[['fps_wall', 'fps_ru']]

    df = pd.merge(cold, warm, left_index=True, right_index=True, suffixes=['_cold', '_warm'])

    df = df.reset_index(drop=False).sort_values(['video_name',
                                                 'clip_new_shape',
                                                 'workload_seek',
                                                 'location_name',
                                                 'workload_num_threads',
                                                 'codec_name',
                                                 'workload_batch_size',
                                                 'workload_sequential'])

    print(df)
    df.to_html(op.expanduser('~/warm0-warm1-cold0-cold1.html'))


def xbyx(df,
         video_name='manakins',
         ignore_codecs=(),
         clip_resolution='medium',
         speed_metric='fps_ru',  # 'fps_ru' or 'fps_wall'
         disk_metric='space_savings',
         workload_sequential=False,
         workload_batch_size=32,
         location_name='mumbler-hdd',
         workload_fs_cache='always-warm',
         workload_num_threads=(1, 2, 4),
         col='workload_sequential',
         row='workload_batch_size',
         title=None):

    df = df.copy()

    query = [
        'codec_codec not in %r' % (ignore_codecs,),
        'video_name == %r' % video_name,
    ]

    if 'clip_resolution' not in (row, col):
        query += ['clip_resolution == %r' % clip_resolution]

    if 'workload_sequential' not in (row, col):
        query += ['workload_sequential' if workload_sequential else 'not workload_sequential']

    if 'workload_batch_size' not in (row, col):
        query += ['workload_batch_size == %d' % workload_batch_size]

    if 'location_name' not in (row, col):
        query += ['location_name == %r' % location_name]

    if 'workload_fs_cache' not in (row, col):
        query += ['workload_fs_cache == %r' % workload_fs_cache]

    if 'workload_num_threads' not in (row, col):
        query += ['workload_num_threads in %r' % (workload_num_threads,)]

    df = df.query(' and '.join(query))

    # Select the fastest configuration (num_threads) for each codec
    # Some ideas: https://tinyurl.com/y9nn6b9z
    df = df.sort_values(speed_metric, ascending=False).drop_duplicates(['codec_codec', col, row])

    # Tune strings for the legend
    # df['codec_name'] = df['codec_name'] + '_' + df['workload_num_threads'].map(str)
    # df['codec_codec'] = df['codec_codec'] + '_' + df['workload_num_threads'].map(str)

    # Ensure color consistency (just make a dict...)
    # df = df.sort_values('codec_codec')
    df = df.sort_values('codec_name')

    speed_metric_nice = 'FPS (wall time)' if 'wall' in speed_metric else 'FPS (cpu time)'
    disk_metric_nice = 'Space savings'
    df = df.rename(columns={'codec_codec': 'codec',
                            speed_metric: speed_metric_nice,
                            disk_metric: disk_metric_nice})

    # Do the plot
    lm = sns.lmplot(speed_metric_nice, disk_metric_nice, data=df,
                    hue='codec', col=col, row=row,
                    legend_out=False,
                    fit_reg=False, )

    if title is None:
        title = 'Video: %s, Resolution: %s' % (video_name, clip_resolution)
    lm.fig.suptitle(title + ' - maxFPS=%.1f' % df[speed_metric_nice].max(), fontsize=16)
    lm.fig.subplots_adjust(top=.9)

    dest_dir = ensure_dir(op.expanduser('~/seq-vs-random'),
                          video_name + '_' + clip_resolution,
                          location_name,
                          speed_metric)
    fn = '{speed_metric}-{title}'.format(
        speed_metric=speed_metric,
        title=title.replace(' ', '')
    )
    columns = ['codec', row, col, disk_metric_nice, speed_metric_nice]
    (df[columns].
     sort_values(columns).
     to_csv(op.join(dest_dir, fn + '.csv'), index=False, float_format='%.1f'))
    plt.savefig(op.join(dest_dir, fn + '.png'))

    plt.close(lm.fig)


def read_seek_speed_plot(read_df, ignore_codecs=('npy', 'mjpeg')):

    # Let's remove the usual suspects
    df = read_df.query('codec_codec not in %r' % (ignore_codecs,))

    # Get only data for random seeking and batch_size 1, corresponding to complete random access
    df = df.query('not workload_sequential and workload_batch_size == 1').copy()

    df = normalize_metrics(df, groupers=('video_name', 'clip_resolution', 'location_name',
                                         'workload_fs_cache', 'workload_num_threads'),
                           ignore_no_straw_man=True  # mmmm missing a few experiments here,
                                                     # reactivate the check when redone
                           )

    df = df.groupby(['codec_name', 'location_name'])[['space_savings',
                                                      'normalized_fps_wall']].agg([np.mean, np.std]).reset_index()
    print(df)

    df.to_pickle(op.join(BENCHMARKS_ROOT, 'results', 'video_random_seek_is_slow.pkl'))


def write_tradeoffs(write_df):

    df = write_df.copy()

    # Numpy being such an outlier, makes plots ugly; probably also mjpeg should say bye bye
    df = df.query('codec_codec not in ["npy", "mjpeg"]').copy()

    # Choose one (I like space savings the most)
    df['du_MB'] = df['du_bytes'] / 1024 ** 2
    disk_usage_metrics = 'space_savings',    # , 'compression_ratio', 'du_MB'

    # I would like to show both "wall" and "ru" (I worked a lot to make measuring fps_ru kinda work)
    # But let's see if it shows something interesting...
    speed_metrics = 'fps_wall',  'fps_ru'  # , 'wall_time'

    # Choose one? These are interesting mostly in reading workloads
    # location_names = 'mumbler-ssd', 'mumbler-hdd'

    for disk_metric, speed_metric in product(disk_usage_metrics, speed_metrics):
        for video_name, vdf in df.groupby(['video_name']):
            speed_metric_nice = {
                'fps_wall': 'FPS (wall time)',
                'fps_ru': 'FPS (cpu time)',
                'wall_time': 'time (s)',
            }[speed_metric]
            disk_metric_nice = {
                'compression_ratio': 'Compression ratio',
                'du_MB': 'Disk usage (MB)',
                'space_savings': 'Space savings',
            }[disk_metric]
            vdf = vdf.sort_values('codec_name')
            vdf = vdf.rename(columns={'codec_name': 'codec',
                                      speed_metric: speed_metric_nice,
                                      disk_metric: disk_metric_nice})
            # TODO: always keep order of subplots
            lm = sns.lmplot(speed_metric_nice, disk_metric_nice, data=vdf,
                            hue='codec', col='location_type', row='clip_resolution',
                            fit_reg=False, legend_out=True, sharex=False, sharey=False)
            lm.set(ylim=(0, 100))
            lm.set(xlim=(0, None))
            lm.fig.suptitle('%s: %s vs %s' % (video_name, speed_metric_nice, disk_metric_nice), fontsize=16)
            lm.fig.subplots_adjust(top=.9)
            dest_dir = ensure_dir(op.expanduser('~/write-results'))  # video_name, du_metric, speed_metric
            fn = 'write-scatter-{video_name}-{speed_metric}-{du_metric}'.format(
                video_name=video_name,
                speed_metric=speed_metric,
                du_metric=disk_metric,
            )
            plt.savefig(op.join(dest_dir, fn + '.png'))
            plt.close()


if __name__ == '__main__':

    read_results = True

    df = load_all_results(read_results=read_results, reread=False)
    df.info()

    if not read_results:
        print(df['total_time'].mean(), len(df))
        df = normalize_metrics(df)
        print(df.groupby(['codec_name', 'location_name'])[['normalized_fps_wall',
                                                           'normalized_fps_ru',
                                                           'normalized_space_savings']].agg([np.mean, np.std]))
        df = rank_group_metrics(df)
        print(df.groupby(['codec_name', 'location_name'])[['ranked_fps_wall',
                                                           'ranked_fps_ru',
                                                           'ranked_space_savings']].agg([np.mean, np.std]))
        pd.to_pickle(df, op.join(BENCHMARKS_ROOT, 'results', 'write_results_normalized.pkl'))
        write_tradeoffs(df)

    if read_results:
        # Make sure there are no read errors
        assert len(df.query('pct_correct < 100')) == 0, 'There are error seeks'

        # Generate a DF aggregating random seek without amortized sequential reads
        read_seek_speed_plot(df)

        # Generate a nice DF to inspect some trade-offs
        # evict_rate_vs(df)

        # Is trying several different resolutions worth it in this experiment?
        # That is, do we see different things between resolutions?
        # for video in df['video_name'].unique():
        #     xbyx(df=df,
        #          speed_metric='fps_wall',
        #          ignore_codecs=('mjpeg',),
        #          video_name=video,
        #          col='clip_resolution', row='workload_fs_cache')

        # From now on, let's just see results in the medium resolution
        # They usually extrapolate to other resolutions (for example in small ones ffvhuff shines)
        # df = df.query('clip_resolution == "medium"')

        speed_metrics = 'fps_wall',  # , 'fps_ru'
        workload_sequentials = True, False
        workload_batch_sizes = 1, 32
        location_names = 'mumbler-ssd', 'mumbler-hdd'
        workload_fs_caches = 'always-warm', 'always-cold', 'start-cold'
        workload_num_threadss = (1,), (1, 2, 4),

        for speed_metric in speed_metrics:
            for (video, resolution), vdf in df.groupby(['video_name', 'clip_resolution']):

                # --- Look at location + cache state relation

                # xbyx(df=vdf,
                #      speed_metric=speed_metric,
                #      ignore_codecs=('mjpeg',),
                #      video_name=video, clip_resolution=resolution,
                #      col='location_name', row='workload_fs_cache')

                # --- How does speed change depending on access pattern?

                for location_name, workload_fs_cache, workload_num_threads in product(
                    location_names,
                    workload_fs_caches,
                    workload_num_threadss
                ):
                    title = 'video={video}-{resolution}, {location}, {cache}, {num_threads}'.format(
                        video=video, resolution=resolution,
                        location=location_name.split('-')[1],
                        cache=workload_fs_cache,
                        num_threads='single-threaded' if workload_num_threads == (1,) else 'up-to-4-threads'
                    )
                    xbyx(df=vdf,
                         speed_metric=speed_metric,
                         ignore_codecs=('mjpeg',),
                         video_name=video, clip_resolution=resolution,
                         location_name=location_name,
                         workload_fs_cache=workload_fs_cache,
                         workload_num_threads=workload_num_threads,
                         col='workload_sequential', row='workload_batch_size',
                         title=title)

    print('DONE')
