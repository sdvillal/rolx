from __future__ import print_function, division
import os
import os.path as op
import socket
import datetime
import sys
from subprocess import check_output
import json
import psutil
import numpy as np
import cv2


# --- Image

def to_shape(image_or_shape):
    if isinstance(image_or_shape, tuple):
        num_rows, num_cols = image_or_shape[0], image_or_shape[1]
    else:
        img = np.atleast_3d(image_or_shape)
        num_rows, num_cols = img.shape[0], img.shape[1]
    return num_rows, num_cols


def to_size(image_or_shape):
    height, width = to_shape(image_or_shape)
    return width, height


def new_size(image_or_shape, new_width=None, new_height=None):
    if new_width is not None and new_height is not None:
        return new_width, new_height
    old_width, old_height = to_size(image_or_shape)
    if new_width is None and new_height is None:
        new_width, new_height = old_width, old_height
    elif new_width is not None:
        new_width = int(round(old_width * new_height / old_height))
    elif new_height is None:
        new_height = int(round(old_height * new_width / old_width))
    return new_width, new_height


def resize(image, new_width=None, new_height=None, force_copy=False):

    # Compute the new shape
    old_width, old_height = to_size(image)
    new_width, new_height = new_size(image, new_width=new_width, new_height=new_height)

    # No-op?
    if new_width == old_width and new_height == old_height:
        return image if not force_copy else image.copy()

    # Select interpolation method
    old_area = old_width * old_height
    new_area = new_width * new_height
    shrinking = new_area < old_area
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC

    # Resize
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


# --- File system

def _check_dir(path, check_writable=False):
    if not op.isdir(path):
        raise Exception('%s exists but it is not a directory' % path)
    if check_writable and not os.access(path, os.W_OK):
        raise Exception('%s is a directory but it is not writable' % path)


def _ensure_dir(path, check_writable=False):
    if os.path.exists(path):
        _check_dir(path, check_writable=check_writable)
    else:
        try:
            os.makedirs(path)
        except Exception:
            if os.path.exists(path):  # Simpler than using a file lock to work on multithreading...
                _check_dir(path, check_writable=check_writable)
            else:
                raise
    return path


def ensure_dir(path0, *parts):
    return _ensure_dir(os.path.join(path0, *parts), check_writable=False)


def ensure_writable_dir(path0, *parts):
    return _ensure_dir(os.path.join(path0, *parts), check_writable=True)


# --- Benchmarking utilities (mainly for linux).
#
# Some of these are inspired by bloscpack / bloscpack-benchmarks.
#   https://github.com/Blosc/bloscpack-benchmarking
#
# Copied from jagged, get back there any changes.


def timestr():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def hostname():
    return socket.gethostname()


# noinspection PyUnusedLocal
def collect_sysinfo(dest=None):
    """
    Collects basic information from the machine using several tools.
    This needs to run as root.
    Note that speeds are theoretical, not measured
    (specially peak network and network drives speeds should be measured).

    Prerequisites
    -------------
    If in ubuntu:
      sudo apt-get install smartmontools inxi dmidecode
    If in arch:
     sudo pacman -S smartmontools inxi dmidecode

    What is run
    -----------
    # Basic information about mount points
    mount > mount.info
    # Inxi reports
    inxi > inxi.info
    # Full dmidecode
    dmidecode > dmidecode.info
    # Network speed information
    dmesg | grep -i duplex > network-speed.info
    # SMART information
    sudo smartctl -a /dev/sda > smartctl-sda.info

    References
    ----------
    http://www.binarytides.com/linux-commands-hardware-info/
    http://www.cyberciti.biz/faq/linux-command-to-find-sata-harddisk-link-speed/
    http://www.cyberciti.biz/faq/howto-setup-linux-lan-card-find-out-full-duplex-half-speed-or-mode/
    http://www.cyberciti.biz/tips/linux-find-out-wireless-network-speed-signal-strength.html
    """

    raise NotImplementedError()

    #
    # Any way of getting actual memory latencies, CAS...?
    # Also we could look at pure python libraries like dmidecode
    #

    # if dest is None:
    #     dest = op.join(op.dirname(__file__), 'sysinfo')
    # dest = op.join(ensure_dir(op.join(dest, hostname())), timestr() + '.json')
    #
    # info = {
    #     'mount': check_output('mount'),
    #     'dmesg-eth': '\n'.join(line for line in check_output('dmesg').splitlines() if 'duplex' in line),
    #     'iwconfig': check_output('iwconfig'),
    #     'inxiF': check_output(['inxi', '-c 0', '-F']),
    #     # add some more inxi stuff
    # }
    #
    # with open(dest, 'w') as writer:
    #     json.dump(info, writer, indent=2, sort_keys=True)
    #
    # return info


def du(path):
    """Returns the size of the tree under path in bytes."""
    return int(check_output(['du', '-s', '-L', '-B1', path]).split()[0].decode('utf-8'))


def _fincore(*file_paths):

    # could also use vmtouch -v for this, but fincore's json is just too convenient

    if file_paths:
        results = check_output(['fincore', '--json', '--bytes', '--'] + list(file_paths))
        results = json.loads(results).get('fincore', [])

        for file_result in results:
            for field in ('res', 'pages', 'size'):
                file_result[field] = int(file_result[field])

        return results

    return []


def fincore(path):
    """
    Returns a list of dictionaries for all the files in `path` (recursively).

    The dictionaries contain:
      - pages: file data resident in memory in pages
      - res: file data resident in memory in bytes
      - size: size of the file in bytes
      - file: file path
    """
    if not op.isdir(path):
        return _fincore(path)
    results = []
    for dirpath, dirnames, filenames in os.walk(path):
        results += _fincore(*[op.join(op.abspath(dirpath), fn) for fn in filenames])
    return results


def drop_caches(path, drop_level=3, max_size='1000G', do_sync=False, verbose=False):
    #
    # Some light reading
    #   http://www.linuxatemyram.com/play.html
    # vmtouch
    #   https://hoytech.com/vmtouch/
    #   https://aur.archlinux.org/packages/vmtouch/
    #   http://serverfault.com/questions/278454/is-it-possible-to-list-the-files-that-are-cached
    #   http://serverfault.com/questions/43383/caching-preloading-files-on-linux-into-ram
    #
    # To drop system caches, one needs root.
    # The best approach is to use vmtouch, which can selectively evict pages under a path.
    # My approach:
    #  1- install vmtouch
    #     there are packages for arch an modern ubuntus
    #     compiling is easy
    #
    #  2- add it to sudoers so no pass is required.
    #     sudo visudo
    #     add something like: santi ALL = (root) NOPASSWD: /usr/bin/vmtouch
    #
    if do_sync:
        sync()
    if 0 != os.system('vmtouch -e -f -q -m %s "%s"' % (max_size, path)):
        if os.geteuid() == 0:
            os.system('echo %d > /proc/sys/vm/drop_caches' % drop_level)
            if verbose:
                print('Full system cache dropped because of %s' % path)
        else:
            raise RuntimeError('Need vmtouch or root permission to drop caches')
    else:
        if verbose:
            print('All pages under %s evicted' % path)


def evict_pages(path, max_size='1000G', do_sync=False, verbose=False):
    if do_sync:
        sync()
    if 0 != os.system('vmtouch -e -f -q -m %s "%s"' % (max_size, path)):
        raise RuntimeError('Cannot evict pages at %s\n\tEnsure vmtouch is setup and the path exists' % path)
    if verbose:
        print('All pages under %s evicted' % path)


ensure_cold = evict_pages


def touch_pages(path, max_size='1000G', do_sync=False, verbose=False):
    # TODO: wrap also -l and -L (to really ensure warm caches) and -p
    if do_sync:
        sync()
    if 0 != os.system('vmtouch -t -f -q -m %s "%s"' % (max_size, path)):
        raise RuntimeError('Cannot touch pages at %s\n\tEnsure vmtouch is setup and the path exists' % path)
    if verbose:
        print('All pages under %s touched' % path)


ensure_warm = touch_pages


def ensure_fs_cache_state(reader, start_type=None):
    if start_type == 'warm':
        ensure_warm(reader.path)
    elif start_type == 'cold':
        ensure_cold(reader.path)
    elif start_type is not None:
        raise ValueError('unknown start_type %r; must be one of ["warm", "cold", None]')


def sync():
    """Flushes buffers to disk."""
    os.system('sync')


def available_ram():
    """Returns system available memory, in bytes."""
    return psutil.virtual_memory().available

# TODO: check other tools:
#   - nocache: https://github.com/Feh/nocache (already installed from AUR)
#   - dd & posix_fadvice & pyadvice:
#     https://unix.stackexchange.com/questions/36907/drop-a-specific-file-from-the-linux-filesystem-cache
#     https://linux.die.net/man/2/posix_fadvise
#     https://chris-lamb.co.uk/projects/python-fadvise
#     https://github.com/lamby/python-fadvise

#
# Timing is hard and we should at least use timeit
# (something with support for calibration and repetition).
# A great resource is also pytest benchmark
#   https://pypi.python.org/pypi/pytest-benchmark/2.5.0
#   https://bitbucket.org/haypo/misc/src/tip/python/benchmark.py
# There are a bunch of benchmarker / timer etc. libraries in pypi

#
# We need to make sure that:
#  - we go beyond microbenchmarks and look at relevant tasks
#    e.g. realtime visualisation or data exploration as opposed to batch
#

# Measure dataset complexity (e.g. lempel ziv via compression) and report it

# Other things we would like to touch/flush: ssd drive caches / buffers, remote caches,
# cached stuff...

#
# Do not forget about /usr/bin/time -v
#   $ sudo pacman -S time
#   $ time -v echo "Hi"
#   	Command being timed: "echo Hi"
# 	    User time (seconds): 0.00
# 	    System time (seconds): 0.00
# 	    Percent of CPU this job got: ?%
# 	    Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.00
# 	    Average shared text size (kbytes): 0
# 	    Average unshared data size (kbytes): 0
# 	    Average stack size (kbytes): 0
# 	    Average total size (kbytes): 0
# 	    Maximum resident set size (kbytes): 1712
# 	    Average resident set size (kbytes): 0
# 	    Major (requiring I/O) page faults: 0
# 	    Minor (reclaiming a frame) page faults: 68
# 	    Voluntary context switches: 1
# 	    Involuntary context switches: 1
# 	    Swaps: 0
# 	    File system inputs: 0
# 	    File system outputs: 0
# 	    Socket messages sent: 0
# 	    Socket messages received: 0
# 	    Signals delivered: 0
# 	    Page size (bytes): 4096
# 	    Exit status: 0
#


# --- Misc

def sanitize_start_stop(x_with_length, start, stop):
    if start is None:
        start = 0
    if stop is None:
        stop = len(x_with_length)

    start = min(max(0, start), len(x_with_length))
    stop = max(0, min(stop, len(x_with_length)))

    return start, stop


def flush():
    sys.stdout.flush()
    sys.stderr.flush()


def force_new_line(stdout=True, stderr=True):
    flush()
    if stdout:
        print(file=sys.stdout)
    if stderr:
        print(file=sys.stderr)
    flush()
