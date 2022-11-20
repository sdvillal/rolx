from __future__ import division

import contextlib
import os
import sys
from array import array
try:
    from collections import OrderedDict, MutableMapping, namedtuple
except ImportError:
    from collections import OrderedDict, namedtuple
    from collections.abc import MutableMapping
from functools import wraps
import time
import resource
import math
import numpy as np


# --- Units

Unit = namedtuple('Unit', ['name', 'plural', 'symbol'])
FrameUnit = Unit('frame', 'frames', 'f')

# --- Cumulative measurements
# These are meant to be flexible, not high precission or efficient.
# In particular, do not measure things that take too little, or accumulated error will dominate.


class IterativeAggregator(object):
    """
    Note `count` vs `n`:
      count is the number of times add has been called since the last reset
      n is the number of datapoints used to inform the mean and std
    """

    def __init__(self, name, units=None, add_filter=None, comments=None):
        super(IterativeAggregator, self).__init__()
        self._count = 0  # Counter for the number of times add has been called
        self.name = name
        self.units = units
        self.comments = comments
        self.add_filter = add_filter

    def add(self, x, count=1):
        self._count += count
        if self.add_filter is not None:
            x = self.add_filter(self, x)
        if x is not None:
            self._add_hook(x, count=count)
        return self

    def _add_hook(self, x, count):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def var(self):
        raise NotImplementedError()

    @property
    def std(self):
        # noinspection PyTypeChecker
        return math.sqrt(self.var)

    @property
    def n(self):
        raise NotImplementedError()

    @property
    def total(self):
        # noinspection PyUnresolvedReferences
        return self.mean * self.n

    @property
    def count(self):
        return self._count

    def __call__(self, x):
        self.add(x)
        return self.mean, self.std

    def __str__(self):
        # noinspection PyStringFormat
        msg = '%s: %.2f +/- %.2f' % (self.name, self.mean, self.std)
        if self.units:
            return msg + ' ' + self.units
        return msg


class OnlineMeanStd(IterativeAggregator):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    def __init__(self, name, units=None, add_filter=None, comments=None):
        super(OnlineMeanStd, self).__init__(name, units=units, add_filter=add_filter, comments=comments)
        self._n = 0
        self._mean = 0
        self._m2 = 0
        self._max = -np.inf
        self._min = np.inf

    def _add_hook(self, x, count):
        x /= count
        for _ in range(count):
            self._n += 1
            delta = x - self._mean
            self._mean += delta / self._n
            delta2 = x - self._mean
            self._m2 += delta * delta2
        self._max = max(self._max, x)
        self._min = min(self._min, x)

    def reset(self):
        self._n = 0
        self._mean = 0
        self._m2 = 0
        self._max = -np.inf
        self._min = np.inf

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        if self._n < 2:
            return float('nan')
        return self._m2 / (self._n - 1)

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    @property
    def n(self):
        return self._n


class KeepAllMeanStd(IterativeAggregator):

    def __init__(self, name, units=None, add_filter=None,
                 dtype='f', keep_indices=False):

        super(KeepAllMeanStd, self).__init__(name, units, add_filter)
        self._dtype = dtype
        self._values = None
        self._indices = None
        self._keep_indices = keep_indices
        self.reset()

    def reset(self):
        self._values = array(self._dtype)
        if self._keep_indices:
            self._indices = array('l')

    def _add_hook(self, x, count):
        x /= count
        for _ in range(count):
            self._values.append(x)
            if self._keep_indices:
                self._indices.append(self.count)

    def values(self):
        return np.frombuffer(self._values, dtype=self._values.typecode)

    @property
    def mean(self):
        return self.values().mean()

    @property
    def var(self):
        return self.values().var()

    @property
    def n(self):
        return len(self._values)


class KeepLastMeanStd(IterativeAggregator):
    """Sort of simple moving average."""
    def __init__(self, name, units=None, add_filter=None,
                 last=100, dtype='d', keep_indices=False):
        super(KeepLastMeanStd, self).__init__(name, units, add_filter)
        self.last = last
        self.dtype = dtype
        self.keep_indices = keep_indices
        self._counter = None
        self._full = None
        self._values = None
        self._indices = None
        self.reset()

    def reset(self):
        self._values = np.empty(self.last, dtype=self.dtype)
        if self.keep_indices:
            self._indices = np.empty(self.last, dtype='l')
        self._full = False
        self._counter = 0

    def _add_hook(self, x, count):
        x /= count
        for _ in range(count):
            if self._counter == self.last:
                self._full = True
                self._counter = 0
            self._values[self._counter] = x
            if self.keep_indices:
                self._indices[self._counter] = self.count
            self._counter += 1

    def values(self):
        if self._full:
            return self._values
        return self._values[:self._counter]

    @property
    def mean(self):
        return self.values().mean()

    @property
    def var(self):
        return self.values().var()

    @property
    def n(self):
        return len(self.values())


def reset_each(aggregator, x, reset_each=10):
    if aggregator.count % reset_each == 0:
        aggregator.reset()
    return x


def in_range(aggregator, x, start=0, end=None, step=1):
    count = aggregator.count
    end = count + 1 if end is None else end
    if start < count < end and 0 == (count - start) % step:
        return x
    return None


def chain(aggregator, x, predicates=None):
    if predicates is None:
        predicates = ()
    for predicate in predicates:
        x = predicate(aggregator, x)
        if x is None:
            return None
    return x


# --- Resource watches

@contextlib.contextmanager
def watch_tic_toc(watch, count=1):
    watch.tic()
    yield
    watch.toc(count=count)


class Watch(object):

    # TODO: enforce number of calls to tictoc and number of units are always counted

    def __init__(self, unit=FrameUnit):
        super(Watch, self).__init__()
        self.unit = unit

    # --- Start and end measuring

    def tic(self):
        raise NotImplementedError()

    def toc(self, count=1):
        raise NotImplementedError()

    # --- Convert to list of tuples (measure, value)

    def to_record(self):
        raise NotImplementedError()

    # --- API conveniences

    def __enter__(self):
        return self.tic()

    def __exit__(self, *_):
        self.toc()

    def tictoc(self, count=1):
        return watch_tic_toc(self, count=count)

    def __call__(self, f, activate=True):

        if not activate:
            return f

        @wraps(f)
        def watched_call(*args, **kwargs):
            with self:
                return f(*args, **kwargs)
        return watched_call


class Rusager(Watch):
    """
    Iteratively accumulate systems resource usage for each execution of a process.

    In linux, from https://linux.die.net/man/2/getrusage
    ---------------------
    The resource usages are returned in the structure pointed to by usage, which has the following form:
    struct rusage {
        struct timeval ru_utime; /* user CPU time used */
        struct timeval ru_stime; /* system CPU time used */
        long   ru_maxrss;        /* maximum resident set size */
        long   ru_ixrss;         /* integral shared memory size */
        long   ru_idrss;         /* integral unshared data size */
        long   ru_isrss;         /* integral unshared stack size */
        long   ru_minflt;        /* page reclaims (soft page faults) */
        long   ru_majflt;        /* page faults (hard page faults) */
        long   ru_nswap;         /* swaps */
        long   ru_inblock;       /* block input operations */
        long   ru_oublock;       /* block output operations */
        long   ru_msgsnd;        /* IPC messages sent */
        long   ru_msgrcv;        /* IPC messages received */
        long   ru_nsignals;      /* signals received */
        long   ru_nvcsw;         /* voluntary context switches */
        long   ru_nivcsw;        /* involuntary context switches */
    };
    Not all fields are completed; unmaintained fields are set to zero by the kernel.
    (The unmaintained fields are provided for compatibility with other systems,
    and because they may one day be supported on Linux.)
    The fields are interpreted as follows:

    ru_utime
      This is the total amount of time spent executing in user mode,
      expressed in a timeval structure (seconds plus microseconds).

    ru_stime
      This is the total amount of time spent executing in kernel mode,
       expressed in a timeval structure (seconds plus microseconds).

    ru_maxrss (since Linux 2.6.32)
      This is the maximum resident set size used (in kilobytes).
      For RUSAGE_CHILDREN, this is the resident set size of the largest child,
      not the maximum resident set size of the process tree.

    ru_ixrss (unmaintained)
      This field is currently unused on Linux.

    ru_idrss (unmaintained)
      This field is currently unused on Linux.

    ru_isrss (unmaintained)
      This field is currently unused on Linux.

    ru_minflt
      The number of page faults serviced without any I/O activity;
      here I/O activity is avoided by "reclaiming" a page frame from
      the list of pages awaiting reallocation.

    ru_majflt
      The number of page faults serviced that required I/O activity.

    ru_nswap (unmaintained)
      This field is currently unused on Linux.

    ru_inblock (since Linux 2.6.22)
      The number of times the file system had to perform input.

    ru_oublock (since Linux 2.6.22)
      The number of times the file system had to perform output.

    ru_msgsnd (unmaintained)
      This field is currently unused on Linux.

    ru_msgrcv (unmaintained)
      This field is currently unused on Linux.

    ru_nsignals (unmaintained)
      This field is currently unused on Linux.

    ru_nvcsw (since Linux 2.6)
      The number of times a context switch resulted due to a process voluntarily
      giving up the processor before its time slice was completed (usually to await
      availability of a resource).

    ru_nivcsw (since Linux 2.6)
      The number of times a context switch resulted due to a higher priority process
      becoming runnable or because the current process exceeded its time slice.
    ---------------------

    See also:
      https://docs.python.org/2/library/resource.html#resource.getrusage
    """

    LINUX_UNMAINTAINED_FIELDS = (
        'ru_ixrss', 'ru_idrss', 'ru_isrss',
        'ru_nswap',
        'ru_msgsnd', 'ru_msgrcv', 'ru_nsignals'
    )

    LINUX_MAINTAINED_FIELDS = {
        'ru_utime': ('s',  'time in user mode'),
        'ru_stime': ('s', 'time in system mode'),

        'ru_maxrss': ('kB', 'maximum resident set size'),

        'ru_minflt': ('faults', 'page faults not requiring I/O'),
        'ru_majflt': ('faults', 'page faults requiring I/O'),

        'ru_inblock': ('operation', 'file system block input operations'),
        'ru_oublock': ('operation', 'file system block output operations'),

        'ru_nvcsw': ('switches', 'voluntary context switches'),
        'ru_nivcsw': ('switches', 'involuntary context switches'),
    }

    EXTREME_FIELDS = 'ru_maxrss',

    # Help IDE autocompletion
    ru_utime = None
    ru_stime = None
    ru_maxrss = None
    ru_minflt = None
    ru_majflt = None
    ru_inblock = None
    ru_oublock = None
    ru_nvcsw = None
    ru_nivcsw = None

    def __init__(self,
                 who=resource.RUSAGE_SELF,
                 fields=None,
                 aggregator=OnlineMeanStd,
                 unit=FrameUnit,
                 fail_if_not_in_linux=False):
        super(Rusager, self).__init__(unit=unit)

        # N.B. units can change if run somewhere other than linux, so that would warrant
        # different metadata for the measurements.
        if not sys.platform.startswith('linux'):
            if fail_if_not_in_linux:
                raise Exception('Only linux is supported by rusager at the moment')
            else:
                print('WARNING: Only linux is supported by rusager at the moment')

        self.who = who

        self._fields = fields
        if self._fields is None:
            self._fields = tuple(self.LINUX_MAINTAINED_FIELDS)

        for field in self._fields:
            units, comments = self.LINUX_MAINTAINED_FIELDS[field]
            setattr(self, field, aggregator(field, units=units, comments=comments))

        self.wall_watch = timer('wall_time', units='s')

        self._usage0 = None

    def tic(self):
        self._usage0 = resource.getrusage(self.who)
        self.wall_watch.tic()
        return self

    def toc(self, count=1):
        self.wall_watch.toc(count=count)
        usage = resource.getrusage(self.who)
        for field in self._fields:
            if field in self.EXTREME_FIELDS:
                value = getattr(usage, field)
            else:
                value = getattr(usage, field) - getattr(self._usage0, field)
            getattr(self, field).add(value, count=count)
        self._usage0 = None
        return self

    # --- Number of times we have polled

    @property
    def num_units(self):
        return self.wall_watch.n

    # --- Time

    @property
    def user_time(self):
        return self.ru_utime.total

    @property
    def user_time_mean_std(self):
        return self.ru_utime.mean, self.ru_utime.std

    @property
    def system_time(self):
        return self.ru_stime.total

    @property
    def system_time_mean_std(self):
        return self.ru_stime.mean, self.ru_stime.std

    @property
    def ru_time(self):
        return self.user_time + self.system_time

    @property
    def ru_time_mean_std(self):
        # Assume independency of user and system times
        u_mean, u_std = self.user_time_mean_std
        s_mean, s_std = self.user_time_mean_std
        return u_mean + s_mean, np.sqrt(u_std ** 2 + s_std ** 2)

    @property
    def wall_time(self):
        return self.wall_watch.total

    @property
    def wall_time_mean_std(self):
        return self.wall_watch.mean, self.wall_watch.std

    @property
    def ups_wall(self):
        return self.num_units / self.wall_time

    @property
    def ups_ru(self):
        return self.num_units / self.ru_time

    # --- RSS

    @property
    def max_rss(self):
        return self.ru_maxrss.max

    @property
    def max_rss_mean_std(self):
        return self.ru_maxrss.mean, self.ru_maxrss.std

    # --- Page faults

    @property
    def minor_faults(self):
        return self.ru_minflt.total

    @property
    def minor_faults_mean_std(self):
        return self.ru_minflt.mean, self.ru_minflt.std

    @property
    def major_faults(self):
        return self.ru_majflt.total

    @property
    def major_faults_mean_std(self):
        return self.ru_majflt.mean, self.ru_majflt.std

    # --- I/O blocking

    @property
    def io_in_block(self):
        return self.ru_inblock.total

    @property
    def io_in_block_mean_std(self):
        return self.ru_inblock.mean, self.ru_inblock.std

    @property
    def io_out_block(self):
        return self.ru_oublock.total

    @property
    def io_out_block_mean_std(self):
        return self.ru_oublock.mean, self.ru_oublock.std

    # --- Context switches

    @property
    def voluntary_switches(self):
        return self.ru_nvcsw.total

    @property
    def voluntary_switches_mean_std(self):
        return self.ru_nvcsw.mean, self.ru_nvcsw.std

    @property
    def involuntary_switches(self):
        return self.ru_nivcsw.total

    @property
    def involuntary_switches_mean_std(self):
        return self.ru_nivcsw.mean, self.ru_nivcsw.std

    def to_record(self):
        return (
            # Each op corresponds to a...
            ('unit', tuple(self.unit)),
            # Number of measurements
            ('num_%s' % self.unit.plural, self.num_units),
            # Time
            ('user_time_s', self.user_time),
            ('user_time_mean_s', self.user_time_mean_std[0]),
            ('user_time_std_s', self.user_time_mean_std[1]),
            ('system_time_s', self.system_time),
            ('system_time_mean_s', self.system_time_mean_std[0]),
            ('system_time_std_s', self.system_time_mean_std[1]),
            ('ru_time_s', self.ru_time),
            ('ru_time_mean_s', self.ru_time_mean_std[0]),
            ('ru_time_std_s', self.ru_time_mean_std[1]),
            ('wall_time_s', self.wall_time),
            ('wall_time_mean_s', self.wall_time_mean_std[0]),
            ('wall_time_std_s', self.wall_time_mean_std[1]),
            ('%sps_wall' % self.unit.symbol, self.ups_wall),
            ('%sps_ru' % self.unit.symbol, self.ups_ru),
            # MaxRSS
            ('max_rss', self.max_rss),
            ('max_rss_mean', self.max_rss_mean_std[0]),
            ('max_rss_std', self.max_rss_mean_std[1]),
            # Page faults
            ('minor_faults', self.minor_faults),
            ('minor_faults_mean', self.minor_faults_mean_std[0]),
            ('minor_faults_std', self.minor_faults_mean_std[1]),
            ('major_faults', self.major_faults),
            ('major_faults_mean', self.major_faults_mean_std[0]),
            ('major_faults_std', self.major_faults_mean_std[1]),
            # I/O
            ('io_in_block', self.io_in_block),
            ('io_in_block_mean', self.io_in_block_mean_std[0]),
            ('io_in_block_std', self.io_in_block_mean_std[1]),
            ('io_out_block', self.io_out_block),
            ('io_out_block_mean', self.io_out_block_mean_std[0]),
            ('io_out_block_std', self.io_out_block_mean_std[1]),
            # Switches
            ('voluntary_switches', self.voluntary_switches),
            ('voluntary_switches_mean', self.voluntary_switches_mean_std[0]),
            ('voluntary_switches_std', self.voluntary_switches_mean_std[1]),
            ('involuntary_switches', self.involuntary_switches),
            ('involuntary_switches_mean', self.involuntary_switches_mean_std[0]),
            ('involuntary_switches_std', self.involuntary_switches_mean_std[1]),
        )

    @staticmethod
    def clock_ticks_per_second():
        return os.sysconf('SC_CLK_TCK')

    @staticmethod
    def page_size():
        resource.getpagesize()


class PSUtilRusager(Watch):

    # For psutil.Process. See:
    #   http://psutil.readthedocs.io/en/latest/#psutil.Process.oneshot
    # For the moment we mimic Rusager fields, but we can access more info on the way.
    # In fact, we probably just switch to psutil for good

    def __init__(self, process=None, aggregator=OnlineMeanStd, unit=FrameUnit):
        super(PSUtilRusager, self).__init__(unit=unit)

        if not sys.platform.startswith('linux'):
            raise Exception('Only linux is supported by rusager at the moment')

        self.process = process

        self.wall_watch = timer('wall_time', units='s')

        self._usage0 = None

        self.ru_utime = aggregator('ru_utime', units='s', comments='time in user mode')
        self.ru_stime = aggregator('ru_stime', units='s', comments='time in system mode')
        self.ru_children_utime = aggregator('ru_children_utime', units='s', comments='time in user mode (children)')
        self.ru_children_stime = aggregator('ru_children_stime', units='s', comments='time in system mode (children)')

        self.ru_nvcsw = aggregator('ru_nvcsw', units='switches', comments='voluntary context switches')
        self.ru_nivcsw = aggregator('ru_nivcsw', units='switches', comments='involuntary context switches')

    def _usage(self):
        with self.process.oneshot():
            return (
                # pcputimes(user=6.06, system=0.41, children_user=0.0, children_system=0.0)
                self.process.cpu_times(),
                # pctxsw(voluntary=21009, involuntary=75)
                self.process.num_ctx_switches(),

                # This is specially slow
                # pfullmem(rss=26570752, vms=170106880, shared=11603968, text=237568,
                #          lib=0, data=88526848, dirty=0, uss=20709376, pss=22578176, swap=0)
                # self.process.memory_full_info()
                None
            )

    def tic(self):
        self._usage0 = self._usage()
        self.wall_watch.tic()
        return self

    def toc(self, count=1):
        self.wall_watch.toc()

        cpu_times0, context_switches0, full_mem0 = self._usage0
        cpu_times, context_switches, full_mem = self._usage()

        self.ru_utime.add(cpu_times.user - cpu_times0.user, count=count)
        self.ru_stime.add(cpu_times.system - cpu_times0.system, count=count)
        self.ru_children_utime.add(cpu_times.children_user - cpu_times0.children_user, count=count)
        self.ru_children_stime.add(cpu_times.children_system - cpu_times0.children_system, count=count)

        self.ru_nvcsw.add(context_switches.voluntary - context_switches0.voluntary, count=count)
        self.ru_nivcsw.add(context_switches.involuntary - context_switches0.involuntary, count=count)

        self._usage0 = None

        return self

    def to_record(self):
        raise NotImplementedError('Mimic Rusager implementation')

    def __getstate__(self):
        odict = self.__dict__
        del odict['process']
        return odict


class PlainProcessRusager(Watch):

    def __init__(self, process, aggregator=OnlineMeanStd):
        super(PlainProcessRusager, self).__init__()

        if not sys.platform.startswith('linux'):
            raise Exception('Only linux is supported by rusager at the moment')

        self.clock_ticks_per_second = os.sysconf('SC_CLK_TCK')

        self.process = process

        self.wall_watch = timer('wall_time', units='s')

        self._usage0 = None

        self.ru_utime = aggregator('ru_utime', units='s', comments='time in user mode')
        self.ru_stime = aggregator('ru_stime', units='s', comments='time in system mode')
        self.ru_children_utime = aggregator('ru_children_utime', units='s', comments='time in user mode (children)')
        self.ru_children_stime = aggregator('ru_children_stime', units='s', comments='time in system mode (children)')

    def _usage(self):
        # https://unix.stackexchange.com/questions/132035/obtain-user-and-kernel-time-of-a-running-process
        # https://linux.die.net/man/5/proc
        with open('/proc/%d/stat' % self.process.pid, 'rt') as reader:
            values = reader.read().split()
            return (
                float(values[14]) / self.clock_ticks_per_second,
                float(values[15]) / self.clock_ticks_per_second,
                float(values[16]) / self.clock_ticks_per_second,
                float(values[17]) / self.clock_ticks_per_second,
            )

    def tic(self):
        self._usage0 = self._usage()
        self.wall_watch.tic()
        return self

    def toc(self, count=1):
        self.wall_watch.toc(count=count)

        u0, s0, cu0, cs0 = self._usage0
        u, s, cu, cs = self._usage()

        self.ru_utime.add(u - u0, count=count)
        self.ru_stime.add(s - s0, count=count)
        self.ru_children_utime.add(cu - cu0, count=count)
        self.ru_children_stime.add(cs - cs0, count=count)

        self._usage0 = None

        return self

    def to_record(self):
        raise NotImplementedError('Mimic Rusager implementation')


class Timer(Watch):

    def __init__(self, aggregator, unit=FrameUnit):
        super(Timer, self).__init__(unit=unit)
        self._start = None
        self.aggregator = aggregator

    def tic(self):
        self._start = time.time()
        return self

    def toc(self, count=1):
        while self._start is None:  # FIXME: safeguard against asynchronous code copying before tic() finishes
                                    # The proper fix is to let this be thread-safe
            pass
        self.aggregator.add(time.time() - self._start, count=count)
        self._start = None
        return self

    @property
    def name(self):
        return self.aggregator.name

    @property
    def mean(self):
        return self.aggregator.mean

    @property
    def std(self):
        return self.aggregator.std

    @property
    def total(self):
        return self.aggregator.total

    @property
    def n(self):
        return self.aggregator.n

    @property
    def count(self):
        return self.aggregator.count

    @property
    def units(self):
        if self.aggregator.units:
            return self.aggregator.units + '/s'
        return None

    def __str__(self):
        # noinspection PyStringFormat
        msg = '%s: %.2f +/- %.2f' % (self.aggregator.name, self.mean, self.std)
        if self.units:
            return msg + ' ' + self.units
        return msg

    def to_record(self):
        raise NotImplementedError('Mimic Rusager implementation')


# --- Convenient API to group watches

class Timers(MutableMapping):

    #
    # Goals:
    #   - Allow dictionary + member like definition
    #     (helps avoiding namespace pollution)
    #   - Coordinate the collection (e.g. allow to add constraints like "these timers should be called in sequence")
    #   - Pretty printing
    #

    #
    # If all of a sudden the world would move to python 3.6,
    # with its ordered dicts, this would all of a sudden become
    # a much sleeker class.
    #

    def __init__(self, *timers):
        super(Timers, self).__init__()
        self.timers = OrderedDict()
        for timer in timers:
            if isinstance(timer, Timer):
                self.add(timer)
            else:
                name, timer = timer
                self.add(timer, name=name)

    def add(self, timer, name=None):
        if name is None:
            name = timer.name
        if name in self:
            raise KeyError('There is already a timer named %r' % name)
        self[name] = timer
        return self

    # --- Attributes magic (and danger)

    def __getattribute__(self, name):
        try:
            # noinspection PyCallByClass
            return object.__getattribute__(self, name)
        except AttributeError:
            return self[name]

    def __setattr__(self, name, value):
        if isinstance(value, Timer):
            self.add(value, name)
        else:
            # noinspection PyCallByClass
            object.__setattr__(self, name, value)

    # --- Make it look like a dict

    def __getitem__(self, key):
        return self.timers.__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, Timer):
            raise ValueError('Only Timer instances are allowed to be set')
        return self.timers.__setitem__(key, value)

    def __len__(self):
        return self.timers.__len__()

    def __iter__(self):
        return self.timers.__iter__()

    def __delitem__(self, key):
        return self.timers.__delitem__(key)

    # --- Pretty formatting

    def summary(self, precision=2):
        names = []
        totals = []
        means = []
        stds = []
        operations = []
        for name, timer in self.items():
            names.append(timer.name)
            totals.append('{total:.{precision}f}'.format(total=timer.total, precision=precision))
            means.append('{mean:.{precision}f}'.format(mean=timer.mean, precision=precision))
            stds.append('{std:.{precision}f}'.format(std=timer.std, precision=precision))
            operations.append('%d' % timer.n)

        def maxlen(l):
            return max(len(elem) for elem in l)

        longer_name = maxlen(names)
        longer_total = maxlen(totals)
        longer_mean = maxlen(means)
        longer_std = maxlen(stds)
        longer_op = maxlen(operations)

        summary = []
        for name, total, mean, std, op in zip(names, totals, means, stds, operations):
            summary.append(
                '{name}: {total} ({mean} +/- {std} s/op, {op} operations)'.format(
                    name=name.rjust(longer_name),
                    total=total.rjust(longer_total),
                    mean=mean.rjust(longer_mean),
                    std=std.rjust(longer_std),
                    op=op.rjust(longer_op)
                )
            )

        return summary

    def to_records(self, t=None):
        if t is None:
            t = time.time()

        return [{'t': t, 'name': timer.name, 'operations': timer.n, 'units': timer.units,
                 'total': timer.total, 'mean': timer.mean, 'std': timer.std} for timer in self.values()]


# --- Convenience interface to create aggregators and timers

def meaner(name, units=None, add_filter=None,
           dtype='f',
           keep_values=False,
           keep_indices=False,
           only_last=None,
           as_timer=True):
    if not keep_indices and not keep_values and only_last is None:
        ms = OnlineMeanStd(name, units, add_filter=add_filter)
    elif only_last is not None:
        ms = KeepLastMeanStd(name, units, add_filter=add_filter,
                             last=only_last, dtype=dtype, keep_indices=keep_indices)
    else:
        ms = KeepAllMeanStd(name, units, add_filter=add_filter,
                            dtype=dtype, keep_indices=keep_indices)
    if not as_timer:
        return ms
    return Timer(ms)


timer = meaner


# TODO: bring timed iterator

#
# if __name__ == '__main__':
#     a_timer = timer('one', units='gb')
#     for _ in range(10):
#         with a_timer:
#             time.sleep(0.05)
#     print(a_timer.mean, a_timer.std, a_timer.n, a_timer.count)
#     print(a_timer)
#
#     timers = Timers(a_timer)
#
#     print(timers.one)
#     timers.mb = timer('two')
#     timers.ll = 3
#     print(str(timers['mb']), timers.ll)
#     timers['kk'] = timer('three')
#     print(timers.kk)
#
#     import pandas as pd
#     df = pd.DataFrame(timers.to_records())
#     print(df)
#
# TODO: largest (or smallest) n mean & std, use a heap; useful for upperbound timings
# TODO: allow timers to tic toc without increasing n and count
#       useful to, for example, temporarily stop timing when other times gets in
#       and resume again afterwards
#       it could also be implemented by Timer composition
# TODO: implement function decorator for Timers
# TODO: add percentage to summary, specify there which one is total time
# TODO: threaded poling of process GPU resources usage
#       (it is probably quite useless to do it when the session is not running)
#
