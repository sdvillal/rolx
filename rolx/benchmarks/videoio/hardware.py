import os.path as op
from rolx.utils import hostname, ensure_writable_dir


class Location(object):

    def __init__(self, name, host, location_type, path, comments=None, theoretical_peak_read=None, **extra):
        super(Location, self).__init__()
        self.name = name
        self.host = host
        self.type = location_type
        self.path = path
        self.comments = comments
        self.theoretical_peak_read = theoretical_peak_read
        self.meta = extra

    def is_available(self):
        return hostname() == self.host and op.exists(self.path)

    def ensure_under(self, name='videoio-benchmarks'):
        return ensure_writable_dir(self.path, name)

    def __str__(self):
        return '%s_%s' % (self.host, self.path)


BENCHMARK_DATA_LOCATIONS = (
    Location(name='snowy-ssd', host='snowy', location_type='ssd', path='/mnt/850pro'),
    # Location(name='snowy-ram', host='snowy', location_type='ram', path='/tmp'),
    Location(name='mumbler-hdd', host='mumbler', location_type='hdd', path='/mnt/rumbler'),
    Location(name='mumbler-ssd', host='mumbler', location_type='ssd', path='/home/santi'),
    # Location(name='mumbler-ram', host='mumbler', location_type='ram', path='/tmp'),
    Location(name='loopg-nfs', host='loopg', location_type='nfs', path='/mnt/loopbio/santi')
)
