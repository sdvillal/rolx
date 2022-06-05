import os.path as op
import numpy as np
import pytest

from rolx.benchmarks.jpegs.imgutils import phash_compare
from rolx.benchmarks.jpegs.turbojpeg import TurboJPEG, TJPF, TJSAMP


def test_turbo():
    img = np.load(op.join(op.dirname(__file__), 'data', 'mona_lisa.npy'))
    turbo = TurboJPEG()

    encoded = turbo.encode(img, quality=95, pixel_format=TJPF.BGR, jpeg_subsample=TJSAMP.YUV420)
    assert len(img.data) > len(encoded)
    assert encoded == turbo.encode(img, quality=95, pixel_format=TJPF.BGR, jpeg_subsample=TJSAMP.YUV420)
    assert turbo.info(encoded) == (341, 229, 'YUV420', 'BGR')

    decoded = turbo.decode(encoded)
    np.testing.assert_equal(decoded, turbo.decode(encoded))
    assert not np.array_equal(decoded, turbo.decode(encoded, fast_dct=True, fast_upsample=False))
    assert not np.array_equal(decoded, turbo.decode(encoded, fast_dct=False, fast_upsample=True))
    assert not np.array_equal(decoded, turbo.decode(encoded, fast_dct=True, fast_upsample=True))
    assert phash_compare(img, decoded) <= 5


if __name__ == '__main__':
    pytest.main()
