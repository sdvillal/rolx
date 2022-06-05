# coding=utf-8
"""Frame retrieval sanity checking."""
from __future__ import print_function, division

from joblib.hashing import NumpyHasher
from xxhash import xxh64, xxh32


# --- Hashing video frames
# This is a way to create expectations when checking seeking in videos we do not rewrite

class XXHasher(NumpyHasher):

    #
    # Nasty Q&D, abuse private API and impl. details injection of xxhash
    # in joblig hashers. See also:
    #   https://github.com/joblib/joblib/issues/343
    #
    # Should just extract all that logic to get
    # a proper array hash + convert to bytes (class, dtype, shape, strides).
    # In my laptop, this is around 7% slower that just hashing
    # the buffer and around 7x faster than md5.
    #

    #
    # On a quick for fun  comparison of numpy array hashing in my laptop:
    #
    #  - Since here we are only concerned with images, which we could
    #    assume always have the same memory layout + (class, dtype, shape, strides),
    #    we could just go ahead and hash directly the buffer.
    #
    #  - Overhead added by joblib pickling approach is minimal if data does not need
    #    to be rearranged, and needed for corectness if data does need to be rearranged.
    #    In other words, it forces equal memory layout for logically equal arrays,
    #    and then differentiates by (array class, dtype, shape, strides).
    #    Note: ask John if he really intends to have the same hash for X and X.T.
    #
    #  - As expected, for speed, crc32 > md5 >> sha256
    #    xxhash is about an order of magnitude faster than crc32
    #    and small hashes should be more than enough for our use case
    #    (more akin to a checksum than anything else).
    #
    #  - whenever watermarking is an option, go for it; John's barcodes can be orders
    #    of magnitude faster if selecting well the barcode size (usually it should be)
    #    fine a small barcode, specially if the image is not going to be subsequently
    #    resized.
    #
    #  - probably all that does not matter much, as hashing speed is irrelevant
    #

    def __init__(self, use64=True, seed=0, coerce_mmap=False):
        NumpyHasher.__init__(self, 'md5', coerce_mmap)
        self._hash = xxh64(seed=seed) if use64 else xxh32(seed=seed)

    def inthash(self, obj):
        NumpyHasher.hash(self, obj, return_digest=False)
        return self._hash.intdigest()

    @staticmethod
    def xxhash(image, use64=True, seed=0, coerce_mmap=False, hexdigest=False):
        hasher = XXHasher(use64=use64, seed=seed, coerce_mmap=coerce_mmap)
        if hexdigest:
            return hasher.hash(image, return_digest=True)
        return hasher.inthash(image)


def hash_frames(trusted_video_reader, frame_numbers=None,
                image_hasher=XXHasher.xxhash,
                fail_if_error=True):
    # Assume *no seeking* is always correct
    # Otherwise, abstract the video reader here and use another trustworthy lib to read
    # (or generate the videos with known ground truth, see e.g. codebar from J)
    # FIXME: finish this adapting this

    if frame_numbers is None:
        frame_numbers = range(len(trusted_video_reader))
    else:
        # filter out-of-video frames (maybe we should just raise)
        max_frame_num = len(trusted_video_reader) - 1
        frame_numbers = [fn for fn in frame_numbers if 0 <= fn <= max_frame_num]

    frame_hashes = {}
    frame_numbers = set(frame_numbers)

    for frame_num in range(max(frame_numbers)):
        try:
            _, image = trusted_video_reader[frame_num]
        except Exception as ex:
            if not fail_if_error:
                # These lying frame count estimations...
                print('WARNING: could not read all frames')
                print(str(ex))
                print('Any other frame will be ignored')
                break
            else:
                raise
        if frame_num in frame_numbers:
            frame_hashes[frame_num] = image_hasher(image)

    return frame_hashes


# --- Rendered barcodes
# When we can watermark a video, this is a more robust and faster option than rendering frame number + OCR
# We probably should extract the barcode functions from loopb

class Barcoder(object):
    """Adds or checks a frame number barcode / watermark to the image."""

    # See also bview.capture.synth.SynthCaptureFramecode

    def __init__(self, nbits=16, barcode_size=64, inplace=False):
        super(Barcoder, self).__init__()
        self.nbits = nbits
        self.barcode_size = barcode_size
        self.inplace = inplace

    def encode(self, image, number):
        from bview.capture.synth import encode_number_barcode_image
        return encode_number_barcode_image(num=number,
                                           nbits=self.nbits,
                                           imgsize=self.barcode_size,
                                           base=image,
                                           base_copy=not self.inplace)

    def decode(self, image):
        from bview.capture.synth import decode_number_barcode_image
        return decode_number_barcode_image(image, nbits=self.nbits, imgsize=self.barcode_size)
