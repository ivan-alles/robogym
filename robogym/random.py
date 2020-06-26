#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

""" Random numbers. """

import hashlib

MAX_HASH_PLUS_ONE = 2**(hashlib.sha512().digest_size * 8)


def str_to_random(in_str):
    """
    Converts a string to a reproducible uniformly random float in the interval [0, 1).
    """
    seed = in_str.encode()
    hash_digest = hashlib.sha512(seed).digest()
    # Uses explicit byteorder for system-agnostic reproducibility
    hash_int = int.from_bytes(hash_digest, 'big')
    return hash_int / MAX_HASH_PLUS_ONE
