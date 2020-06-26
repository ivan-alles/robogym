#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

from robogym.random import *

def test_str_to_random():
    r1 = str_to_random('s1')
    r2 = str_to_random('s2')
    r3 = str_to_random('s1')

    assert 0 <= r1 and r1 < 1
    assert 0 <= r2 and r2 < 1
    assert r1 != r2
    assert r1 == r3
