#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

from robogym.image import *

def test_normalize_intensity():
    def check(input, min, max, k, g, c, b, expected_output):
        input = np.array(input)
        # Case 1: out == None
        output = normalize_intensity(input, min, max, k, g, c, b, out=None)
        assert np.allclose(output, expected_output)

        # Case 1: out != input
        out = np.zeros_like(expected_output)
        output = normalize_intensity(input, min, max, k, g, c, b, out=out)
        assert np.allclose(output, expected_output)
        assert out is output
        assert out is not input

        # Case 2: out == input
        output = normalize_intensity(input, min, max, k, g, c, b, out=input)
        assert np.allclose(output, expected_output)
        assert output is input

    check([1, 2, -1, 10], None, None, None, None, None, None, [1, 2, -1, 10])
    check([1, 2, -1, 10], 0, None, None, None, None, None, [1, 2, 0, 10])
    check([1, 2, -1, 10], None, 5, None, None, None, None, [1, 2, -1, 5])
    check([1, 2, -1, 10], 0, 5, None, None, None, None, [1, 2, 0, 5])
    check([1, 2, -1, 10], 0, 5, 2, None, None, None, [2, 4, 0, 10])
    check([1, 2, -1, 10], 0, 5, None, 2, None, None, [1, 4, 0, 25])
    check([1, 2, -1, 10], 0, 5, None, None, 3, None, [3, 6, 0, 15])
    check([1, 2, -1, 10], 0, 5, None, None, None, 4, [5, 6, 4, 9])
    check([1, 2, -1, 10], 0, 5, 3, 2, 4, -1, [35, 143, -1, 899])
