#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import numpy as np

from robogym import error_corrrection


class HammingErrorCorrectionCodecTest:

    def test_simple(self):

        codec = error_corrrection.HammingErrorCorrectionCodec()

        data = [1]
        code = codec.encode(data)
        assert code == [1, 1, 1]
        data1 = codec.decode(code)
        assert data1 == data
        data1 = codec.decode([0, 1, 1])
        assert data1 == data
        data1 = codec.decode([1, 0, 1])
        assert data1 == data
        data1 = codec.decode([1, 1, 0])
        assert data1 == data

        data = [0]
        code = codec.encode(data)
        assert code == [0, 0, 0]
        data1 = codec.decode(code)
        assert data1 == data
        data1 = codec.decode([1, 0, 0])
        assert data1 == data
        data1 = codec.decode([0, 1, 0])
        assert data1 == data
        data1 = codec.decode([0, 0, 1])
        assert data1 == data

    def test_padding(self):

        codec = error_corrrection.HammingErrorCorrectionCodec()

        # The data should be internally padded with 2 zeros,
        # but the returned code will not contain padding.
        data = [1, 1]
        code = codec.encode(data)
        assert len(code) == 5
        data1 = codec.decode(code)
        assert data1 == data

        # Make 2-bit error.
        code[0] = 0
        code[1] = 0
        data1 = codec.decode(code)
        assert data1 is None

    def test_random(self):
        codec = error_corrrection.HammingErrorCorrectionCodec()

        rng = np.random.RandomState(1)

        for i in range(5000):
            data_len = rng.randint(1, 70)
            data = [rng.randint(0, 2) for x in range(data_len)]
            code = codec.encode(data)
            decoded_data = codec.decode(code)
            assert decoded_data == data
            # Alter 1 bit
            p = rng.randint(len(code))
            code[p] = 1 - code[p]
            decoded_data = codec.decode(code)
            assert decoded_data == data

            # Optionally alter more bits, make sure we do not crash.
            count = rng.randint(data_len)
            for a in range(count):
                p = rng.randint(len(code))
                code[p] = 1 - code[p]
                decoded_data = codec.decode(code)
                assert decoded_data is None or len(decoded_data) == data_len
