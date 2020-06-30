#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

"""
Error detection and correction algorithms.
"""


class HammingErrorCorrectionCodec:
    """
    Implement Hamming algorithm.

    See: https://en.wikipedia.org/wiki/Hamming_code
    """
    def __init__(self):
        pass

    def encode(self, data):
        """
        Encode data.
        :param data: list of bits
        """
        parity_len = self.get_number_of_parity_bits_for_data(len(data))
        code = [0] * (len(data) + parity_len)
        data_pos, parity_pos = self._get_bit_positions(len(code))

        self._copy_bits_to_positions(data, code, data_pos)

        parity, _ = self._compute_parity(code, parity_len)

        return data + parity

    def decode(self, code):
        """
        Decode data.
        :param code: list of bits.
        Can always correct 1-bit errors. Can detect some N-bit errors, if the parity bits are redundant.
        :return: corrected data as list of 0, 1 or None if there is an N bit error.
        """
        data_pos, parity_pos = self._get_bit_positions(len(code))
        parity_len = len(parity_pos)

        # Move parity and data bits to the original position as described in the algo.
        code_orig = [0] * len(code)
        self._copy_bits_to_positions(code[:-parity_len], code_orig, data_pos)
        self._copy_bits_to_positions(code[-parity_len:], code_orig, parity_pos)

        _, parity = self._compute_parity(code_orig, parity_len)
        if parity != 0:
            # There are errors.
            if parity - 1 < len(code_orig):
                code_orig[parity - 1] = 1 - code_orig[parity - 1]
            else:
                # We are trying to correct a padding bit.
                # This is an indication of a 2-bit error.
                return None

        data = [0] * len(data_pos)
        for i in range(len(data_pos)):
            data[i] = code_orig[data_pos[i]]

        return data

    def _get_bit_positions(self, code_len):
        """
        Compute the positions of parity and data bits.
        :param code_len: total length of the code (data + parity).
        :return: a tuple of lists (data_positions, parity_positions).
        """

        data_positions = []
        parity_positions = []
        p = 0
        d = 0
        for i in range(code_len):
            if i + 1 == 2 ** p:
                parity_positions.append(i)
                p += 1
            else:
                data_positions.append(i)
                d += 1

        return (data_positions, parity_positions)

    def get_number_of_parity_bits_for_data(self, data_len):
        m = 0
        while 2 ** m - m - 1 < data_len:
            m += 1
        return m

    def _get_number_of_parity_bits_for_code(self, code_len):
        m = 0
        while 2 ** m - 1 < code_len:
            m += 1
        return m

    def _copy_bits_to_positions(self, src, tgt, positions):
        for i in range(len(positions)):
            tgt[positions[i]] = src[i]

    def _compute_parity(self, code, parity_len):
        parity = 0
        for i in range(len(code)):
            parity ^= (i + 1) * code[i]
        # Convert number to a list containing 0 or 1.
        parity_list = [1 if parity & (2 ** i) else 0 for i in range(parity_len)]
        return parity_list, parity
