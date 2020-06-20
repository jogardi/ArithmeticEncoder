import unittest
import random
from arithmetic_code import *


def make_rand_interval():
    return make_rand_subinterval(Interval(0, 1))


def make_rand_subinterval(interval):
    start = random.uniform(interval.start, interval.end)
    return Interval(start, random.uniform(start, interval.end))


class ArithmeticCodeTest(unittest.TestCase):

    def check_txt(self, txt):
        try:
            compressed = compress(txt)
            self.assertEqual(decompress(compressed), txt,
                             f'did not work for {txt} which compressed to {compressed}')
        except Exception as e:
            print(f"got an error when compressing and decompressing {txt}")
            raise

    def test(self):
        second_quarter = Interval(.25, .5)
        self.assertEqual(second_quarter.to_bits(), '01')
        self.assertEqual(Interval.from_bits('01'), second_quarter)

        self.check_txt('')

        for letter in alphabet:
            compressed = compress(letter)
            length = len(compressed)
            self.assertTrue((2 ** -length <= 1 / len(alphabet) <= 2 ** -(length - 2)),
                            f'compressed txt has the wrong length. The letter {letter} was compressed to {compressed} '
                            f'which has a length of {length}.')
            self.assertEqual(decompress(compressed), letter,
                             f'could not decompress the letter {letter} which compressed to {compressed}')

        for _ in range(50):
            txt = ''.join([random.choice(alphabet) for _ in range(random.randint(2, 4000))])
            print('test')
            self.check_txt(txt)

            bits = ''.join([random.choice(['0', '1']) for _ in range(random.randint(2, 40))])
            self.assertEqual(bits, Interval.from_bits(bits).to_bits())

            interval = make_rand_interval()
            interval_after_conversion = Interval.from_bits(interval.to_bits())
            # bits can't represent an interval with perfect accuracy but
            # the resulting interview should at least be inside the original interval
            self.assertTrue(interval.contains(interval_after_conversion))

            bits_for_interval, subinterval = interval.as_bits_and_subinterval()
            self.assertEqual(interval, Interval.from_bits(bits_for_interval).create_from_subinterval(subinterval))

            # check property in pydoc of extract_sub_interval
            other_interval = make_rand_subinterval(interval)
            x = interval.extract_sub_interval(other_interval)
            reconstructed_interval = interval.create_from_subinterval(x)
            self.assertTrue(abs(other_interval.start - reconstructed_interval.start) <= .00001 and
                            abs(other_interval.end - reconstructed_interval.end) <= .00001,
                            f'interval: {interval}, other_interval: {other_interval}, x: {x}')
