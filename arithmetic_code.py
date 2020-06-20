
from collections import defaultdict
from math import *

alphabet = [chr(ascii_num) for ascii_num in range(ord('a'), ord('z') + 1)]


class Model:

    def __init__(self):
        self.counts = defaultdict(lambda: 0)
        self.total_length = 0

    def add_char(self, new_char):
        self.counts[new_char] += 1
        self.total_length += 1

    def probs(self):
        kernels_for_letters = {}
        for letter in alphabet:
            count = self.counts[letter]
            kernels_for_letters[letter] = (count + 1) / (self.total_length + len(alphabet))

        normalizing_constant = sum(kernels_for_letters.values())
        return {letter: kernel / normalizing_constant for letter, kernel in kernels_for_letters.items()}


class ProbCalculator:

    prob_below = 0
    alphabet_index = 0

    def __init__(self, model):
        self.probs_for_letters = model.probs()

    def next_letter_with_interval(self):
        next_letter = alphabet[self.alphabet_index]
        prob_for_letter = self.probs_for_letters[next_letter]
        end = self.prob_below + prob_for_letter
        # end will sometimes be slightly more than 1 because of numerical error
        # in that case we just truncate to 1
        assert end - 1 < .000000001
        interval = Interval(self.prob_below, min(1, end))
        self.prob_below += prob_for_letter

        self.alphabet_index += 1

        return next_letter, interval

    def has_next(self):
        return self.alphabet_index < len(alphabet)


class Interval(object):
    """
    Represents the interval from P(char < x) to P(char <= x) where x is some letter in the alphabet
    It doesn't matter what ordering you use to compare char and x as long as you are consistent
    """

    def __init__(self, start, end):
        assert 0 <= start <= end <= 1
        self.start = start
        self.end = end

    def to_bits(self):
        """
        there is no way to encode the interval exactly so that
        bits_to_interval(interval_to_bits(interval)) == bits
        so I will make a bit string x such that self.contains(Interval.from_bits(x))
        """

        bits, subinterval = self.as_bits_and_subinterval()
        if subinterval.size == 1:
            # awesome! the interval is represented by bits with perfect accuracy
            return bits
        # We're in this awkward position where
        # subinterval.start < .5 and subinterval.end > .5. It won't be possible
        # to make bits that represent self with perfect accuracy.
        # So in many cases what bits we return are partially arbitrary.
        # it will work as long as the interval for the returned bits are entirely contained by
        # self

        if subinterval.end - .5 > .5 - subinterval.start:
            bits += '1'
            # subtract 1 to account for the bit we just added
            num_bits_left_to_add = ceil(-log(subinterval.end - .5, 2)) - 1
            bits += num_bits_left_to_add * '0'
        else:
            bits += '0'
            i = 2
            val_of_added_bits = 0
            while val_of_added_bits < subinterval.start:
                val_of_added_bits += 2 ** -i
                bits += '1'

                i += 1

        return bits

    def contains(self, other_interval):
        return self.start <= other_interval.start and self.end >= other_interval.end

    def as_bits_and_subinterval(self):
        """
        If the returned tuple is (bits, interval)
        then Interval.from_bits(bits).create_from_subinterval(interval) will be equal to self.
        Therefore the returned interval is an equivalent representation of self.
        This function is useful because the computer has limited precision.
         We will have serious problems if the interval size is less that the smallest number
         that can be stored.
        As a result storing an interval as one interval that continually gets smaller may result
        in round off errors. So this function will try to maximize the length of bits and
        the size of interval in order to minimize the chance of round off errors.
        :return: A tuple with bits and an interval
        """
        interval = Interval(self.start, self.end)
        bits = ''
        while True:
            if interval.start >= .5:
                bits += '1'
                interval = interval_for_1_bit.extract_sub_interval(interval)
            elif interval.end <= .5:
                bits += '0'
                interval = interval_for_0_bit.extract_sub_interval(interval)
            else:
                break

        return bits, interval

    @staticmethod
    def from_bits(bits):
        start = 0
        for i in range(len(bits)):
            start += int(bits[i]) * 2 ** -(i + 1)

        return Interval(start, start + 2 ** -len(bits))

    def create_from_subinterval(self, other_interval):
        sub_interval_start = self.start + self.size * other_interval.start
        sub_interval_end = sub_interval_start + (self.size * other_interval.size)
        return Interval(sub_interval_start, sub_interval_end)

    def extract_sub_interval(self, other_interval):
        """
        this is inverse of create_from_sub_interval. By that it returns an interval x such that
        self.create_from_sub_interval(x) == other_interval
        """
        assert self.contains(other_interval)
        new_start = (other_interval.start - self.start) / self.size
        new_end = new_start + other_interval.size / self.size
        return Interval(new_start, new_end)

    @property
    def size(self):
        return self.end - self.start

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __str__(self):
        return f'{{ start: {self.start}, end: {self.end} }}'


interval_for_1_bit = Interval.from_bits('1')
interval_for_0_bit = Interval.from_bits('0')


def compress(txt):
    model = Model()
    # we will converge on the interval for txt. The interval for txt
    # is represented with the bits in result and sub_interval which is a sub interval
    # of the interval represented by result.
    # we can't just represent the interval for txt with one normal interval because
    # the computer has limited precision. On the other hand we can't represent the interval
    # with just the bits in result because you can't write a bit when interval is not
    # completely contained by (0, .5) or (.5, 1). So we need the combination of both
    result = ''
    sub_interval = Interval(0, 1)
    for i in range(len(txt)):
        interval_for_next_char = interval_given_txt(txt[i], model)
        sub_interval = sub_interval.create_from_subinterval(interval_for_next_char)
        new_bits, sub_interval = sub_interval.as_bits_and_subinterval()
        result += new_bits
        model.add_char(txt[i])

    return result + sub_interval.to_bits()


def interval_given_txt(next_char, model):
    """
    returns an interval within (0, 1) that corresponds to the P (next_char | preceding text)
    The preceding text is not passed in explicitly but the model has all of the preceding text
    because model.add_char should have been called for each preceding character in the text
    """
    prob_calc = ProbCalculator(model)

    while prob_calc.has_next():
        letter, interval = prob_calc.next_letter_with_interval()
        if letter == next_char:
            return interval

    raise Exception(f'did not find next_char which is {next_char}')


def decompress(bits):
    # mirrors compress. Just does the reverse. See comments in compress to understand why
    # we represent the interval for txt with a combination of result and sub_interval
    model = Model()
    result = ''
    sub_interval = Interval(0, 1)
    interval_for_remaining_bits = Interval.from_bits(bits)

    # Idk how long the symbol will be but I know the first symbol start at index 0
    symbol_start = 0
    should_continue = True
    while should_continue:
        should_continue = False
        prob_calc = ProbCalculator(model)
        while prob_calc.has_next():
            letter, interval_for_letter = prob_calc.next_letter_with_interval()
            subinterval_for_letter = sub_interval.create_from_subinterval(interval_for_letter)
            if subinterval_for_letter.contains(interval_for_remaining_bits):
                new_bits, sub_interval = subinterval_for_letter.as_bits_and_subinterval()
                symbol_start += len(new_bits)
                interval_for_remaining_bits = Interval.from_bits(bits[symbol_start:])
                result += letter
                model.add_char(letter)
                should_continue = True
                break

    return result
