import functools
import math
import sys
import gzip


class WordSegment:
    def __init__(self):
        onegrams = OneGramDist(filename='1gram.txt')
        self.word_seq_fitness = functools.partial(onegram_log, onegrams)

    def segment(self, word):
        if not word or not self.word_seq_fitness:
            return []
        all_segmentations = [[first] + self.segment(rest)
                             for (first, rest) in split_pairs(word)]
        # print(len(all_segmentations))
        return max(all_segmentations, key=self.word_seq_fitness)


class OneGramDist(dict):
    """
    1-gram probability distribution for corpora.
    """
    def __init__(self, filename='count_1w_cleaned.txt'):
        self.total = 0
        print('building probability table...')
        _open = open
        if filename.endswith('gz'):
            _open = gzip.open
        with _open(filename) as handle:
            for line in handle:
                word, count = line.strip().split('\t')
                self[word] = int(count)
                self.total += int(count)

    def __call__(self, word):
        try:
            result = float(self[word]) / self.total
            # print(word, result)
        except KeyError:
            # result = 1.0 / self.total
            return 1.0 / (self.total * 10**(len(word) - 2))
            # return sys.float_info.min
        return result


def onegram_log(onegrams, words):
    """
    Use the log trick to avoid tiny quantities.
    http://machineintelligence.tumblr.com/post/4998477107/the-log-sum-exp-trick
    """

    result = functools.reduce(lambda x, y: x + y, (math.log10(onegrams(w)) for w in words))

    return result


def split_pairs(word):
    return [(word[:i + 1], word[i + 1:]) for i in range(len(word))]


if __name__ == '__main__':
    sentence = ''.join(sys.argv[1:]).lower()
    if not sentence:
        sentence = 'sharing'
    print(sentence)
    print(WordSegment().segment(sentence))