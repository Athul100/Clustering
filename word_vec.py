import gensim
import re


class Word2Vec:
    def __init__(self):
        print('loading model')
        self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300-SLIM.bin.gz',
                                                                     binary=True)
        print('loaded model')


class SpellCorrect:
    def __init__(self, word_vec_model):
        print('loading model')
        self.model = word_vec_model
        print('loaded model')

        print('create list of words')
        words = self.model.index2word
        w_rank = {}
        for i, word in enumerate(words):
            w_rank[word] = i

        self.WORDS = w_rank

    def words(self, text):
        return re.findall(r'\w+', text.lower())

    def P(self, word):
        """Probability of `word."""
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.WORDS.get(word, 0)

    def correction(self, word):
        """Most probable spelling correction for word."""
        return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        """Generate possible spelling corrections for word."""
        return self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word]

    def known(self, words):
        """The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        """All edits that are one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


if __name__ == '__main__':
    print('test')
    # print(correction('quikly'))
    # text = 'freekeith forgetus btruetolife truepg daentertainah knickanator donaldp bronxprodigyzzz ' \
    #        'jerrelxl trace avp bigfreezie ajthemanchild ant hatemgr knicks woodshed rahmmagick lcamarketing ' \
    #               'net steven lukehenderson khaleel goknickstape cillejones sivjacobsnyc nyc sportz delblogo rtaylor' \
    #               ' kerlond p diazny adv celenaali kingboney fitlyfeapparel hill inc kashcoop shamiek gmbcashout' \
    #               ' mweshler oluwaswank shwinnypooh dayviddee forevershinin ttoo fredodgawd garthgriffen kristaps' \
    #               ' szn knicksguy hardknickslife koton knickfilmschool ever earns starting point guard hope earn ' \
    #               ' szn knicksguy hardknickslife koton knickfilmschool ever earns starting point guard hope earn ' \
    #               'support sharing opinions'
    #
    # corrected_text = ''
    # for each_word in text.split():
    #     corrected_text = corrected_text + ' ' + correction(each_word)
    #
    # print(corrected_text)