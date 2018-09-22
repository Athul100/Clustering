
import re
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory


def pre_process(text, spell_correction_model, word_segment):
    text = clean(text, spell_correction_model, word_segment)
    return text


def clean(text, spell_correction_model, word_segment):
    lang = ''
    try:
        DetectorFactory.seed = 0
        lang = str(detect(text))
    except:
        pass
    if lang != 'en':
        print(lang)
        return None

    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    letter_only = re.sub('[^a-zA-Z]', ' ', text)

    print('Before Spell Correction', letter_only)
    spell_corrected_text = ''
    for word in letter_only.split():
        print('Spelling 1')
        spell_correction_model.correction(word)
        print('Segmenting')
        corrected_text_array = word_segment.segment(spell_correction_model.correction(word))

        arr = []
        print('Spelling 2')
        if len(corrected_text_array)>1:
            for each in corrected_text_array:
                arr.append(spell_correction_model.correction(each))
        else:
            arr = corrected_text_array

        spell_corrected_text = spell_corrected_text + ' ' + ' '.join(arr)

    print('After Spell Correction', spell_corrected_text)
    words = spell_corrected_text.lower().split()
    stopwords_eng = set(stopwords.words("english"))
    useful_words = [x for x in words if x not in stopwords_eng]

    # Combine words into a paragraph again
    useful_words_string = ' '.join(useful_words)
    return useful_words_string

