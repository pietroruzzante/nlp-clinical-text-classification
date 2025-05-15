import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import nltk
nltk.download('wordnet')


def clean_text(text):
    #crea un testo in minuscolo, senza punteggiatura, numeri, e simboli specifici.
    text = text.translate(str.maketrans('', '', string.punctuation)) #rimuove punteggiatura
    text1 = ''.join([w for w in text if not w.isdigit()]) #remove numbers
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    # BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    text2 = text1.lower() # minuscolo
    text2 = REPLACE_BY_SPACE_RE.sub('', text2)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    # text2 = BAD_SYMBOLS_RE.sub('', text2)
    return text2


def lemmatize_text(text):
    wordlist = []
    lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(text) #tokenize text

    #intial_sentences = sentences[0:1]
    #final_sentences = sentences[len(sentences) - 2: len(sentences) - 1]

    # for sentence in intial_sentences:
    #     words = word_tokenize(sentence)
    #     for word in words:
    #         wordlist.append(lemmatizer.lemmatize(word))
    # for sentence in final_sentences:
    #     words = word_tokenize(sentence)
    #     for word in words:
    #         wordlist.append(lemmatizer.lemmatize(word))

    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    return ' '.join(wordlist)