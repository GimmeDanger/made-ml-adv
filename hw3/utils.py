import numpy as np
from nltk import ngrams
from collections import Counter

np.random.seed(1024)

ru_alpha = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
assert len(ru_alpha) == 33 + 1


def clean_text(text, alphabet=ru_alpha):
    '''
    Очистка текста: только русский алфавит, lowercase, без пунктуации
    '''
    text = text.lower()
    text = ''.join([c for c in text if c in alphabet])
    text = ' '.join(text.split())
    return text


def get_corpus():
    '''
    Загружаем тексты, сливаем их в один, очищаем
    '''
    text = ''
    for file_name in ['data/AnnaKarenina.txt', 'data/WarAndPeace.txt']: 
        with open(file_name, 'r') as f:
            text += f.read()
    return clean_text(text)

def accuracy(text1, text2):
    """
    Посимвольная точность расшифровки
    """
    assert len(text1) == len(text2)
    matching_chars = sum((c1 == c2) for c1, c2 in zip(text1, text2))
    return matching_chars / len(text1)


def encode_mapping(freqs):
    """
    Создаем маппинг случайной перестановки символов текста
    """
    original_ngrams = list(freqs.keys())
    permutated_ngrams = np.random.permutation(original_ngrams)
    mapping = dict(zip(original_ngrams, permutated_ngrams))
    return mapping


def apply_mapping(text, mapping):
    """
    Применяем маппинг для шифровки / расшифровки текста
    """
    return "".join([mapping.get(char, '*') for char in text])


def ngram_freq_dict(text, n_gram):
    '''
    Считаем частоты символов в корпусе
    '''
    if n_gram > 1:
        text = ["".join(ngram) for ngram in ngrams(text, n=n_gram)]
    freqs = {
        k: v / len(text)
        for k, v in sorted(Counter(text).items(), key=lambda item: item[1], reverse=True)
        if v > 0
    }
    return freqs

