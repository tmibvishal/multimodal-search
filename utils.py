import collections
import json
import re
from functools import lru_cache
from typing import Dict, List
import pandas as pd
import numpy as np
import os
import math

from config import delimiters, debug, stemmer, stop_words




@lru_cache(maxsize=10000000)
def stem(string: str):
    s = string.encode("ascii", "ignore").decode()
    return stemmer.stem(s)


def splitter(text):
    word_tokens = re.split(delimiters, text)
    words_taken = []
    for word in word_tokens:
        word = word.lower()
        if word not in stop_words:
            word = stem(word)
            words_taken.append(word)
    return words_taken


def tokenize_and_store(texts: str or List[str], dictionary: Dict[str, int], frequency=True):
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        word_tokens = splitter(text)
        for w in word_tokens:
            if frequency:
                dictionary[w] += 1
            else:
                dictionary[w] = 1
    if '' in dictionary:
        dictionary.pop('')


def tokenize_fill_TF(texts: str or List[str], TF: np.ndarray, vocabulary: Dict[str, int]):
    if debug:
        assert TF.size == len(vocabulary)
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        word_tokens = splitter(text)
        for w in word_tokens:
            if w in vocabulary:
                TF[vocabulary[w]] += 1
            else:
                if debug:
                    assert w == ''


