# --inp_path data/flickr8k/Images/667626_18933d713e.jpg --collection_saved data/collection_saved.pkl
# --caption_collection_txt data/flickr8k/small_captions_collections.txt --output_file_path output/output1.txt


import collections
from heapq import heapify, heappop, heappush
from typing import List

import numpy as np

from config import max_retrieved_documents
from utils import tokenize_and_store, splitter


class Collection:
    def __init__(self, CF, DF, vocabulary, documents, doc_number, L, trans_probs):
        self.CF = CF
        self.DF = DF
        self.vocabulary = vocabulary
        self.documents = documents
        self.doc_number = doc_number
        self.length = L
        self.trans_probs = trans_probs
        self.index = {}

    def generate_index(self):
        for document in self.documents:
            for word in document.TF:
                if word in self.index:
                    self.index[word].append(document)
                else:
                    self.index[word] = [document]


class Document:
    TF = {}  # Term Frequency
    TFIDF = {}
    norm = 0
    doc_len = 0
    text = ''

    def __init__(self, TF, text):
        self.TF = TF
        self.text = text
        for freq in TF.values():
            self.doc_len += freq

    def generate_vsm_params(self, DF, N):
        for word in self.TF:
            self.TFIDF[word] = (1 + np.log2(self.TF[word])) / (np.log2(1 + N / DF[word]))
            self.norm += self.TFIDF[word] ** 2
        self.norm = self.norm ** 0.5


def make_collection(captions_collection_path: str, trans_probs) -> Collection:
    documents = []  # doc_number -> Document Object
    CF = {}  # Collection Frequency. How many times a term appears in the entire collection
    DF = {}  # Document Frequency. How many documents contains a term
    doc_number = 0
    with open(captions_collection_path, encoding="utf8", errors='ignore') as f:
        for line in f:
            # Each line is a new document
            dictionary = collections.defaultdict(int)
            tokenize_and_store(line, dictionary, frequency=True)
            documents.append(Document(dictionary, line.rstrip('\n')))
            # print(document)
            for w in dictionary:
                CF[w] = CF.get(w, 0) + dictionary[w]
                DF[w] = DF.get(w, 0) + 1
            doc_number += 1

    # Finding vocabulary which maps word to an index in [0, total_unique_words)
    vocabulary = {}
    index = 0
    for word in CF:
        vocabulary[word] = index
        index += 1

    L = 0
    for document in documents:
        document.generate_vsm_params(DF, doc_number)
        L += document.doc_len

    return Collection(CF, DF, vocabulary, documents, doc_number, L, trans_probs)


def bm2_score(query: str, document: Document, collection: Collection, k: float = 2, b: float = 0.75) -> int:
    N = collection.doc_number

    def IDF(term: str):
        DFqi = collection.DF.get(term, 0)
        return np.log(1 + ((N - DFqi + 0.5) / (DFqi + 0.5)))

    L = 0
    for d in collection.documents:
        L += d.doc_len
    L /= N

    score = 0
    for query_term in splitter(query):
        tf = document.TF.get(query_term, 0)
        score += IDF(query_term) * (tf * (k + 1)) / (tf + k * (1 - b + b * document.doc_len / L))
    return score


def vsm_score(query: str, document: Document, collection: Collection) -> float:
    N = collection.doc_number

    # print(collection.length)
    def tfidf_score(tf_, term):
        dfi = collection.DF.get(term, 1)
        return (1 + np.log2(tf_)) / np.log2(1 + N / dfi)

    qf = {}
    for query_term in splitter(query):
        if query_term not in qf:
            qf[query_term] = 0
        qf[query_term] += 1

    score = 0
    qnorm = 0

    for query_term in qf:
        qf[query_term] = tfidf_score(qf[query_term], query_term)
        qnorm += qf[query_term] ** 2
    qnorm = qnorm ** 0.5
    for query_term in qf:
        tf = document.TFIDF.get(query_term, 0)
        score += tf * qf[query_term]
    return score / qnorm / document.norm


def lm_score(query: str, document: Document, collection: Collection, raw=False) -> float:
    N = collection.doc_number

    def dirichlet_score(term, mu=1):
        cfi = collection.CF.get(term, 0)
        tfi = document.TF.get(term, 0)

        return np.log((tfi + mu * cfi / collection.length) / (document.doc_len + mu))

    score = 0
    for query_term in splitter(query, raw):
        score += dirichlet_score(query_term)
    return score


def lmt_score(query: str, document: Document, collection: Collection, raw=False) -> float:
    N = collection.doc_number
    tprobs = collection.trans_probs

    def dirichlet_score(term, mu=1):
        cfi = collection.CF.get(term, 0)
        tfi = document.TF.get(term, 0)
        return (tfi + mu * cfi / collection.length) / (document.doc_len + mu)

    score = 0
    for query_term in splitter(query, raw):
        if query_term in tprobs:
            temp = 0
            for tqterm in tprobs[query_term]:
                temp += dirichlet_score(tqterm) * tprobs[query_term][tqterm]
            score += np.log(temp)
        # else:
        #     score += dirichlet_score(query_term)
    return score


def similarity(words_tokens_1: List[str], document: Document):
    common_words = 0
    for word in words_tokens_1:
        if word in document.TF:
            common_words += 1
    return common_words / (len(words_tokens_1) + document.doc_len)


def get_top_docs(query, collection: Collection, scoring_function, raw=False):
    priority_queue = []  # Keep top 200 documents
    heapify(priority_queue)
    word_token_1 = splitter(query, raw)
    for document in collection.documents:
        if similarity(word_token_1, document) > 0.01:
            heappush(priority_queue, (scoring_function(query, document, collection), document.text))
            if len(priority_queue) > max_retrieved_documents:
                heappop(priority_queue)
    # priority_queue has top 200 elements sorted in reverse order
    top_200 = []
    while len(priority_queue) > 0:
        top_200.append(heappop(priority_queue))
    top_200.reverse()
    return top_200
