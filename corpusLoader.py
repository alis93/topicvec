# -*- coding=GBK -*-

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters
from nltk.tokenize import sent_tokenize
import HTMLParser
import os
import pdb
from utils import *

def extractSentenceWords(doc, remove_url=True, remove_punc="utf-8", min_length=1):
    if remove_punc:
        # ensure doc_u is in unicode
        if not isinstance(doc, unicode):
            encoding = remove_punc
            doc_u = doc.decode(encoding)
        else:
            doc_u = doc
        # remove unicode punctuation marks, keep ascii punctuation marks
        doc_u = doc_u.translate(unicode_punc_tbl)
        if not isinstance(doc, unicode):
            doc = doc_u.encode(encoding)
        else:
            doc = doc_u

    if remove_url:
        re_url = r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        doc = re.sub( re_url, "", doc )

    sentences= sent_tokenize(doc)
    wc = 0
    wordsInSentences = []

    for sentence in sentences:
        if sentence == "":
            continue

        if not re.search( "[A-Za-z0-9]", sentence ):
            continue

        # split at spaces
        words = sentence.split(' ')


        words = filter( lambda w: w, words )

        if len(words) >= min_length:
            wordsInSentences.append(words)
            wc += len(words)

    return wordsInSentences, wc

def load_docs(filepath):
    data = []
    # with open('dataset/yelp_reviews.txt') as dataset:
    with open(filepath,'r') as dataset:

        lines = dataset.readlines()
        for l in lines:
            data.append(l.decode('utf-8'))

    setDocNum = len(data)
    orig_docs_words = []
    orig_docs_name =[]


    for idx,doc in enumerate(data):
        text = doc
        wordsInSentences, wc = extractSentenceWords(text)
        orig_docs_words.append(wordsInSentences)
        orig_docs_name.append("id-"+str(idx))

    return setDocNum, orig_docs_words, orig_docs_name
