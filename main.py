# -*- coding: utf-8 -*-

import nltk
from nltk.probability import FreqDist
from nltk.util import bigrams

print ()

f = open('effie.txt')
raw = f.read()
effie_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(raw)

# Exclude stopwords
# TODO: case-insensitive filtering
# reallocate stopwords filtering to LATER stage, after bigram collection; because NEW, invalid bigrams occur
f = open('stopwords.txt')
stopwords = f.read()
stopwords_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(raw)

tokenized_filtered = [word for word in stopwords_tokenized if word not in stopwords]

fdist1 = FreqDist(tokenized_filtered)
print ("Descriptive counts from Fredist: ", fdist1)
print ("Most common words in Effie Briest: \n", fdist1.most_common(50))


effie_bigrams = list(bigrams(tokenized_filtered))
fdist_bigrams = nltk.FreqDist(effie_bigrams)
print("most frequent Bigrams in Effie Briest: \n", fdist_bigrams.most_common(50))
