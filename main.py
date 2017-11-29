# -*- coding: utf-8 -*-

import nltk
from nltk.probability import FreqDist
from nltk.util import bigrams

german = u"Veränderungen über einen Walzer"

f = open('effie.txt')
raw = f.read()
effie_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(raw)

#print ()
#print (len(effie_tokenized))
#print (effie_tokenized[0:500])

fdist1 = FreqDist(effie_tokenized)
print (fdist1)
print (fdist1.most_common(10))
print ("total number of samples:", fdist1.N())

effie_bigrams = list(bigrams(effie_tokenized))
print("first 10 effie bigrams:", effie_bigrams[0:10])

fdist_bigrams = nltk.FreqDist(effie_bigrams)
print("first 10 frequency determinations of effie bigrams:", fdist_bigrams.most_common(10))
