# -*- coding: utf-8 -*-

import nltk
from nltk.probability import FreqDist
from nltk.util import bigrams
from operator import itemgetter

print ()

f = open('effie.txt')
raw = f.read()
effie_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(raw)

# reallocate stopwords filtering to LATER stage, after bigram collection; because NEW, invalid bigrams occur

# Exclude stopwords
# tokenized_filtered = [word for word in stopwords_tokenized if word.lower() not in stopwords.lower()]

fdist1 = FreqDist(effie_tokenized)
print ("Descriptive counts from FreqDist: ", fdist1)
print ("Most common words in Effie Briest: \n", fdist1.most_common(4))

effie_bigrams = list(bigrams(effie_tokenized))

# Determine number of occurences of every bigram

fdist_bigrams = nltk.FreqDist(effie_bigrams)
print("Most frequent Bigrams in Effie Briest: \n", fdist_bigrams.most_common(4))

# Print several nltk outputs for testing

#print("First Bigrams (with count) in Effie Briest: \n", fdist_bigrams)
#print ("fdist_bigrams: ", fdist_bigrams)
#print ("type(fdist_bigrams): ", type(fdist_bigrams))
#print ("(fdist_bigrams.freq((',', 'daß')))*fdist_bigrams.N(): ", (fdist_bigrams.freq((',', 'daß')))*fdist_bigrams.N())
#print ("(fdist_bigrams._cumulative_frequencies((',', 'daß'))): ", (fdist_bigrams._cumulative_frequencies((',', 'daß'))))
#print ("fdist_bigrams.hapaxes()[0:5]: ", fdist_bigrams.hapaxes()[0:5])
#print ("len(fdist_bigrams)", len(fdist_bigrams))

# Prepare filtering of bigrams list by stopwords

# Load pre-defined stopwords list

# f = open('stopwords.txt')
# stopwords = f.read()
# stopwords_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(raw)

# Filter bigrams list by stopwords

# create bigram list (with counts) ordered by first bigram component

bigrams_sortby_first = sorted(fdist_bigrams, key=itemgetter(0,1))
#print("First Bigrams sorted by first component: \n", bigrams_sortby_first[450:475])

# create bigram list of lists:
# | bigram AB | Nr. of AB | Nr. of A~B | Nr. of ~AB | Nr. of ~A~B |
bigram_listoflists = []

print (type(bigrams_sortby_first))

for i in range (0, len(bigrams_sortby_first)):
    current_bigram = bigrams_sortby_first[i]
    bigram_listoflists.append([current_bigram,0,0,0,0])
    
    # Nr. of AB is calculated:
    bigram_listoflists[i][1] = round(fdist_bigrams.freq(current_bigram)*fdist_bigrams.N(), 0)
    
    # Nr. of A~B is calculated: number_of_A_total - number of bigrams AB    
    number_of_A_total = round(fdist1.freq(current_bigram[0])*fdist1.N(), 0)
    bigram_listoflists[i][2] = number_of_A_total - bigram_listoflists[i][1]
    
    # Nr. of ~AB is calculated: number_of_B_total - number of bigrams AB
    number_of_B_total = round(fdist1.freq(current_bigram[1])*fdist1.N(), 0)
    bigram_listoflists[i][3] = number_of_B_total - bigram_listoflists[i][1]
    
    # Nr. of ~A~B is calculated: number_of_different_bigrams - (Nr. of A~B + Nr. of ~AB)
    number_of_different_bigrams = len(fdist_bigrams) - (bigram_listoflists[i][1] + bigram_listoflists[i][2] + bigram_listoflists[i][3])
    bigram_listoflists[i][4] = number_of_different_bigrams
    
    #bigram_listoflists.append([bigrams_sortby_first[i],0,0,0,0])
    #bigram_listoflists[i][1] = (fdist_bigrams.freq())*fdist_bigrams.N()

# print ("bigram_listoflists: ", bigram_listoflists)

# Sort created list of bigrams with counts by number of bigram occurence, descending

bigramlist_occurence_descending = sorted(bigram_listoflists, key=itemgetter(1,0), reverse=True)
print ("bigramlist_occurence_descending[0:100] ", bigramlist_occurence_descending[0:100])

# Create delimiters-stopword list, such as '.', '»', '!', '?', ',', ';', '(', ')' (comma to be discussed)

# Filter bigrams by delimiters

#fdist_bigrams = nltk.FreqDist(effie_bigrams)
#print("Most frequent Bigrams in Effie Briest: \n", fdist_bigrams.most_common(4))
