# -*- coding: utf-8 -*-

import nltk
from nltk.probability import FreqDist
from nltk.util import bigrams
from operator import itemgetter
#import json

# TODO: delete unnecessary print statements

print ()

f = open('effie.txt')
raw = f.read()
effie_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(raw)

# reallocate stopwords filtering to LATER stage, after bigram collection; because NEW, invalid bigrams occur

fdist1 = FreqDist(effie_tokenized)
print ("Descriptive counts from FreqDist: ", fdist1)

effie_bigrams = list(bigrams(effie_tokenized))

# Print several nltk outputs for testing

#print ("effie_bigrams[0:20]", effie_bigrams[0:20])
#print ("type(effie_bigrams[3][0])", type(effie_bigrams[3][0]))
#print ("effie_bigrams[3][0]", effie_bigrams[3][0])
#print ("effie_bigrams[3][0].lower()", effie_bigrams[3][0].lower())
#print("First Bigrams (with count) in Effie Briest: \n", fdist_bigrams)
#print ("fdist_bigrams: ", fdist_bigrams)
#print ("type(fdist_bigrams): ", type(fdist_bigrams))
#print ("(fdist_bigrams.freq((',', 'daß')))*fdist_bigrams.N(): ", (fdist_bigrams.freq((',', 'daß')))*fdist_bigrams.N())
#print ("(fdist_bigrams._cumulative_frequencies((',', 'daß'))): ", (fdist_bigrams._cumulative_frequencies((',', 'daß'))))
#print ("fdist_bigrams.hapaxes()[0:5]: ", fdist_bigrams.hapaxes()[0:5])
#print ("len(fdist_bigrams)", len(fdist_bigrams))

# Prepare filtering of bigrams list by stopwords
# We filter after creating the bigrams. It would not make sense to filter directly after tokenization,
# because we don't want to let words that were no bigrams in the original text to move together;
# even in the sentence "wollen, daß" we don't want "wollen" and "daß" to become a bigram -
# commas separate clauses, and these usually represent self-contained units with their own meaning and message.
# In sentiment analysis, word n-grams would need to be analyzed with respect to inter-clausal relationships,
# but this is not part of our agenda here - aggregating words from separate clauses would distort our results now.

# Load pre-defined stopwords list

f = open('stopwords.txt')
stopwords = f.read()
stopwords_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(stopwords)

# print ("stopwords_tokenized[0:20]", stopwords_tokenized[0:20])

# Create delimiters-stopword list, such as ',','.', '»', '!', '?', ';', '(', ')' (comma to be discussed)

# TODO: check result bigramlist_occurence_descending (by creating string file) and adjust delimiter list + '-',...

delimiters_tokenized = [',', '.', '»', '«', '.«', ',«', "'«", '?«', '!«', '«,', '!', '?', ';', ':', "'", '(', ')', '),', '...', '...!', '...,', '...?', '...«', '..«', '.«‹', '.‹']
stopchars_tokenized = stopwords_tokenized + delimiters_tokenized

# Create a linebreak-delimited string from these tokenized stopwords again, which is easier to search and compare with the novel bigrams later

stopchars = ''.join(stopchars_tokenized)

# Add delimiters to stopwords list

# Filter bigrams list by stopwords

# Exclude stopwords
# rewrite for accessing both bigram components: tokenized_filtered = [word for word in stopwords_tokenized if word.lower() not in stopwords.lower()]

# Determine number of occurences of every bigram

effie_bigrams_tmp = []
for h in range (0, len(effie_bigrams)):
    if (effie_bigrams[h][0].lower() not in stopchars.lower()) and (effie_bigrams[h][1].lower() not in stopchars.lower()):
        effie_bigrams_tmp.append(effie_bigrams[h])

# The bigrams are cleaned of stopwords and stop-delimiters now

effie_bigrams = effie_bigrams_tmp

fdist_bigrams = nltk.FreqDist(effie_bigrams)
#print("Most frequent Bigrams in Effie Briest: \n", fdist_bigrams.most_common(4))

# create bigram list (with counts) ordered by first bigram component

bigrams_sortby_first = sorted(fdist_bigrams, key=itemgetter(0,1))
#print("First Bigrams sorted by first component: \n", bigrams_sortby_first[450:475])

# create bigram list of lists:
# | bigram AB | Nr. of AB | Nr. of A~B | Nr. of ~AB | Nr. of ~A~B |
bigram_listoflists = []

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
print ("bigramlist_occurence_descending[0:10] ", bigramlist_occurence_descending[0:10])

# def calcL(k,n,p)
# p^k * (1-p)^(n-k)

# def calcLogLambda(N, c1, c2, c12)

"""

# Create a text file with all bigrams and counts

with open('bigrams.txt', 'w') as f:
    for k in range (0, len(bigramlist_occurence_descending)):
        rowstring = '\n'
        rowstring += bigramlist_occurence_descending[k][0][0]
        rowstring += ", "
        rowstring += bigramlist_occurence_descending[k][0][1]
        rowstring += "\t"
        for j in range (1, len(bigramlist_occurence_descending[k])):
            rowstring += str(bigramlist_occurence_descending[k][j])
            rowstring += "\t"
        rowstring += "\n"
        f.write(rowstring + "\n")
        f.write('\n')
"""