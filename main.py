# -*- coding: utf-8 -*-

import nltk
from nltk.probability import FreqDist
from nltk.util import bigrams
from operator import itemgetter
import math
import re
#import json

def only_letters(tested_string):
    match = re.match("^[A-ZÄÖÜa-zäöü0-9_]*$", tested_string)
    return match is not None

def _col_log_likelihood(count_a, count_b, count_ab, N):
    """
    A function that will just compute log-likelihood estimate, in
    the original paper it's described in algorithm 6 and 7.

    This *should* be the original Dunning log-likelihood values,
    unlike the previous log_l function where it used modified
    Dunning log-likelihood values
    """
    p = count_b / N
    try:
        p1 = count_ab / count_a
    except ZeroDivisionError as e:
        p1 = 1

    try:
        p2 = (count_b - count_ab) / (N - count_a)
    except ZeroDivisionError as e:
        p2 = 1

    try:
        summand1 = (count_ab * math.log(p) +
                    (count_a - count_ab) * math.log(1.0 - p))
    except ValueError as e:
        summand1 = 0

    try:
        summand2 = ((count_b - count_ab) * math.log(p) +
                    (N - count_a - count_b + count_ab) * math.log(1.0 - p))
    except ValueError as e:
        summand2 = 0

    if count_a == count_ab or p1 <= 0 or p1 >= 1:
        summand3 = 0
    else:
        summand3 = (count_ab * math.log(p1) +
                    (count_a - count_ab) * math.log(1.0 - p1))

    if count_b == count_ab or p2 <= 0 or p2 >= 1:
        summand4 = 0
    else:
        summand4 = ((count_b - count_ab) * math.log(p2) +
                    (N - count_a - count_b + count_ab) * math.log(1.0 - p2))

    likelihood = summand1 + summand2 - summand3 - summand4

    return (-2.0 * likelihood)

# TODO: delete unnecessary print statements

print ()

f = open('effie.txt')
raw = f.read()
# Make sure that "-"" in Hohen-Cremmen and similar combinations is not taken as separate "word", but a delimiter
raw = re.sub(r'\b-\b', ' ', raw)
raw = raw.lower()
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
#print ("First Bigrams (with count) in Effie Briest: \n", fdist_bigrams)
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
# TODO: 1) change delimiter-rule to a different rule: only allow bigrams with chars from [A-ZÄÖÜa-zäöü] without ß and é
# TODO: 2) make sure that "-"" in Hohen-Cremmen and similar combinations is not taken as separate "word", but a delimiter
# TODO: change for-loop where counts are calculated so that .N() is taken from the filtered bigrams list
    # to be discussed, because the number of single word occurrences should be the same before and after filtering
    # and it is way more convenient to use the count from the tokenized word list
# TODO: half DONE, revise: calculate c1 c2 c12 p1 p2 p12 etc. and create LogLikelihoods
# TODO: achieve that this .py-file may be called together with a text file in a "python *.py *.txt" manner from a command line

# We could start filtering for "weird" symbols here already, but it's better to do that later by regular expressions

# delimiters_tokenized = [',', '.', '»', '›', '«', '.«', ',«', '.‹«', "'«", '?«', '!«', '«,', '!', '?', ';', ':', "'", '(', ')', '),', '...', '...!', '...,', '...?', '...«', '..«', '.«‹', '.‹']
# stopchars_tokenized = stopwords_tokenized + delimiters_tokenized
stopchars_tokenized = stopwords_tokenized

# Create a linebreak-delimited string from these tokenized stopwords again, which is easier to search and compare with the novel bigrams later

stopchars = ''.join(stopchars_tokenized)

# Filter bigrams list by stopwords

effie_bigrams_tmp = []
for h in range (0, len(effie_bigrams)):
    if (effie_bigrams[h][0].lower() not in stopchars.lower()) and (effie_bigrams[h][1].lower() not in stopchars.lower()):
        effie_bigrams_tmp.append(effie_bigrams[h])
        
effie_bigrams = effie_bigrams_tmp
        
# Filter bigrams list so that only those bigrams with symbols from [0-9] and [A-Za-z] are kept

effie_bigrams_tmp = []
for h in range (0, len(effie_bigrams)):
    if (only_letters(effie_bigrams[h][0]) and only_letters(effie_bigrams[h][1])):
        effie_bigrams_tmp.append(effie_bigrams[h])
        
effie_bigrams = effie_bigrams_tmp

# The bigrams are cleaned of stopwords and "weird symbols" now

# Determine number of occurences of every bigram

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
    bigram_listoflists.append([current_bigram,0,0,0,0,0])
    
    # Nr. of AB is calculated:
    bigram_listoflists[i][1] = round(fdist_bigrams.freq(current_bigram)*fdist_bigrams.N(), 0)
    
    # Nr. of A is calculated: number_of_A_total   
    number_of_A_total = round(fdist1.freq(current_bigram[0])*fdist1.N(), 0)
    # bigram_listoflists[i][2] = number_of_A_total - bigram_listoflists[i][1]
    bigram_listoflists[i][2] = number_of_A_total
    
    # Nr. of B is calculated: number_of_B_total
    number_of_B_total = round(fdist1.freq(current_bigram[1])*fdist1.N(), 0)
    # bigram_listoflists[i][3] = number_of_B_total - bigram_listoflists[i][1]
    bigram_listoflists[i][3] = number_of_B_total
    
    # Nr. of ~A~B is calculated: number_of_different_bigrams - (Nr. of A~B + Nr. of ~AB)
    number_of_different_bigrams = len(fdist_bigrams) - (- bigram_listoflists[i][1] + bigram_listoflists[i][2] + bigram_listoflists[i][3])
    bigram_listoflists[i][4] = number_of_different_bigrams
    
    count_ab = bigram_listoflists[i][1]
    count_a = bigram_listoflists[i][2]
    count_b = bigram_listoflists[i][3]
    N = fdist_bigrams.N()
    
    #print ("current_bigram, count_a, count_b, count_ab, N:", current_bigram, count_a, count_b, count_ab, N)
    
    current_bigram_LogL = _col_log_likelihood(count_a, count_b, count_ab, N)
    bigram_listoflists[i][5] = current_bigram_LogL
    
    #bigram_listoflists.append([bigrams_sortby_first[i],0,0,0,0])
    #bigram_listoflists[i][1] = (fdist_bigrams.freq())*fdist_bigrams.N()

# print ("bigram_listoflists: ", bigram_listoflists)

# Sort created list of bigrams with counts by number of bigram occurence, descending

bigramlist_occurence_descending = sorted(bigram_listoflists, key=itemgetter(5,0), reverse=True)
print ("bigramlist_occurence_descending[0:30] ", bigramlist_occurence_descending[0:30])


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
