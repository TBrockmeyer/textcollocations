# -*- coding: utf-8 -*-

"""

This example finds bigrams that occur statistically significantly often in a given text.
Statistical analysis follows the work of Dunning in "Accurate Methods for the Statistics of Surprise and Coincidence".

Run the script on a text file that you would like to know the most significant bigrams for.
The text should be encoded in UTF-8:
    $ python main.py <path-to-text-file>
Try this on a sample text in the resources directory, the German novel "Effie Briest" by Theodor Fontane:
    $ python main.py effie.txt
    
"""

import nltk
from nltk.probability import FreqDist
from nltk.util import bigrams
from operator import itemgetter
import math
import re
import argparse
import sys
#import json

def only_letters(tested_string):
    match = re.match("^[A-Za-z0-9_]*$", tested_string)
    return match is not None

def calculate_log_likelihood(count_a, count_b, count_ab, N):
    p = count_b / N
    try:
        p1 = count_ab / count_a
    except ZeroDivisionError as e:
        p1 = 1
    try:
        p2 = (count_b - count_ab) / (N - count_a)
    except ZeroDivisionError as e:
        p2 = 1
    
    L1 = math.pow(p,count_ab)*math.pow((1.0-p),(count_a - count_ab))
    L2 = math.pow(p,count_b - count_ab)*math.pow((1.0-p),(N - count_a - count_b + count_ab))
    L3 = math.pow(p1,count_ab)*math.pow((1.0-p1),(count_a - count_ab))
    L4 = math.pow(p2,count_b - count_ab)*math.pow((1.0-p2),(N - count_a - count_b + count_ab))
    
    if(L1!=0):
        logL1 = math.log(L1,math.exp(1))
    else:
        logL1 = 0
    if(L2!=0):
        logL2 = math.log(L2,math.exp(1))
    else:
        logL2 = 0
    if(L3!=0):
        logL3 = math.log(L3,math.exp(1))
    else:
        logL3 = 0
    if(L4!=0):
        logL4 = math.log(L4,math.exp(1))
    else:
        logL4 = 0
        
    likelihood = logL1 + logL2 - logL3 - logL4
    
    # return (-2.0 * likelihood)
    return (-2.0 * likelihood)

def main(text_file):
    
    f = open(text_file)
    raw = f.read()
    # Make sure that "-"" in Hohen-Cremmen and similar combinations is not taken as separate "word", but a delimiter
    raw = re.sub(r'\b-\b', ' ', raw)
    raw = raw.lower()
    giventext_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(raw)
    
    # reallocate stopwords filtering to LATER stage, after bigram collection; because NEW, invalid bigrams occur
    
    # Filter tokenized words list so that only those words with symbols from [0-9] and [A-Za-z] are kept
    # (and no "words" like [',', '.', '»', '›', '«', '.«', ',«', '.‹«', "'«", '?«', '!«', '«,', '!', '?', ';', ':', "'", '(', ')', '),', '...', '...!', '...,', '...?', '...«', '..«', '.«‹', '.‹'])
    
    giventext_tokenized_tmp = []
    for l in range (0, len(giventext_tokenized)):
        if (only_letters(giventext_tokenized[l]) and only_letters(giventext_tokenized[l])):
            giventext_tokenized_tmp.append(giventext_tokenized[l])
            
    # Load pre-defined stopwords list
    
    f = open('stopwords.txt')
    stopwords = f.read()
    stopwords_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(stopwords)
    
    stopchars_tokenized = stopwords_tokenized
    stopchars = stopwords_tokenized
    
    # Filter tokenized words by stopwords
    giventext_tokenized_tmp_tmp = []
    for o in range (0, len(giventext_tokenized_tmp)):
        if (giventext_tokenized_tmp[o].lower() not in stopchars):
            giventext_tokenized_tmp_tmp.append(giventext_tokenized_tmp[o])
    giventext_tokenized_tmp = giventext_tokenized_tmp_tmp
    
    # The bigrams are cleaned of stopwords and "weird symbols" now
                 
    # We take later word counts and create the giventext_bigrams from the giventext_tokenized_tmp list,
    # because it only contains "real words" 
    # However, we could create the giventext_bigrams list from giventext_tokenized
    # AFTER creation of bigrams because this way,
    # we would respect that "weird symbols" often serve as delimiters
    # that separate words that have no reasonable "bigram relationship"
    
    fdist2 = FreqDist(giventext_tokenized_tmp)
    # print ("Descriptive counts from FreqDist fdist2: ", fdist2)
    
    # We create bigrams from the already filtered list of tokenized words
    
    giventext_bigrams = list(bigrams(giventext_tokenized_tmp))
    
    # Determine number of occurences of every bigram
    
    fdist_bigrams = nltk.FreqDist(giventext_bigrams)
    
    # create bigram list (with counts) ordered by first bigram component
    
    bigrams_sortby_first = sorted(fdist_bigrams, key=itemgetter(0,1))
    
    # create bigram list of lists:
    # | bigram AB | Nr. of AB | Nr. of A~B | Nr. of ~AB | Nr. of ~A~B |
    bigram_listoflists = []
    
    for i in range (0, len(bigrams_sortby_first)):
        current_bigram = bigrams_sortby_first[i]
        bigram_listoflists.append([current_bigram,0,0,0,0])
        
        # Nr. of AB is calculated:
        number_of_AB = round(fdist_bigrams.freq(current_bigram)*fdist_bigrams.N(), 0)
        bigram_listoflists[i][4] = number_of_AB
        
        # Nr. of A is calculated: number_of_A_total   
        number_of_A_total = round(fdist2.freq(current_bigram[0])*fdist2.N(), 0)
        # bigram_listoflists[i][2] = number_of_A_total - bigram_listoflists[i][1]
        bigram_listoflists[i][2] = number_of_A_total
        
        # Nr. of B is calculated: number_of_B_total
        number_of_B_total = round(fdist2.freq(current_bigram[1])*fdist2.N(), 0)
        # bigram_listoflists[i][3] = number_of_B_total - bigram_listoflists[i][1]
        bigram_listoflists[i][3] = number_of_B_total
        
        count_ab = number_of_AB
        count_a = number_of_A_total
        count_b = number_of_B_total
        N = len(giventext_tokenized_tmp)
        # N can also be obtained from fdist2:
        #N = fdist2.N()
            
        current_bigram_LogL = calculate_log_likelihood(count_a, count_b, count_ab, N)
        bigram_listoflists[i][1] = current_bigram_LogL
    
    # Sort created list of bigrams with counts by number of bigram occurence, descending
    
    bigramlist_occurence_descending = sorted(bigram_listoflists, key=itemgetter(1,0), reverse=True)
    bigramlist_occurence_descending.insert(0, ['(bigram)' , '2*log*lambda' , 'c1' , 'c2' , 'c12'])
    
    for m in range (0, 21):
        print(bigramlist_occurence_descending[m])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'text_file',
        help='A file containing the document to process.  '
        'Should be encoded in UTF-8')
    args = parser.parse_args()
    main(args.text_file)
