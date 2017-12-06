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
    match = re.match("^[A-ZÄÖÜa-zäöü0-9_]*$", tested_string)
    return match is not None

def _col_log_likelihood(count_a, count_b, count_ab, N):
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
    effie_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(raw)
    
    # reallocate stopwords filtering to LATER stage, after bigram collection; because NEW, invalid bigrams occur
    
    # Filter tokenized words list so that only those words with symbols from [0-9] and [A-Za-z] are kept
    # (and no "words" like [',', '.', '»', '›', '«', '.«', ',«', '.‹«', "'«", '?«', '!«', '«,', '!', '?', ';', ':', "'", '(', ')', '),', '...', '...!', '...,', '...?', '...«', '..«', '.«‹', '.‹'])
    
    effie_tokenized_tmp = []
    for l in range (0, len(effie_tokenized)):
        if (only_letters(effie_tokenized[l]) and only_letters(effie_tokenized[l])):
            effie_tokenized_tmp.append(effie_tokenized[l])
            
    # Load pre-defined stopwords list
    
    f = open('stopwords.txt')
    stopwords = f.read()
    stopwords_tokenized = nltk.tokenize.WordPunctTokenizer().tokenize(stopwords)
    
    stopchars_tokenized = stopwords_tokenized
    
    # Create a linebreak-delimited string from these tokenized stopwords again, which is easier to search and compare with the novel bigrams later
    
    stopchars = ''.join(stopchars_tokenized)
    
    # Filter tokenized words by stopwords
    effie_tokenized_tmp_tmp = []
    for o in range (0, len(effie_tokenized_tmp)):
        if (effie_tokenized_tmp[o].lower() not in stopchars.lower()):
            effie_tokenized_tmp_tmp.append(effie_tokenized_tmp[o])
    effie_tokenized_tmp = effie_tokenized_tmp_tmp
            
    # print ("effie_tokenized_tmp[0:1000]:",effie_tokenized_tmp[0:1000])
                 
    # We will take later word counts from the effie_tokenized_tmp list, because it only contains "real words" 
    # However, we will create the effie_bigrams list from effie_tokenized because this way,
    # we do consider "weird symbols" as delimiters that separate words that have no bigram relationship
    
    fdist2 = FreqDist(effie_tokenized_tmp)
    # print ("Descriptive counts from FreqDist fdist2: ", fdist2)
    
    fdist1 = FreqDist(effie_tokenized)
    # print ("Descriptive counts from FreqDist fdist1: ", fdist1)
    
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
    
    # print ("stopwords_tokenized[0:20]", stopwords_tokenized[0:20])
    
    # Create delimiters-stopword list, such as ',','.', '»', '!', '?', ';', '(', ')' (comma to be discussed)
    
    # TODO: check result bigramlist_occurence_descending (by creating string file) and adjust delimiter list + '-',...
    # TODO: change for-loop where counts are calculated so that .N() is taken from the filtered bigrams list
    # TODO: delete unneccessary print statements
    # TODO: achieve that this .py-file may be called together with a text file in a "python *.py *.txt" manner from a command line
    
    # We could start filtering for "weird" symbols here already, but it's better to do that later by regular expressions
    
    # delimiters_tokenized = [',', '.', '»', '›', '«', '.«', ',«', '.‹«', "'«", '?«', '!«', '«,', '!', '?', ';', ':', "'", '(', ')', '),', '...', '...!', '...,', '...?', '...«', '..«', '.«‹', '.‹']
    # stopchars_tokenized = stopwords_tokenized + delimiters_tokenized
        
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
        N = fdist2.N()
        #N = fdist_bigrams.N()
        
        #print ("current_bigram, count_a, count_b, count_ab, N:", current_bigram, count_a, count_b, count_ab, N)
        
        current_bigram_LogL = _col_log_likelihood(count_a, count_b, count_ab, N)
        bigram_listoflists[i][1] = current_bigram_LogL
        
        #bigram_listoflists.append([bigrams_sortby_first[i],0,0,0,0])
        #bigram_listoflists[i][1] = (fdist_bigrams.freq())*fdist_bigrams.N()
    
    # print ("bigram_listoflists: ", bigram_listoflists)
    
    # Sort created list of bigrams with counts by number of bigram occurence, descending
    
    bigramlist_occurence_descending = sorted(bigram_listoflists, key=itemgetter(1,0), reverse=True)
    bigramlist_occurence_descending.insert(0, ['(bigram)' , '2*log*lambda' , 'c1' , 'c2' , 'c12'])
    
    #wanted_bigrams = [('hohen','cremmen'),('ge','frau'),('vetter','briest'),('fra','briest'),('sei','dank'),('gleich','danach'),('weites','feld'),('gott','sei'),('doktor','hannemann'),('selben','augenblicke'),('sidonie','grasenabb'),('mutter','tochter'),('calatrava','ritter'),('junge','frau'),('pastor','lindequist'),('tante','therese'),('jungen','frau'),('sagte','effi'),('baron','innstetten'),('links','rechts')]
    
    #wanted_bigrams = ''.join(wanted_bigrams)
    
    #for m in range (0, 2000):
    #    if(m<20):
    #        print(bigramlist_occurence_descending[m])
    #    if(   bigramlist_occurence_descending[m][0] in wanted_bigrams   ):
    #        print(bigramlist_occurence_descending[m])
    
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
