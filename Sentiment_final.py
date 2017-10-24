# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:40:40 2017

@author: Venkata Ramana
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.metrics import precision, recall
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, '1.csv')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, '0csv')
random.seed(123)
#Forming Unigrams
#creates lists of all positive and negative unigrams  and creating model
#creating list of list of all entries/sentences
pos_uni = []
with open("~//1.csv", 'r') as posSentences:
    for i in posSentences:
        posWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        posWord = ','.join([c for c in posWord if c != "'"])
        posWord = posWord.split(",")
        posWord = list(posWord)
        posWord = [w for w in posWord if not w in stopwords.words('english')]
        pos_uni.append(posWord)  

neg_uni = []
with open("~\\0.csv", 'r') as negSentences:
    for i in negSentences:
        negWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())
        negWord = ','.join([c for c in negWord if c != "'"])
        negWord = negWord.split(",")
        negWord = list(negWord)
        negWord = [w for w in negWord if not w in stopwords.words('english')]
        neg_uni.append(negWord)     

#breaking all entries/sentences into individual string/character         
pos_unis = list(itertools.chain(*pos_uni))
neg_unis = list(itertools.chain(*neg_uni)) 

#build frequency distibution of all words and then frequency distributions of unigrams
# within positive and negative labels
word_fd = FreqDist()
cond_word_fd = ConditionalFreqDist()
for word in pos_unis:
    word_fd[word.lower()] += 1
    cond_word_fd['pos'][word.lower()] += 1
for word in neg_unis:
    word_fd[word.lower()] += 1
    cond_word_fd['neg'][word.lower()] += 1

#Calculating individual and total counts of pos and neg unigrams  
pos_uni_count = cond_word_fd['pos'].N()
neg_uni_count = cond_word_fd['neg'].N()
total_uni_count = pos_uni_count + neg_uni_count    
 
#creating dictionary of unigrams along with scores based on chi-squared test
uni_scores = {}
for word, freq in word_fd.iteritems():
    pos_uni_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_uni_count), total_uni_count)
    neg_uni_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_uni_count), total_uni_count)
    uni_scores[word] = pos_uni_score + neg_uni_score
    
# sub setting positive and negative unigrams with scores into separate dictionaries
pos_uni_dict = dict([(x, uni_scores[x]) for x in (pos_unis)])
neg_uni_dict = dict([(x, uni_scores[x]) for x in (neg_unis)])

# finds the top 1200 best 'number' of pos and neg unigrams based on word scores
# ran the model by iterating best 'number' of pos and neg unigrams from 800-1200
# got best results with best 'number' of pos and neg unigrams = 1200
def find_best_pos_unigrams(pos_uni_dict, number):
    best_pos_vals = sorted(pos_uni_dict.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_pos_unigrams = set([w for w, s in best_pos_vals])
    return best_pos_unigrams
best_pos_unigrams = list(find_best_pos_unigrams(pos_uni_dict, 1200))

def find_best_neg_unigrams(neg_uni_dict, number):
    best_neg_vals = sorted(neg_uni_dict.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_neg_unigrams = set([w for w, s in best_neg_vals])
    return best_neg_unigrams           
best_neg_unigrams = list(find_best_neg_unigrams(neg_uni_dict, 1200))

total_best_unigrams = best_pos_unigrams + best_neg_unigrams

#creates feature selection mechanism that only uses best unigrams
def best_uni_features(words):
    return dict([(word, True) for word in words if word in total_best_unigrams])

#creating list of unigrams and checking whether best unigrams are present in them
best_pos_uni = []
with open("~\\1.csv", 'r') as posSentences:
    for i in posSentences:
        posWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        posWord = ','.join([c for c in posWord if c != "'"])
        posWord = posWord.split(",")
        posWord = [w for w in posWord if not w in stopwords.words('english')]
        posuni = [best_uni_features(posWord), 'pos']
        best_pos_uni.append(posuni)        
best_pos_uni = [x for x in best_pos_uni if x != [{}, 'pos']]

best_neg_uni = []    
with open("~\\0.csv", 'r') as negSentences:
    for i in negSentences:
        negWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        negWord = ','.join([c for c in negWord if c != "'"])
        negWord = negWord.split(",")
        negWord = [w for w in negWord if not w in stopwords.words('english')]       
        neguni = [best_uni_features(negWord), 'neg']
        best_neg_uni.append(neguni)
best_neg_uni = [x for x in best_neg_uni if x != [{}, 'neg']]

#Splitting into training and test
pos_uni_train, pos_uni_test = train_test_split(best_pos_uni, test_size=.25)
neg_uni_train, neg_uni_test = train_test_split(best_neg_uni, test_size=.25)
train_unigrams_Features = pos_uni_train + neg_uni_train
test_unigrams_Features = pos_uni_test + neg_uni_test
                
#training a Naive Bayes Classifier
classifier_uni = NaiveBayesClassifier.train(train_unigrams_Features) 
        
#initiates referenceSets and testSets
referenceSets_uni = collections.defaultdict(set)
testSets_uni = collections.defaultdict(set) 
        
#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(test_unigrams_Features):
    referenceSets_uni[label].add(i)
    predicted_uni = classifier_uni.classify(features)
    testSets_uni[predicted_uni].add(i) 
        
#prints metrics to see how well the feature selection did
print 'train accuracy:', nltk.classify.util.accuracy(classifier_uni, train_unigrams_Features)
print 'test accuracy:', nltk.classify.util.accuracy(classifier_uni, test_unigrams_Features)
print 'pos precision:', precision(referenceSets_uni['pos'], testSets_uni['pos'])
print 'pos recall:', recall(referenceSets_uni['pos'], testSets_uni['pos'])
print 'neg precision:', precision(referenceSets_uni['neg'], testSets_uni['neg'])
print 'neg recall:', recall(referenceSets_uni['neg'], testSets_uni['neg'])

#Forming Bigrams
random.seed(123)
#creates lists of all positive and negative bigrams  and creating model
#creating list of list of all entries/sentences
pos_lines = []
with open("~\\1.csv", 'r') as posSentences:
    for i in posSentences:
        posWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        posWord = ','.join([c for c in posWord if c != "'"])
        posWord = posWord.split(",")
        posWord = list(posWord)
        posWord = [w for w in posWord if not w in stopwords.words('english')]
        pos_lines.append(posWord)

# forming positive bi grams        
pos_bi = []    
for index in range(len(pos_lines)):
    line = pos_lines[index]
    for word in range(len(line)):
        if word < len(line)-1:
           pos_bi.append(''.join([line[word], line[word+1]]))
                            
neg_lines = []  
with open("~\\0.csv", 'r') as negSentences:
    for i in negSentences:
        negWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())
        negWord = ','.join([c for c in negWord if c != "'"])
        negWord = negWord.split(",")
        negWord = list(negWord)
        negWord = [w for w in negWord if not w in stopwords.words('english')]
        neg_lines.append(negWord)     

# forming negative bi grams          
neg_bi = []
for index in range(len(neg_lines)):
    line = neg_lines[index]
    for word in range(len(line)):
        if word < len(line)-1:
           neg_bi.append(''.join([line[word], line[word+1]]))

#build frequency distibution of all words and then frequency distributions of bigrams within positive and negative labels           
bi_fd = FreqDist()
cond_bi_fd = ConditionalFreqDist()
for word in pos_bi:
    bi_fd[word.lower()] += 1
    cond_bi_fd['pos'][word.lower()] += 1
for word in neg_bi:
    bi_fd[word.lower()] += 1
    cond_bi_fd['neg'][word.lower()] += 1

#Calculating individual and total counts of pos and neg bigrams        
pos_bi_count = cond_bi_fd['pos'].N()
neg_bi_count = cond_bi_fd['neg'].N()
total_bi_count = pos_bi_count + neg_bi_count              

#creating dictionary of bigrams along with scores based on chi-squared test           
bi_scores = {}
for word, freq in bi_fd.iteritems():
    pos_bi_score = BigramAssocMeasures.chi_sq(cond_bi_fd['pos'][word], (freq, pos_bi_count), total_bi_count)
    neg_bi_score = BigramAssocMeasures.chi_sq(cond_bi_fd['neg'][word], (freq, neg_bi_count), total_bi_count)
    bi_scores[word] = pos_bi_score + neg_bi_score
    
# sub setting positive and negative bigrams with scores into seperate dictionares
pos_bi_dict = dict([(x, bi_scores[x]) for x in (pos_bi)])
neg_bi_dict = dict([(x, bi_scores[x]) for x in (neg_bi)])

# finds the best 6000, 7000 and 8000 'number' bigrams based on bigrams scores
# ran the model by iterating best 'number' of pos and neg bigrams from 4000-9000
# got best results with best 'number' of pos and neg bigrams 6000, 7000 and 8000 

def find_best_pos_bigrams(pos_bi_dict, number):
    best_pos_vals = sorted(pos_bi_dict.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_pos_bigrams = set([w for w, s in best_pos_vals])
    return best_pos_bigrams
best_pos_bigrams6 = list(find_best_pos_bigrams(pos_bi_dict, 6000))
best_pos_bigrams7 = list(find_best_pos_bigrams(pos_bi_dict, 7000))
best_pos_bigrams8 = list(find_best_pos_bigrams(pos_bi_dict, 8000))

def find_best_neg_bigrams(neg_bi_dict, number):
    best_neg_vals = sorted(neg_bi_dict.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_neg_bigrams = set([w for w, s in best_neg_vals])
    return best_neg_bigrams           
best_neg_bigrams6 = list(find_best_neg_bigrams(neg_bi_dict, 6000))
best_neg_bigrams7 = list(find_best_neg_bigrams(neg_bi_dict, 7000))
best_neg_bigrams8 = list(find_best_neg_bigrams(neg_bi_dict, 8000))

total_best_bigrams6 = best_pos_bigrams6 + best_neg_bigrams6
total_best_bigrams7 = best_pos_bigrams7 + best_neg_bigrams7
total_best_bigrams8 = best_pos_bigrams8 + best_neg_bigrams8

#creates feature selection mechanism that only uses best bigrams    
def best_bigrams_features6(words):
    return dict([(word, True) for word in words if word in total_best_bigrams6])

def best_bigrams_features7(words):
    return dict([(word, True) for word in words if word in total_best_bigrams7])

def best_bigrams_features8(words):
    return dict([(word, True) for word in words if word in total_best_bigrams8])

#creating list of bigrams and checking whether best bigrams are present in top 6000 pos and neg bigrams  
best_pos_bi6 = []       
pos_bi_Features6 = []
with open("~\\1.csv", 'r') as posSentences:
    #file = posSentences
    for i in posSentences:
        posWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())  
        posWord = ','.join([c for c in posWord if c != "'"])
        posWord = posWord.split(",")
        posWord = [w for w in posWord if not w in stopwords.words('english')] 
        pos_bi_Features6 = []
        for word in range(0,len(posWord)):
            if word < len(posWord)-1:
                pos_bi_Features6.append(''.join([posWord[word], posWord[word+1]]))        
        posbi6 = [best_bigrams_features6(pos_bi_Features6), 'pos']
        best_pos_bi6.append(posbi6)
best_pos_bi6 = [x for x in best_pos_bi6 if x != [{}, 'pos']]

best_neg_bi6 = []
neg_bi_Features6 = []        
with open("~\\0.csv", 'r') as negSentences:
    for i in negSentences:
        negWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())  
        negWord = ','.join([c for c in negWord if c != "'"])
        negWord = negWord.split(",")
        negWord = [w for w in negWord if not w in stopwords.words('english')] 
        neg_bi_Features6 = []
        for word in range(0,len(negWord)):
            if word < len(negWord)-1:
                neg_bi_Features6.append(''.join([negWord[word], negWord[word+1]]))       
        negbi6 = [best_bigrams_features6(neg_bi_Features6), 'neg']
        best_neg_bi6.append(negbi6)
best_neg_bi6 = [x for x in best_neg_bi6 if x != [{}, 'neg']]

#Splitting into training and test
pos_bi_train6, pos_bi_test6 = train_test_split(best_pos_bi6, test_size=.25)
neg_bi_train6, neg_bi_test6 = train_test_split(best_neg_bi6, test_size=.25)
train_bi_Features6 = pos_bi_train6 + neg_bi_train6
test_bi_Features6 = pos_bi_test6 + neg_bi_test6
               
#trains a Naive Bayes Classifier
classifier_bi6 = NaiveBayesClassifier.train(train_bi_Features6) 
        
#initiates referenceSets and testSets
referenceSets_bi6 = collections.defaultdict(set)
testSets_bi6 = collections.defaultdict(set) 
        
#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(test_bi_Features6):
    referenceSets_bi6[label].add(i)
    predicted6 = classifier_bi6.classify(features)
    testSets_bi6[predicted6].add(i) 
        
#prints metrics to show how well the feature selection did
print 'train accuracy6:', nltk.classify.util.accuracy(classifier_bi6, train_bi_Features6)
print 'test accuracy6:', nltk.classify.util.accuracy(classifier_bi6, test_bi_Features6)
print 'pos precision6:', precision(referenceSets_bi6['pos'], testSets_bi6['pos'])
print 'pos recall6:', recall(referenceSets_bi6['pos'], testSets_bi6['pos'])
print 'neg precision6:', precision(referenceSets_bi6['neg'], testSets_bi6['neg'])
print 'neg recall6:', recall(referenceSets_bi6['neg'], testSets_bi6['neg'])


#creating list of bigrams and checking whether best bigrams are present in top 7000 pos and neg bigrams  
best_pos_bi7 = []       
pos_bi_Features7 = []
with open("~\\1.csv", 'r') as posSentences:
    #file = posSentences
    for i in posSentences:
        posWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())  
        posWord = ','.join([c for c in posWord if c != "'"])
        posWord = posWord.split(",")
        posWord = [w for w in posWord if not w in stopwords.words('english')] 
        pos_bi_Features7 = []
        for word in range(0,len(posWord)):
            if word < len(posWord)-1:
                pos_bi_Features7.append(''.join([posWord[word], posWord[word+1]]))        
        posbi7 = [best_bigrams_features7(pos_bi_Features7), 'pos']
        best_pos_bi7.append(posbi7)
best_pos_bi7 = [x for x in best_pos_bi7 if x != [{}, 'pos']]

best_neg_bi7 = []
neg_bi_Features7 = []        
with open("~\\0.csv", 'r') as negSentences:
    for i in negSentences:
        negWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())  
        negWord = ','.join([c for c in negWord if c != "'"])
        negWord = negWord.split(",")
        negWord = [w for w in negWord if not w in stopwords.words('english')] 
        neg_bi_Features7 = []
        for word in range(0,len(negWord)):
            if word < len(negWord)-1:
                neg_bi_Features7.append(''.join([negWord[word], negWord[word+1]]))       
        negbi7 = [best_bigrams_features7(neg_bi_Features7), 'neg']
        best_neg_bi7.append(negbi7)
best_neg_bi7 = [x for x in best_neg_bi7 if x != [{}, 'neg']]

#Splitting into training and test
pos_bi_train7, pos_bi_test7 = train_test_split(best_pos_bi7, test_size=.25)
neg_bi_train7, neg_bi_test7 = train_test_split(best_neg_bi7, test_size=.25)
train_bi_Features7 = pos_bi_train7 + neg_bi_train7
test_bi_Features7 = pos_bi_test7 + neg_bi_test7
               
#trains a Naive Bayes Classifier
classifier_bi7 = NaiveBayesClassifier.train(train_bi_Features7) 
        
#initiates referenceSets and testSets
referenceSets_bi7 = collections.defaultdict(set)
testSets_bi7 = collections.defaultdict(set) 
        
#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(test_bi_Features7):
    referenceSets_bi7[label].add(i)
    predicted7 = classifier_bi7.classify(features)
    testSets_bi7[predicted7].add(i) 
        
#prints metrics to show how well the feature selection did
print 'train accuracy7:', nltk.classify.util.accuracy(classifier_bi7, train_bi_Features7)
print 'test accuracy7:', nltk.classify.util.accuracy(classifier_bi7, test_bi_Features7)
print 'pos precision7:', precision(referenceSets_bi7['pos'], testSets_bi7['pos'])
print 'pos recall7:', recall(referenceSets_bi7['pos'], testSets_bi7['pos'])
print 'neg precision7:', precision(referenceSets_bi7['neg'], testSets_bi7['neg'])
print 'neg recall7:', recall(referenceSets_bi7['neg'], testSets_bi7['neg'])

           
#creating list of bigrams and checking whether best bigrams are present in top 8000 pos and neg bigrams  
best_pos_bi8 = []       
pos_bi_Features8 = []
with open("~\\1.csv", 'r') as posSentences:
    #file = posSentences
    for i in posSentences:
        posWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())  
        posWord = ','.join([c for c in posWord if c != "'"])
        posWord = posWord.split(",")
        posWord = [w for w in posWord if not w in stopwords.words('english')] 
        pos_bi_Features8 = []
        for word in range(0,len(posWord)):
            if word < len(posWord)-1:
                pos_bi_Features8.append(''.join([posWord[word], posWord[word+1]]))        
        posbi8 = [best_bigrams_features8(pos_bi_Features8), 'pos']
        best_pos_bi8.append(posbi8)
best_pos_bi8 = [x for x in best_pos_bi8 if x != [{}, 'pos']]

best_neg_bi8 = []
neg_bi_Features8 = []        
with open("~\\0.csv", 'r') as negSentences:
    for i in negSentences:
        negWord = re.findall(r"[a-zA-Z]+|['']", i.rstrip())  
        negWord = ','.join([c for c in negWord if c != "'"])
        negWord = negWord.split(",")
        negWord = [w for w in negWord if not w in stopwords.words('english')] 
        neg_bi_Features8 = []
        for word in range(0,len(negWord)):
            if word < len(negWord)-1:
                neg_bi_Features8.append(''.join([negWord[word], negWord[word+1]]))       
        negbi8 = [best_bigrams_features8(neg_bi_Features8), 'neg']
        best_neg_bi8.append(negbi8)
best_neg_bi8 = [x for x in best_neg_bi8 if x != [{}, 'neg']]

#Splitting into training and test
pos_bi_train8, pos_bi_test8 = train_test_split(best_pos_bi8, test_size=.25)
neg_bi_train8, neg_bi_test8 = train_test_split(best_neg_bi8, test_size=.25)
train_bi_Features8 = pos_bi_train8 + neg_bi_train8
test_bi_Features8 = pos_bi_test8 + neg_bi_test8
               
#trains a Naive Bayes Classifier
classifier_bi8 = NaiveBayesClassifier.train(train_bi_Features8) 
        
#initiates referenceSets and testSets
referenceSets_bi8 = collections.defaultdict(set)
testSets_bi8 = collections.defaultdict(set) 
        
#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(test_bi_Features8):
    referenceSets_bi8[label].add(i)
    predicted8 = classifier_bi8.classify(features)
    testSets_bi8[predicted8].add(i) 
        
#prints metrics to show how well the feature selection did
print 'train accuracy8:', nltk.classify.util.accuracy(classifier_bi8, train_bi_Features8)
print 'test accuracy8:', nltk.classify.util.accuracy(classifier_bi8, test_bi_Features8)
print 'pos precision8:', precision(referenceSets_bi8['pos'], testSets_bi8['pos'])
print 'pos recall8:', recall(referenceSets_bi8['pos'], testSets_bi8['pos'])
print 'neg precision8:', precision(referenceSets_bi8['neg'], testSets_bi8['neg'])
print 'neg recall8:', recall(referenceSets_bi8['neg'], testSets_bi8['neg'])

#Combining 1200 unigrams and 6000 bigrams 
best_pos_uni_bi6 = best_pos_unigrams + best_pos_bigrams6
best_neg_uni_bi6 = best_neg_unigrams + best_neg_bigrams6
total_best_uni_bi6 = best_pos_uni_bi6 + best_neg_uni_bi6

#Combining 1200 unigrams and 7000 bigrams 
best_pos_uni_bi7 = best_pos_unigrams + best_pos_bigrams7
best_neg_uni_bi7 = best_neg_unigrams + best_neg_bigrams7
total_best_uni_bi7 = best_pos_uni_bi7 + best_neg_uni_bi7

#Combining 1200 unigrams and 8000 bigrams 
best_pos_uni_bi8 = best_pos_unigrams + best_pos_bigrams8
best_neg_uni_bi8 = best_neg_unigrams + best_neg_bigrams8
total_best_uni_bi8 = best_pos_uni_bi8 + best_neg_uni_bi8

#creates feature selection mechanism that only uses best unigrams and bigrams    
def best_uni_bi_features6(words):
    return dict([(word, True) for word in words if word in total_best_uni_bi6])

def best_uni_bi_features7(words):
    return dict([(word, True) for word in words if word in total_best_uni_bi7])

def best_uni_bi_features8(words):
    return dict([(word, True) for word in words if word in total_best_uni_bi8])

#creating list of unigrams and bigrams and checking whether best bigrams are present in top 6000 pos and neg uni and bigrams  
pos_uni_bi6 = []
with open("~\\1.csv", 'r') as posSentences:
    for i in posSentences:
        posWord1 = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        posWord1 = ','.join([c for c in posWord1 if c != "'"])
        posWord1 = posWord1.split(",")
        posWord1 = [w for w in posWord1 if not w in stopwords.words('english')]
        pos_bi_Features6a = []
        for word in range(0,len(posWord1)):
            if word < len(posWord1)-1:
                pos_bi_Features6a.append(''.join([posWord1[word], posWord1[word+1]]))
        ab = [best_uni_bi_features6(posWord1 + pos_bi_Features6a), 'pos']
        pos_uni_bi6.append(ab)
pos_uni_bi6 = [x for x in pos_uni_bi6 if x != [{}, 'pos']]       

neg_uni_bi6 = []
with open("~\\0.csv", 'r') as negSentences:
    for i in negSentences:
        negWord1 = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        negWord1 = ','.join([c for c in negWord1 if c != "'"])
        negWord1 = negWord1.split(",")
        negWord1 = [w for w in negWord1 if not w in stopwords.words('english')]
        neg_bi_Features6b = []
        for word in range(0,len(negWord1)):
            if word < len(negWord1)-1:
                neg_bi_Features6b.append(''.join([negWord1[word], negWord1[word+1]]))
        bc = [best_uni_bi_features6(negWord1 + neg_bi_Features6b), 'neg']
        neg_uni_bi6.append(bc)
neg_uni_bi6 = [x for x in neg_uni_bi6 if x != [{}, 'neg']] 

#Splitting into training and test
pos_uni_bi_train6, pos_uni_bi_test6 = train_test_split(pos_uni_bi6, test_size=.25)
neg_uni_bi_train6, neg_uni_bi_test6 = train_test_split(neg_uni_bi6, test_size=.25)
train_uni_bi_Features6 = pos_uni_bi_train6 + neg_uni_bi_train6
test_uni_bi_Features6 = pos_uni_bi_test6 + neg_uni_bi_test6
      
       
#trains a Naive Bayes Classifier
classifier_uni_bi6 = NaiveBayesClassifier.train(train_uni_bi_Features6) 
        
#initiates referenceSets and testSets
referenceSets_uni_bi6 = collections.defaultdict(set)
testSets_uni_bi6 = collections.defaultdict(set) 
        
#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(test_uni_bi_Features6):
    referenceSets_uni_bi6[label].add(i)
    predicted6a = classifier_uni_bi6.classify(features)
    testSets_uni_bi6[predicted6a].add(i) 
        
#prints metrics to show how well the feature selection did
print 'train accuracy6:', nltk.classify.util.accuracy(classifier_uni_bi6, train_uni_bi_Features6)
print 'test accuracy6:', nltk.classify.util.accuracy(classifier_uni_bi6, test_uni_bi_Features6)
print 'pos precision6:', precision(referenceSets_uni_bi6['pos'], testSets_uni_bi6['pos'])
print 'pos recall6:', recall(referenceSets_uni_bi6['pos'], testSets_uni_bi6['pos'])
print 'neg precision6:', precision(referenceSets_uni_bi6['neg'], testSets_uni_bi6['neg'])
print 'neg recall6:', recall(referenceSets_uni_bi6['neg'], testSets_uni_bi6['neg'])


#creating list of unigrams and bigrams and checking whether best bigrams are present in top 7000 pos and neg uni and bigrams  
pos_uni_bi7 = []
with open("~\\1.csv", 'r') as posSentences:
    for i in posSentences:
        posWord1 = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        posWord1 = ','.join([c for c in posWord1 if c != "'"])
        posWord1 = posWord1.split(",")
        posWord1 = [w for w in posWord1 if not w in stopwords.words('english')]
        pos_bi_Features7a = []
        for word in range(0,len(posWord1)):
            if word < len(posWord1)-1:
                pos_bi_Features7a.append(''.join([posWord1[word], posWord1[word+1]]))
        cd = [best_uni_bi_features7(posWord1 + pos_bi_Features7a), 'pos']
        pos_uni_bi7.append(cd)
pos_uni_bi7 = [x for x in pos_uni_bi7 if x != [{}, 'pos']]       

neg_uni_bi7 = []
with open("~\\0.csv", 'r') as negSentences:
    for i in negSentences:
        negWord1 = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        negWord1 = ','.join([c for c in negWord1 if c != "'"])
        negWord1 = negWord1.split(",")
        negWord1 = [w for w in negWord1 if not w in stopwords.words('english')]
        neg_bi_Features7b = []
        for word in range(0,len(negWord1)):
            if word < len(negWord1)-1:
                neg_bi_Features7b.append(''.join([negWord1[word], negWord1[word+1]]))
        de = [best_uni_bi_features7(negWord1 + neg_bi_Features7b), 'neg']
        neg_uni_bi7.append(de)
neg_uni_bi7 = [x for x in neg_uni_bi7 if x != [{}, 'neg']] 

#Splitting into training and test
pos_uni_bi_train7, pos_uni_bi_test7 = train_test_split(pos_uni_bi7, test_size=.25)
neg_uni_bi_train7, neg_uni_bi_test7 = train_test_split(neg_uni_bi7, test_size=.25)
train_uni_bi_Features7 = pos_uni_bi_train7 + neg_uni_bi_train7
test_uni_bi_Features7 = pos_uni_bi_test7 + neg_uni_bi_test7
      
       
#trains a Naive Bayes Classifier
classifier_uni_bi7 = NaiveBayesClassifier.train(train_uni_bi_Features7) 
        
#initiates referenceSets and testSets
referenceSets_uni_bi7 = collections.defaultdict(set)
testSets_uni_bi7 = collections.defaultdict(set) 
        
#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(test_uni_bi_Features7):
    referenceSets_uni_bi7[label].add(i)
    predicted7a = classifier_uni_bi7.classify(features)
    testSets_uni_bi7[predicted7a].add(i) 
        
#prints metrics to show how well the feature selection did
print 'train accuracy7:', nltk.classify.util.accuracy(classifier_uni_bi7, train_uni_bi_Features7)
print 'test accuracy7:', nltk.classify.util.accuracy(classifier_uni_bi7, test_uni_bi_Features7)
print 'pos precision7:', precision(referenceSets_uni_bi7['pos'], testSets_uni_bi7['pos'])
print 'pos recall7:', recall(referenceSets_uni_bi7['pos'], testSets_uni_bi7['pos'])
print 'neg precision7:', precision(referenceSets_uni_bi7['neg'], testSets_uni_bi7['neg'])
print 'neg recall7:', recall(referenceSets_uni_bi7['neg'], testSets_uni_bi7['neg'])


#creating list of unigrams and bigrams and checking whether best bigrams are present in top 8000 pos and neg uni and bigrams  
pos_uni_bi8 = []
with open("~\\1.csv", 'r') as posSentences:
    for i in posSentences:
        posWord1 = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        posWord1 = ','.join([c for c in posWord1 if c != "'"])
        posWord1 = posWord1.split(",")
        posWord1 = [w for w in posWord1 if not w in stopwords.words('english')]
        pos_bi_Features8a = []
        for word in range(0,len(posWord1)):
            if word < len(posWord1)-1:
                pos_bi_Features8a.append(''.join([posWord1[word], posWord1[word+1]]))
        ef = [best_uni_bi_features8(posWord1 + pos_bi_Features8a), 'pos']
        pos_uni_bi8.append(ef)
pos_uni_bi8 = [x for x in pos_uni_bi8 if x != [{}, 'pos']]       

neg_uni_bi8 = []
with open("~\\0.csv", 'r') as negSentences:
    for i in negSentences:
        negWord1 = re.findall(r"[a-zA-Z]+|['']", i.rstrip())         
        negWord1 = ','.join([c for c in negWord1 if c != "'"])
        negWord1 = negWord1.split(",")
        negWord1 = [w for w in negWord1 if not w in stopwords.words('english')]
        neg_bi_Features8b = []
        for word in range(0,len(negWord1)):
            if word < len(negWord1)-1:
                neg_bi_Features8b.append(''.join([negWord1[word], negWord1[word+1]]))
        fg = [best_uni_bi_features8(negWord1 + neg_bi_Features8b), 'neg']
        neg_uni_bi8.append(fg)
neg_uni_bi8 = [x for x in neg_uni_bi8 if x != [{}, 'neg']] 

#Splitting into training and test
pos_uni_bi_train8, pos_uni_bi_test8 = train_test_split(pos_uni_bi8, test_size=.25)
neg_uni_bi_train8, neg_uni_bi_test8 = train_test_split(neg_uni_bi8, test_size=.25)
train_uni_bi_Features8 = pos_uni_bi_train8 + neg_uni_bi_train8
test_uni_bi_Features8 = pos_uni_bi_test8 + neg_uni_bi_test8
      
       
#trains a Naive Bayes Classifier
classifier_uni_bi8 = NaiveBayesClassifier.train(train_uni_bi_Features8) 
        
#initiates referenceSets and testSets
referenceSets_uni_bi8 = collections.defaultdict(set)
testSets_uni_bi8 = collections.defaultdict(set) 
        
#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
for i, (features, label) in enumerate(test_uni_bi_Features8):
    referenceSets_uni_bi8[label].add(i)
    predicted8a = classifier_uni_bi8.classify(features)
    testSets_uni_bi8[predicted8a].add(i) 
        
#prints metrics to show how well the feature selection did
print 'train accuracy8:', nltk.classify.util.accuracy(classifier_uni_bi8, train_uni_bi_Features8)
print 'test accuracy8:', nltk.classify.util.accuracy(classifier_uni_bi8, test_uni_bi_Features8)
print 'pos precision8:', precision(referenceSets_uni_bi8['pos'], testSets_uni_bi8['pos'])
print 'pos recall8:', recall(referenceSets_uni_bi8['pos'], testSets_uni_bi8['pos'])
print 'neg precision8:', precision(referenceSets_uni_bi8['neg'], testSets_uni_bi8['neg'])
print 'neg recall8:', recall(referenceSets_uni_bi8['neg'], testSets_uni_bi8['neg'])

#Creating list with positive & negative probability & given and calculated labels
totallist6 = []
templist6 = []
t6 = {}
for i, (features, label) in enumerate(test_uni_bi_Features6):
    templist6 = []
    dist6 = classifier_uni_bi6.prob_classify(test_uni_bi_Features6[i][0])
    for label in dist6.samples():
        t6 = ("%s: %f" % (label, dist6.prob(label)))
        if (label == 'pos'):
            templist6.insert(0,dist6.prob(label))
        else:
            templist6.insert(1,dist6.prob(label))            
    templist6.insert(2,test_uni_bi_Features6[i][1])
    totallist6.append(templist6)

for sub_list in totallist6:
    if (sub_list[0] > sub_list[1]):
        sub_list.insert(3, 'pos')
    else:
        sub_list.insert(3, 'neg')

totallist7 = []
templist7 = []
t7 = {}
for i, (features, label) in enumerate(test_uni_bi_Features7):
    templist7 = []
    dist7 = classifier_uni_bi7.prob_classify(test_uni_bi_Features7[i][0])
    for label in dist7.samples():
        t7 = ("%s: %f" % (label, dist7.prob(label)))
        if (label == 'pos'):
            templist7.insert(0,dist7.prob(label))
        else:
            templist7.insert(1,dist7.prob(label))            
    templist7.insert(2,test_uni_bi_Features7[i][1])
    totallist7.append(templist7)

for sub_list in totallist7:
    if (sub_list[0] > sub_list[1]):
        sub_list.insert(3, 'pos')
    else:
        sub_list.insert(3, 'neg')       
        
totallist8 = []
templist8 = []
t8 = {}
for i, (features, label) in enumerate(test_uni_bi_Features8):
    templist8 = []
    dist8 = classifier_uni_bi8.prob_classify(test_uni_bi_Features8[i][0])
    for label in dist8.samples():
        t8 = ("%s: %f" % (label, dist8.prob(label)))
        if (label == 'pos'):
            templist8.insert(0,dist8.prob(label))
        else:
            templist8.insert(1,dist8.prob(label))            
    templist8.insert(2,test_uni_bi_Features8[i][1])
    totallist8.append(templist8)

for sub_list in totallist8:
    if (sub_list[0] > sub_list[1]):
        sub_list.insert(3, 'pos')
    else:
        sub_list.insert(3, 'neg')   
        
final_list6 = []
final_list7 = []
final_list8 = []

threshold_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
for i in threshold_list:
    for sub_list in totallist6:
        if (sub_list[0] > i):
            sub_list[3] = 'pos'
        else:
            sub_list[3] = 'neg'
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for sub_list in totallist6:
        if (sub_list[2] == 'pos' and sub_list[3] == 'pos'):
            tp += 1
        if (sub_list[2] == 'neg' and sub_list[3] == 'pos'):
            fp += 1
        if (sub_list[2] == 'neg' and sub_list[3] == 'neg'):
            tn += 1
        if (sub_list[2] == 'pos' and sub_list[3] == 'neg'):
            fn += 1
    final_list6.append([i, tp*1.0/(tp+fn), fp*1.0/(fp+tn)])        
    
for i in threshold_list:
    for sub_list in totallist7:
        if (sub_list[0] > i):
            sub_list[3] = 'pos'
        else:
            sub_list[3] = 'neg'
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for sub_list in totallist7:
        if (sub_list[2] == 'pos' and sub_list[3] == 'pos'):
            tp += 1
        if (sub_list[2] == 'neg' and sub_list[3] == 'pos'):
            fp += 1
        if (sub_list[2] == 'neg' and sub_list[3] == 'neg'):
            tn += 1
        if (sub_list[2] == 'pos' and sub_list[3] == 'neg'):
            fn += 1
    final_list7.append([i, tp*1.0/(tp+fn), fp*1.0/(fp+tn)])   

for i in threshold_list:
    for sub_list in totallist8:
        if (sub_list[0] > i):
            sub_list[3] = 'pos'
        else:
            sub_list[3] = 'neg'
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for sub_list in totallist8:
        if (sub_list[2] == 'pos' and sub_list[3] == 'pos'):
            tp += 1
        if (sub_list[2] == 'neg' and sub_list[3] == 'pos'):
            fp += 1
        if (sub_list[2] == 'neg' and sub_list[3] == 'neg'):
            tn += 1
        if (sub_list[2] == 'pos' and sub_list[3] == 'neg'):
            fn += 1
    final_list8.append([i, tp*1.0/(tp+fn), fp*1.0/(fp+tn)])   

def column(matrix, i):
    return [row[i] for row in matrix]
y6 = column(final_list6, 1)      
x6 = column(final_list6, 2) 
auc6 = auc(x6, y6)

y7 = column(final_list7, 1)      
x7 = column(final_list7, 2) 
auc7 = auc(x7, y7)

y8 = column(final_list8, 1)      
x8 = column(final_list8, 2) 
auc8 = auc(x8, y8)

metrics.auc(fpr, tpr)

plt.plot(x6, y6, 'r', label = "1200uni & 6000bi")
plt.plot(x7, y7, 'b', label = "1200uni & 7000bi")
plt.plot(x8, y8, 'g', label = "1200uni & 8000bi")
plt.suptitle('ROC Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

print auc6 
print auc7
print auc8   
max_auc = max(auc6, auc7, auc8) 
d = {'1200uni & 6000bi' : auc6, '1200uni & 7000bi' : auc7, '1200uni & 8000bi' : auc8} 
w = max(d, key=d.get)
print("We got max AUC {} with {}".format(max_auc, w))



