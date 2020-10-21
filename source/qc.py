'''
Created on 13/10/2020

@author: yu
'''
import nltk
import csv
import pickle
import sys
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from scipy.constants.codata import txt2002

taxonomy=sys.argv[1]
train=open(sys.argv[2],'r')   #Reads in the training questions text file
dev=open(sys.argv[3],'r')  #Reads in the development question text file


train_data=train.read()
train_sent=train_data.splitlines()                                            #split the training set into its corresponding 
all_sentence=""
coarse_fine_list=[]
for line in train_sent:
    coarse_fine_list.append(line.split(None, 1)[0])
for line in train_sent:
    all_sentence= all_sentence + line.split(None, 1)[1] + '\n'

final_set=[]
all_words1=[]
token=nltk.RegexpTokenizer(r'\w+')                  #the word tokenizer that does not read in punctuation
all_words=token.tokenize(all_sentence)  
for j in all_words:                                  
    if j.isdigit() is False:                        #Read in only non numerical words present in the entire train set
        all_words1.append(j)
e=0
for i in train_sent:                    # Creates a list of list of lists with words of each question and the 
    words=[]                            # corresponding label [0-6]
    set1=[]
    words=nltk.word_tokenize(i)
    set1.append(words[2:]) 
    set1.append(coarse_fine_list[e])
    final_set.append(set1)
    e=e+1
    
    

all_words2=nltk.FreqDist(all_words1)    #The frequency distribution of all of the words present in the train file
word_features=list(all_words2.keys())
print(len(word_features))



def find_features(sent):                # Finding the features of each question and storing it as a dictionary
    words2=set(sent)
    features={}
    for w in word_features:
        features[w]=(w in words2)
    return features

featuresets=[(find_features(rev),category) for (rev, category) in final_set]



training_set=featuresets[:3200]
#testing_set=featuresets[3200:]
#Split of 80:20 for training and testing set



LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
#print(nltk.classify.accuracy(LinearSVC_classifier, testing_set))



test_data=dev.read()                          #data
test_set=test_data.splitlines()



final_test=[]                               #Putting all the words in the same form as that for training data
for i in test_set:
    words=[]
    set1=[]
    words=nltk.word_tokenize(i)
    set1.append(words) 
    final_test.append(set1)



answer=[]




id1=301                                   #Predicting for all the testing data and writing it in a list
for r in final_test:
    prediction=LinearSVC_classifier.classify(find_features(r[0]))    
    answer.append(prediction)



f = open("Prediction.txt", "w")
for line in answer:
    f.write(line+'\n')




