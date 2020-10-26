
import sys
import pandas as pd
import numpy as np
import re, nltk
import spacy
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import svm
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

print("Computing...")

taxonomy=sys.argv[1]
f_train = open(sys.argv[2],'r+')
f_test_question = open(sys.argv[3], 'r+')

train = pd.DataFrame(f_train.readlines(), columns = ['Question'])
test_question = pd.DataFrame(f_test_question.readlines(), columns = ['Question'])

train['QType'] = train.Question.apply(lambda x: x.split(' ', 1)[0])
train['Question'] = train.Question.apply(lambda x: x.split(' ', 1)[1])
train['QType-Coarse'] = train.QType.apply(lambda x: x.split(':')[0])


all_corpus = pd.Series(train.Question.tolist() + test_question.Question.tolist()).astype(str)


def text_clean(corpus, keep_list):


    cleaned_corpus = pd.Series([],dtype=pd.StringDtype()) 
    for row in corpus:
        qs = []
        for word in row.split():
            if word not in keep_list:
                p1 = re.sub(r'[^\w\s]','',str(word).lower())
                qs.append(p1)
            else : qs.append(word)
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
    return cleaned_corpus


def preprocess(corpus, keep_list, cleaning = True, stemming = True, lemmatization = False, remove_stopwords = True):

    if cleaning == True:
        corpus = text_clean(corpus, keep_list)
    
    if remove_stopwords == True:
        wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
        stop = set(stopwords.words('english'))
        for word in wh_words:
            stop.remove(word)
        corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    else :
        corpus = [[x for x in x.split()] for x in corpus]
    
    if lemmatization == True:
        lem = WordNetLemmatizer()
        corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    
    if stemming == True:
        stemmer = PorterStemmer()
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    
    corpus = [' '.join(x) for x in corpus]
        

    return corpus


common_dot_words_found = ['U.S.S.R','D.A','etc.','Inc','No.', 'p.m','U.S.', 'aol.com', 
                        'yahoo.com', 'F.', 'Jan.', 'J.R.R', 'St.', 'W.C.', 'H.G.', 'Mr.', 'D.', 'J.D.', 'LL.M.', 'Answers.com', 'Mrs.', 'E.', 'C.', 'T.V.', '.tbk', 'P.T.', 'A.G', 'Dr.']
all_corpus = preprocess(all_corpus, common_dot_words_found)

train_corpus = all_corpus[0:train.shape[0]]
test_question_corpus = all_corpus[train.shape[0]:]


"""
Loading the English model for Spacy.<br>
NLTK version for the same performs too slowly, hence opting for Spacy.
"""


nlp = spacy.load('en')


"""
# Obtaining Features from Train Data, which would be fed to CountVectorizer

Lemmas, POS Tags and Orthographic Features using shape.<br>
"""



all_lemma = []
all_tag = []
all_shape = []
for row in train_corpus:
    doc = nlp(row)
    present_lemma = []
    present_tag = []
    present_shape = []
    for token in doc:
        present_lemma.append(token.lemma_)
        present_tag.append(token.tag_)
        present_shape.append(token.shape_)
    all_lemma.append(" ".join(present_lemma))
    all_tag.append(" ".join(present_tag))
    all_shape.append(" ".join(present_shape))


"""
Converting the attributes obtained above into vectors using CountVectorizer.
ngram_range=(1, 2)=unigram + bigram
""" 



count_vec_lemma = CountVectorizer(ngram_range=(1, 2)).fit(all_lemma)
count_vec_tag = CountVectorizer(ngram_range=(1, 2)).fit(all_tag)
count_vec_shape = CountVectorizer(ngram_range=(1, 2)).fit(all_shape)
lemma_ft = count_vec_lemma.transform(all_lemma)
tag_ft = count_vec_tag.transform(all_tag)
shape_ft = count_vec_shape.transform(all_shape)


"""
Combining the features obtained into 1 matrix
"""


x_all_ft_train = hstack([lemma_ft, tag_ft,shape_ft])

"""
# Now we will obtain the Feature vectors for the test_question set using the CountVectorizers Obtained from the Training Corpus
"""


all_test_question_lemma = []
all_test_question_tag = []
all_test_question_shape = []
for row in test_question_corpus:
    doc = nlp(row)
    present_lemma = []
    present_tag = []
    present_shape = []
    for token in doc:
        present_lemma.append(token.lemma_)
        present_tag.append(token.tag_)
        present_shape.append(token.shape_)
    all_test_question_lemma.append(" ".join(present_lemma))
    all_test_question_tag.append(" ".join(present_tag))
    all_test_question_shape.append(" ".join(present_shape))



lemma_test_question_ft = count_vec_lemma.transform(all_test_question_lemma)
tag_test_question_ft = count_vec_tag.transform(all_test_question_tag)
shape_test_question_ft = count_vec_shape.transform(all_test_question_shape)


x_all_ft_test_question = hstack([lemma_test_question_ft, tag_test_question_ft, shape_test_question_ft])





model = svm.LinearSVC()


if taxonomy=='-fine':
    f = open("develop68-fine.txt", "w")

    
    model.fit(x_all_ft_train, train['QType'])
    prediction = model.predict(x_all_ft_test_question)
    for line in prediction:
        f.write(line+'\n')
elif taxonomy=='-coarse':
    f = open("develop68-coarse.txt", "w")

    model.fit(x_all_ft_train, train['QType-Coarse'])
    prediction = model.predict(x_all_ft_test_question)
    for line in prediction:
       f.write(line+'\n')
else:
    print("wrong taxonomy");
    exit;



