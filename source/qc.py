
import sys
import pandas as pd
import numpy as np
import re, nltk
from sner import Ner
import spacy
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
import spacy
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import fbeta_score, accuracy_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder



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
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
    
    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process
    
    Output : Returns the cleaned text corpus
    
    '''
    cleaned_corpus = pd.Series([],dtype=pd.StringDtype()) 
    for row in corpus:
        qs = []
        for word in row.split():
            if word not in keep_list:
                p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                p1 = p1.lower()
                qs.append(p1)
            else : qs.append(word)
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
    return cleaned_corpus


def preprocess(corpus, keep_list, cleaning = True, stemming = True, stem_type = "snowball", lemmatization = False, remove_stopwords = True):
    
    '''
    Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)
    
    Input : 
    'corpus' - Text corpus on which pre-processing tasks will be performed
    'keep_list' - List of words to be retained during cleaning process
    'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should 
                                                                  be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                  Stemmer. 'snowball' corresponds to Snowball Stemmer
    
    Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together
    
    Output : Returns the processed text corpus
    
    '''
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
        if stem_type == 'snowball':
            stemmer = SnowballStemmer(language = 'english')
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
        else :
            stemmer = PorterStemmer()
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    
    corpus = [' '.join(x) for x in corpus]
        

    return corpus


common_dot_words_found = ['U.S.S.R','D.A','etc.','Inc','No.', 'p.m','U.S.', 'aol.com', 
                        'yahoo.com', 'F.', 'Jan.', 'J.R.R', 'St.', 'W.C.', 'H.G.', 'Mr.', 'D.', 'J.D.', 'LL.M.', 'Answers.com', 'Mrs.', 'E.', 'C.', 'T.V.', '.tbk', 'P.T.', 'A.G', 'Dr.']
all_corpus = preprocess(all_corpus, keep_list = common_dot_words_found, remove_stopwords = True)

train_corpus = all_corpus[0:train.shape[0]]
test_question_corpus = all_corpus[train.shape[0]:]


"""
Loading the English model for Spacy.<br>
NLTK version for the same performs too slowly, hence opting for Spacy.
"""


nlp = spacy.load('en')


"""
# Obtaining Features from Train Data, which would be fed to CountVectorizer

Creating list of Named Entitites, Lemmas, POS Tags, Syntactic Dependency Relation and Orthographic Features using shape.<br>
Later, these would be used as features for our model.
"""


all_ner = []
all_lemma = []
all_tag = []
all_dep = []
all_shape = []
for row in train_corpus:
    doc = nlp(row)
    present_lemma = []
    present_tag = []
    present_dep = []
    present_shape = []
    present_ner = []
    #print(row)
    for token in doc:
        present_lemma.append(token.lemma_)
        present_tag.append(token.tag_)
        #print(present_tag)
        present_dep.append(token.dep_)
        present_shape.append(token.shape_)
    all_lemma.append(" ".join(present_lemma))
    all_tag.append(" ".join(present_tag))
    all_dep.append(" ".join(present_dep))
    all_shape.append(" ".join(present_shape))
    for ent in doc.ents:
        present_ner.append(ent.label_)
    all_ner.append(" ".join(present_ner))


"""
Converting the attributes obtained above into vectors using CountVectorizer.
"""


count_vec_ner = CountVectorizer(ngram_range=(1, 2)).fit(all_ner)
ner_ft = count_vec_ner.transform(all_ner)
count_vec_lemma = CountVectorizer(ngram_range=(1, 2)).fit(all_lemma)
lemma_ft = count_vec_lemma.transform(all_lemma)
count_vec_tag = CountVectorizer(ngram_range=(1, 2)).fit(all_tag)
tag_ft = count_vec_tag.transform(all_tag)
count_vec_dep = CountVectorizer(ngram_range=(1, 2)).fit(all_dep)
dep_ft = count_vec_dep.transform(all_dep)
count_vec_shape = CountVectorizer(ngram_range=(1, 2)).fit(all_shape)
shape_ft = count_vec_shape.transform(all_shape)


"""
Combining the features obtained into 1 matrix
"""


x_all_ft_train = hstack([ner_ft, lemma_ft, tag_ft])


x_all_ft_train


"""
Converting from COOrdinate format to Compressed Sparse Row format for easier mathematical computations.
"""


x_all_ft_train = x_all_ft_train.tocsr()
x_all_ft_train


"""
# Now we will obtain the Feature vectors for the test_question set using the CountVectorizers Obtained from the Training Corpus
"""


all_test_question_ner = []
all_test_question_lemma = []
all_test_question_tag = []
all_test_question_dep = []
all_test_question_shape = []
for row in test_question_corpus:
    doc = nlp(row)
    present_lemma = []
    present_tag = []
    present_dep = []
    present_shape = []
    present_ner = []
    for token in doc:
        present_lemma.append(token.lemma_)
        present_tag.append(token.tag_)
        present_dep.append(token.dep_)
        present_shape.append(token.shape_)
    all_test_question_lemma.append(" ".join(present_lemma))
    all_test_question_tag.append(" ".join(present_tag))
    all_test_question_dep.append(" ".join(present_dep))
    all_test_question_shape.append(" ".join(present_shape))
    for ent in doc.ents:
        present_ner.append(ent.label_)
    all_test_question_ner.append(" ".join(present_ner))


ner_test_question_ft = count_vec_ner.transform(all_test_question_ner)
lemma_test_question_ft = count_vec_lemma.transform(all_test_question_lemma)
tag_test_question_ft = count_vec_tag.transform(all_test_question_tag)
dep_test_question_ft = count_vec_dep.transform(all_test_question_dep)
shape_test_question_ft = count_vec_shape.transform(all_test_question_shape)


x_all_ft_test_question = hstack([ner_test_question_ft, lemma_test_question_ft, tag_test_question_ft])





model = svm.LinearSVC()
f = open("predicted-labels.txt", "w")


if taxonomy=='-fine':
        
    model.fit(x_all_ft_train, train['QType'])
    prediction = model.predict(x_all_ft_test_question)
    
    for line in prediction:
        f.write(line+'\n')
elif taxonomy=='-coarse':
    model.fit(x_all_ft_train, train['QType-Coarse'])
    prediction = model.predict(x_all_ft_test_question)
    for line in prediction:
       f.write(line+'\n')
else:
    print("wrong taxonomy");
    exit;



