# %%
#import nltk
#import re
#from nltk.corpus import reuters
#import spacy
#from spacy import displacy
#from collections import Counter
#import en_core_web_sm
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.svm import LinearSVC
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline
#import pandas as pd
#import numpy as np
#import bs4
#import sys
#import string
#import pickle
import Classify


np.set_printoptions(threshold=sys.maxsize)
nlp = en_core_web_sm.load()


# %%
def CleanText(text, sents = False, lower = False):
    if sents == False:
        x = re.sub(r'(?<=[A-Z])\.*', '',reuters.raw(text))
    else: 
        x = re.sub(r'(?<=[A-Z])', '',reuters.raw(text))
    x = re.sub(r'[\n\\>",\(\)]|(\'[a-z])|(\&[a-z]*;)', '', x)
    x = re.sub(r'\s{2,}|-', ' ', x)
    if lower == True:
        x = x.lower()
    return x


def SortNE(ent_list):
    ent_labels = set([X.label_ for X in ent_list])
    ent_dict = {y: [] for y in ent_labels}
    
    for y in [(X.text, X.label_) for X in ent_list]:
        ent_dict['{}'.format(y[1])].append(y[0])
           
    return ent_dict


def NECount(ent_list):
    di = SortNE(ent_list)
    text = [X.text for X in ent_list]
    y = []
    
    for i in di:
        for j in set(di[i]):
            y.append((j, text.count(j)))
    return y


def CreateList(dictionary, list_categories):
    #where dictionary is {'Category Name': int} of length 90
    #text_category is a list (len=sample size) of lists (len=# of categories for that doc)
    cat_list = []
    for j in list_categories:
        sub_list = []
        
        cols = {dictionary[i] for i in j}
        #list of str category values to list of int category values
        for i in dictionary:
            category_sub_list = {dictionary[i]}
            if category_sub_list.issubset(cols):
                sub_list.append(1)
            else:
                sub_list.append(0)
        cat_list.append(sub_list)
    #cat_array = np.array(cat_list)
    return cat_list


# %%
class text:
    def __init__(self, file):
        self.file = file

    def blocktext(self, sents = False, lower = False):
        return CleanText(self.file, sents, lower)

    def words(self):
        return reuters.words(self.file)

    def sents(self):
        return reuters.sents(self.file)

    def clean_words(self):
        return nltk.tokenize.word_tokenize(self.blocktext())

    def words_cat(self): 
        word_list = []
        for y in reuters.categories(self.file):
            for z in self.clean_words():
                word_list.append((z,y))
        return word_list   

    def named_ent(self, count = False):
        x = nlp(self.blocktext(sents = True, lower = True))

        if count == True:
            return NECount(x.ents)
        else:    
            return SortNE(x.ents)
        

# %%
filelist = reuters.fileids()
target_names = reuters.categories()
cat_dict = {reuters.categories()[i]: i for i in range(0, len(reuters.categories()))}
cat_list = CreateList(cat_dict, [reuters.categories(i) for i in filelist])
d = {'Category': [reuters.categories(i) for i in filelist], 
     'File': filelist, 
     'Text': [text(i).blocktext(sents=True) for i in filelist], 
     #'Named Entities': [text(i).named_ent() for i in filelist],
     'Type': [re.sub(r'/[0-9]*', '', i) for i in filelist],
     'Category List': cat_list
    }
df = pd.DataFrame(d)
grouped = df.groupby(df.Type)
train, test = grouped.get_group('training'), grouped.get_group('test')
X_train_list, y_train = train['Text'].tolist(), train['Category List'].tolist()
X_test_list, y_test = test['Text'].tolist(), test['Category List'].tolist()
#X_train_list, y_train = df['Text'].tolist(), df['Category List'].tolist()

classifier1 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
classifier1.fit(X_train_list, y_train)
print(classifier1.score(X_test_list, y_test))

classifier2 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=2000)))])
classifier2.fit(X_train_list, y_train)
print(classifier2.score(X_test_list, y_test))





vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(pd.DataFrame(d)['Text'])
feature_names = np.array(vectorizer.get_feature_names())

def get_top_tfidf_words(response, top_n=2):
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    return feature_names[response.indices[sorted_nzs]]


responses = vectorizer.transform(X_test_list)
#print([get_top_tfidf_words(response,2) for response in responses])



#classifier2 = Pipeline([
#    ('vectorizer', CountVectorizer()),
#    ('tfidf', TfidfTransformer()),
#    ('clf', OneVsRestClassifier(LinearSVC(dual=False, max_iter=2000)))])
#classifier2.fit(X_train_list, y_train)

#print(classifier2.score(X_test_list, y_test))




#file = open('OVR_TextClassifier', 'wb')
#pickle.dump(classifier, file)
#file.close()

