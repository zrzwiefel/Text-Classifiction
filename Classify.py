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
import ClassifyTraining

file = open('OVR_TextClassifier', 'rb')
classifier = pickle.load(file)

np.set_printoptions(threshold=sys.maxsize)
nlp = en_core_web_sm.load()

target_names = reuters.categories()

def SortNE(ent_list):
    ent_labels = set([X.label_ for X in ent_list])
    ent_dict = {y: [] for y in ent_labels}
    
    for y in [(X.text, X.label_) for X in ent_list]:
        ent_dict['{}'.format(y[1])].append(y[0])
           
    return ent_dict


#def NECount(ent_list):
#    di = SortNE(ent_list)
#    text = [X.text for X in ent_list]
#    y = []
#    
#    for i in di:
#        for j in set(di[i]):
#            y.append((j, text.count(j)))
#    return y


def CleanText(url):
    page = requests.get(url)
    if page.status_code != 404:
        soup = bs4.BeautifulSoup(page.text, 'html.parser')
        raw_text = soup.get_text()
        clean_text = raw_text.translate(str.maketrans('', '', re.sub('\(\)', '', string.punctuation)))
        clean_text = re.sub(r'\n|\s{1,}', ' ', clean_text)
        clean_text =  re.findall(r'(?<=Reuters\)).*?(?=Additional reporting by|Reporting by|Editing by)', clean_text)
        return clean_text
    else:
        return False

def GetNamedEnt(Text):
    text_nlp = nlp(Text[0])    
    x = SortNE(text_nlp.ents)
    return x


def ClassifyText(url_section, url_article, named_ent = False):
    article_text = CleanText(url_section + url_article)
    if article_text == False:
        article_text = (CleanText('https://www.reuters.com/' + url_article))
    article_title = re.sub('-', ' ', url_section + url_article)
    article_title = re.findall(r'(?<=\/){1,1}[A-Za-z0-9_ ]*?(?=idUS)', article_title)
    predicted = classifier.predict(article_text)
    categories = []
    for i, j in zip(target_names, predicted[0]):
        if j == 1:
            categories.append(i)
    if categories == []:
        categories.append('acq')
    if named_ent == True:
        named_ent_list = []
        named_ent_set = []
        for i in ['GPE', 'ORG']:
            if {i}.issubset(set(GetNamedEnt(article_text).keys())):
                named_ent_list.append(GetNamedEnt(article_text)[i])
        for i in named_ent_list:
            x = set(i).difference({'Reuters', 'SolutionsLegalReuters News', 'Thomson Reuters'})
            named_ent_set.append(x)
        return categories, named_ent_set, article_title, [url_section + url_article], article_text
    else:
        return categories, article_title, [url_section + url_article]

print('z')