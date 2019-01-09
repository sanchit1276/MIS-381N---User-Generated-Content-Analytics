#!/usr/bin/python3.6
#install scapy by using the following command in bash: pip3.6 install --user https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-1.2.0/en_core_web_sm-1.2.0.tar.gz#egg=en_core_web_sm-1.2.0
#link the library by using the following command in bash: python3.6 -m spacy download en
#Name of the review column should be Restaurant_review
import csv
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import PunktSentenceTokenizer,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tempfile import NamedTemporaryFile
import shutil
import warnings
import spacy
#import en_core_web_sm
#nlp = en_core_web_sm.load()

nlp = spacy.load('en')


#to ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#reading the attributes file
#check into the "attributes.txt" file for the proper format
#each attribute has to be listed in a new line.
attributes=list(line.strip() for line in open('attributes.txt'))
attributes=" ".join(attributes)


reviews_df=pd.read_excel("yelp_reviews.xlsx",encoding='utf8', errors='ignore')

#checking for nulls if present any
reviews_df=reviews_df.fillna("NA")

reviews_df['similarity'] = -1

for index, row in reviews_df.iterrows():
    reviews_df.loc[index,'similarity'] = nlp(row["Restaurant_review"]).similarity(nlp(attributes))


#writing to an output file
reviews_df.to_excel("similarity.xlsx",index=False)




