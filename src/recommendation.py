import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pkgutil import get_data
from get_data import get_data,read_params
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from load_data import load_data

def movie_recommedation(config_path,movie):
    config = read_params(config_path)
    new_df = load_data(config_path)
    tfidf = TfidfVectorizer(strip_accents='ascii',analyzer='word',stop_words='english',max_features=15000)
    vectorizer = tfidf.fit_transform(new_df['Info'])
    similarity = cosine_similarity(vectorizer)
    rec_list = []                                      
    movies = new_df[new_df['Title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[movies])),reverse=True,key = lambda x: x[1])
    for i in distance[1:11]:
        rec_list.append(new_df.iloc[i[0]].Title)
    return rec_list
    

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    movie_recommedation(config_path = parsed_args.config,movie="Dick Johnson Is Dead")