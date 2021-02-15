import os
import pandas as pd
import numpy as np
import pickle
from numpy import random
from multiprocessing import Pool
from itertools import combinations, permutations, product

from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

from sklearn import metrics
from sklearn.linear_model import LinearRegression

from scipy.stats import hmean

from tqdm import tqdm

from nltk.corpus import stopwords
import warnings


def preprocessing(df, stopwords):
    # clean text and title and create new column "tokenized"
    tokens = df['title'].apply(simple_preprocess, max_len=30) + df['body'].apply(simple_preprocess, max_len=30)
    # remove stopwords
    tokens = tokens.apply(lambda x: [w for w in x if w not in stopwords])
    # tokens = [w for w in tokens if w not in stopwords]
    return tokens


def get_sample_text(df_text, sample_ids):
    df = df_text.reset_index(drop=True)
    df = df.set_index('id')
    df_sample = df.loc[sample_ids]
    return df_sample


def get_clusters(lda, corpus, min_prob=0.1):
    clusters = [list() for i in range(20)]

    for i, doc in enumerate(corpus):
        topics = lda.get_document_topics(doc, minimum_probability=min_prob)

        # incase not belong to any topic > threshold
        if len(topics) == 0:
            topics = lda.get_document_topics(doc)
            topics = [topics[0]]

        topics = [t[0] for t in topics]

        for topic in topics:
            clusters[topic].append(i)

    return clusters


def get_cluster_change(clusters, sample):
    changes = []

    for i, ids in enumerate(clusters):
        # merge article index
        cluster = sample.iloc[ids]
        counts = cluster['time'].value_counts().sort_index()
        maj_class = cluster['category'].value_counts(normalize=True).index[0]
        # diff = times.diff().fillna(0)
        changes.append((i, maj_class, counts))

    return changes


def load_model(i):
    path = os.path.join('models/lda/lda_models/lda_sample_' + str(i))
    model = LdaModel.load(path)
    dct = Dictionary.load('models/lda/lda_models/lda_sample_' + str(i) + '.id2word')
    return model, dct


if __name__ == '__main__':
    print('read data')
    df = pd.read_json('data/dev/cluster_12_cats.json')
    df['body'] = df['title'] + '. ' + df['body']
    stops_fi = set(stopwords.words('finnish'))
    stops_fi2 = open("data/stopwords_fi_nlf.txt", "r").readlines()
    stops_fi2 = [w.split()[1] for w in stops_fi2]
    stops_fi.update(stops_fi2)
    stops_fi = list(stops_fi)
    print('preprocessing data')
    df['tokens'] = preprocessing(df, stops_fi)

    df.drop(['title', 'body', 'subjects', 'date'], axis=1, inplace=True)

    print('load samples')
    dataset = pickle.load(open("data/dev/dataset_1_event.pkl", "rb"))

    # dataset = list(range(10000))
    # df = pd.DataFrame()

    def do_cache(i):
        # print('start', i)
        sample = dataset[i]
        sample.reset_index(inplace=True, drop=True)
        model, dictionary = load_model(i)
        sample_text = get_sample_text(df, sample['id'])
        corpus = [dictionary.doc2bow(doc) for doc in sample_text['tokens']]
        clusters = get_clusters(model, corpus, min_prob=0.2)

        # # save clusters
        pickle.dump(clusters, open("models/lda_cluster/clusters_" + str(i) + ".pkl", "wb"))
        print('Done:', i)
        return i

    def cache_cluster():
        # done = list(range(613))
        # data = [139, 141, 1147, 1148, 1150, 1151, 1153, 1155, 1156, 1160, 1164, 1183,
        #         1185, 1186, 1193, 1194, 1197, 1198, 1447, 1449, 1455, 1456, 1457, 1460, 1461]
        # missing = done + missing

        # data = [i for i in range(2000) if i not in missing]
        data = range(2000)
        p = Pool(4)

        for i in data:
            # print('start data', i)
            p.apply_async(do_cache, args=(i, ))

        p.close()
        p.join()

    print('do cache')
    cache_cluster()
