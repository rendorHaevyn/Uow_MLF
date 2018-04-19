# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 14:01:29 2018
About: Demo / practice - clustering analysis
@author: Admin
"""

from __future__ import print_function
import graphlab as gl
import graphlab.aggregate as agg
import os
os.chdir('c:/users/admin/documents/github/uow/uow_mlf/week 4/')


# GL Setup
pk = open('c:/users/admin/documents/github/gl_product_key.txt','r').read()
gl.product_key.set_product_key(pk)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS',4)

# Load wikipedia people data
sf = gl.SFrame('people_wiki.gl/')

sf.head(4)

obama = sf[sf['name'] == 'Barack Obama']
clooney = sf[sf['name'] == 'George Clooney']

obama['word_count'] = gl.text_analytics.count_words(obama['text'])
len(obama['word_count'])
obama['word_count'][0].keys()

# sort word counts for obama - we can only use "stack" function as part of SFrame
obama_wc_table = obama[['word_count']].stack('word_count',new_column_name=['word','count'])
obama_wc_table = obama_wc_table.sort(['count'],ascending=False)
obama_wc_table.head(2)

# compute TF-IDF (term frequency / inverse document frequency) for the corpus (body of works)
sf['word_count'] = gl.text_analytics.count_words(sf['text'])

# Note: tfidf --> 0 for common words, and --> 1 for rare words
tfidf = gl.text_analytics.tf_idf(sf['word_count'])  # SFrame one-stop-shop
sf['tfidf'] = tfidf

# repeat for obama, creating tfidf table and sorting on value descending
obama = sf[sf['name'] == 'Barack Obama']
obama_tfidf_table = obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf_value']).sort(['tfidf_value'],ascending=False)

# Calculate TFIDF distance between representative people
clinton = sf[sf['name'] == 'Bill Clinton']
beckham = sf[sf['name'] == 'David Beckham']
trump = sf[sf['name'] == 'Donald Trump']


# Distance assessment using cosine similarity - use the [0] to refer to dict element
# Note: Lower score must reflect a closer match
gl.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])
gl.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])

# Build K nearest neighbour model
knn_model = gl.nearest_neighbors.create(sf,features=['tfidf'],label='name')

# Apply KNN model for retrieval
knn_model.query(obama)
knn_model.query(clinton)
knn_model.query(beckham)

# Additional document retrieval examples
swift = sf[sf['name'] == 'Taylor Swift']
knn_model.query(swift,label='name')

