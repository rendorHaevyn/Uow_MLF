# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 14:01:29 2018
About: Answers to University of Washington Machine Learning, Foundations, Week 4
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


# compute TF-IDF (term frequency / inverse document frequency) for the corpus (body of works)
sf['word_count'] = gl.text_analytics.count_words(sf['text'])

# Note: tfidf --> 0 for common words, and --> 1 for rare words
tfidf = gl.text_analytics.tf_idf(sf['word_count'])  # SFrame one-stop-shop
sf['tfidf'] = tfidf

##======= Question 1 =======##
# For Elton John, create word count and tfidf table and sorting on value descending
elton = sf[sf['name'] == 'Elton John']
elton_wc_tbl = elton[['word_count']].stack('word_count',new_column_name=['word','count']).sort(['count'],ascending=False)
elton_tfidf_tbl = elton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort(['tfidf'],ascending=False)

ej_wc_wrd_t3    = elton_wc_tbl['word'][0:3][:]
ej_tfidf_wrd_t3 = elton_tfidf_tbl['word'][0:3][:]

print("{}: {} = {} {} = {}.".format("QUESTION 1","Elton John Top 3 Words by Count",ej_wc_wrd_t3,"and Elton John Top 3 Words by TFIDF",ej_tfidf_wrd_t3))


##======= Question 2 =======##
# Calculate TFIDF distance between representative people
vbeckham    = sf[sf['name'] == 'Victoria Beckham']
pmccartney  = sf[sf['name'] == 'Paul McCartney']

# Distance assessment using cosine similarity - use the [0] to refer to dict element
# Note: Lower score must reflect a closer match
cosdist_ej_pm = gl.distances.cosine(elton['tfidf'][0],pmccartney['tfidf'][0])
cosdist_ej_vb = gl.distances.cosine(elton['tfidf'][0],vbeckham['tfidf'][0])

print("{}: {} = {} {} = {}.".format("QUESTION 2","Elton John Cosine Dist to Vic Beckham",cosdist_ej_vb,"and Elton John Cosine Dist to Paul McC",cosdist_ej_pm))


##======= Question 3 =======##
# Build K nearest neighbour models
knn_tfidf_model = gl.nearest_neighbors.create(sf,features=['tfidf'],label='name',distance='cosine')
knn_wrdct_model = gl.nearest_neighbors.create(sf,features=['word_count'],label='name',distance='cosine')

# Apply KNN model for retrieval
ej_knn_tfidf_cosdist    = knn_tfidf_model.query(elton)['reference_label'][0:3]
ej_knn_wrdcnt_cosdist   = knn_wrdct_model.query(elton)['reference_label'][0:3]
vb_knn_tfidf_cosdist    = knn_tfidf_model.query(vbeckham)['reference_label'][0:3]
vb_knn_wrdcnt_cosdist   = knn_wrdct_model.query(vbeckham)['reference_label'][0:3]

print("{}:\n{} = \n{} / \n{},\n{} = \n{} / \n{}.".format("QUESTION 3","Elton John KNN Cosine Dist by TFIDF / WrdCnt",ej_knn_tfidf_cosdist,ej_knn_wrdcnt_cosdist
                                        ,"Vic Beckham KNN Cosine Dist by TFIDF / WrdCnt",vb_knn_tfidf_cosdist,vb_knn_wrdcnt_cosdist))



knn_model.query(beckham)

# Additional document retrieval examples
swift = sf[sf['name'] == 'Taylor Swift']
knn_model.query(swift,label='name')

