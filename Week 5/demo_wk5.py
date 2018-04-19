# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 14:01:29 2018
About: Demo / practice - recommender systems
@author: Admin
"""

from __future__ import print_function
import graphlab as gl
import graphlab.aggregate as agg
import os
os.chdir('c:/users/admin/documents/github/uow/uow_mlf/week 5/')


# GL Setup
pk = open('c:/users/admin/documents/github/gl_product_key.txt','r').read()
gl.product_key.set_product_key(pk)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS',4)

# Load wikipedia people data
sf = gl.SFrame('song_data.gl/')

# Create users array
users = sf['user_id'].unique()


sf.head(4)
sf['song'].show()
lst_songs = sf['song'].unique()
n_songs = len(lst_songs)

# Split into train and test data without any thought to the class split...yeah.

train_data,test_data=sf.random_split(0.8,seed=0)

# Popularity model
pop_model = gl.popularity_recommender.create(train_data
                                             ,user_id='user_id'
                                             ,item_id='song')
pop_model.recommend(users=[users[0]])

# Personalised model
pers_model = gl.item_similarity_recommender.create(train_data
                                                ,user_id='user_id'
                                                ,item_id='song')
pers_model.recommend(users=[users[1]])
pers_model.get_similar_items(items=['With Or Without You - U2'])

# Modal Comp using precision recall
gl.canvas.set_target('browser')
model_perf = gl.recommender.util.compare_models(test_data
                                                ,models=[pop_model,pers_model]
                                                ,model_names=['pop_model','pers_model']
                                                ,user_sample=0.1)
