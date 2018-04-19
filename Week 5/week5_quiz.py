# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 14:01:29 2018
About: Answers to University of Washington Machine Learning, Foundations, Week 5
Note: FWIW, this week's lectures were shithouse.  Recommend Andrew NG ML, any time.
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



##======= Question 1 =======##
# Get user id counts for songs by the following artists:
## 1. Kanye West
nKW_fbs = len(sf[sf['artist'] == 'Kanye West'].unique())
## 2. Foo Fighters
nFF_fbs = len(sf[sf['artist'] == 'Foo Fighters'].unique())
## 3. Taylor Swift
nTS_fbs = len(sf[sf['artist'] == 'Taylor Swift'].unique())
## 4. Lady GaGa
nLG_fbs = len(sf[sf['artist'] == 'Lady GaGa'].unique())

print("{}: {} = {} {} = {} {} = {} {} = {}.".format("QUESTION 1"
      ,"\nUnique Kanye users",nKW_fbs
      ,"\nUnique Foo users",nFF_fbs
      ,"\nUnique Taylor users",nTS_fbs
      ,"\nUnique Lady Dickhead users",nLG_fbs
      ))


##======= Question 2 =======##
# Group by artists and determine song play counts
artist_lst_cnt = sf.groupby(key_columns='artist',operations={'play_count': agg.SUM('listen_count')})
artist_lst_cnt = artist_lst_cnt.sort(['play_count'],ascending=False)

artist_playcnt_high = artist_lst_cnt['artist'][0]
artist_playcnt_low = artist_lst_cnt['artist'][-1]


print("{}: {} = {} {} = {}.".format("QUESTION 2"
      ,"\nArtist with high play count",artist_playcnt_high
      ,"\nArtist with low play count",artist_playcnt_low
      ))


##======= Question 3 =======##
train_data,test_data = sf.random_split(0.8,seed=0)

# create similarity model
sim_model = gl.item_similarity_recommender.create(train_data
                                                  ,user_id='user_id'
                                                  ,item_id='song')
# subset users
subset_test_users = test_data['user_id'].unique()[0:10000]

# recommend a song to each 10k users
subset_song_rec = sim_model.recommend(users=subset_test_users,k=1)
song_pred_cnt = subset_song_rec.groupby(key_columns='song',operations={'rec_count': agg.COUNT()})

most_rec_song = song_pred_cnt.sort('rec_count',ascending=False)['song'][0]
least_rec_song = song_pred_cnt.sort('rec_count',ascending=False)['song'][-1]

print("{}: {} = {} {} = {}.".format("QUESTION 3"
      ,"\nMost recommend song to top 10k users",most_rec_song
      ,"\nLeast recommend song to top 10k user",least_rec_song
      ))
