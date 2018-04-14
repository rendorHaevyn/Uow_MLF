# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 14:01:29 2018
About: Demo / practice - analysing product sentiment
@author: Admin
"""

from __future__ import print_function
import graphlab as gl
import graphlab.aggregate as agg
import os
os.chdir('c:/users/admin/documents/github/uow/uow_mlf/week 3/')


# GL Setup
pk = open('c:/users/admin/documents/github/gl_product_key.txt','r').read()
gl.product_key.set_product_key(pk)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS',4)

# Load product review data
sf = gl.SFrame('amazon_baby.gl/')


# Explore data
sf.head().show()
sf.show(view='Categorical')

# Build word count vector by review
sf['word_count'] = gl.text_analytics.count_words(sf['review'])
sf['name'].show()

# Giraffe teether
giraffe_reviews = sf[(sf['name'] == 'Vulli Sophie the Giraffe Teether')]
giraffe_reviews.num_rows()
giraffe_reviews['rating'].show(view='Categorical')

# Build Sentiment Classifer

## Data engineer - define positive and negative reviews by dropping all 3*'s

sf_bin = sf[(sf['rating'] != 3)]
len(sf)
len(sf_bin)
sf_bin['sentiment'] = sf_bin['rating'] > 3
sf_bin['sentiment_tp2'] = sf_bin.apply(lambda x: 1 if x['rating'] > 3 else 0)
sf_bin['sentiment'].show(view='Categorical')

## Split data and run logistic regresison model

train_data, test_data = sf_bin.random_split(0.8,seed=0)
sentiment_model = gl.logistic_classifier.create(
                    dataset=train_data
                    ,target='sentiment'
                    ,features=['word_count']
                    ,validation_set=test_data
                    ,max_iterations=100
                    )
## Evaluate model
sentiment_model.evaluate(test_data)
sentiment_model.show(view='Evaluation')

## Apply model to giraffe reviews
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews,output_type='probability')
giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
giraffe_reviews.head()
giraffe_reviews['actual_sentiment'] = giraffe_reviews.apply(lambda x: 1 if x['rating'] > 3 else 0)
giraffe_reviews['result'] = giraffe_reviews['predicted_sentiment'] == giraffe_reviews['actual_sentiment']

# Review some of the top and bottom reviews
giraffe_reviews['review'][0]
giraffe_reviews['review'][1]
giraffe_reviews['review'][-2]
giraffe_reviews['review'][-1]
