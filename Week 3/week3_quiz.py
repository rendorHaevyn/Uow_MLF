# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 14:01:29 2018
About: Answers to University of Washington Machine Learning, Foundations, Week 3
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

# Load data
sf = gl.SFrame('amazon_baby.gl/')

# Build word count vector
selected_words = [
    'awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible'
    , 'bad', 'terrible', 'awful', 'wow', 'hate']

sf['word_count'] = gl.text_analytics.count_words(sf['review'])
sf['word_count'].show()


##======= Question 1 =======##
# Iterate words and create named fields if value found in word_count dictionary
for word in selected_words:
    sf[word] = sf.apply(lambda x: 0 if x['word_count'].get(word) == None else x['word_count'].get(word))

dword = {}
for word in selected_words:
    dword[word] = sf[word].sum()

sw_min = min(dword.values())
sw_max = max(dword.values())

dw_min = dword.keys()[dword.values().index(sw_min)]
dw_max = dword.keys()[dword.values().index(sw_max)]

print("{}: {} = {} {} = {}.".format("QUESTION 1","Lowest Sel Wrd Cnt",dw_min,"and Highest Sel Wrd Cnt",dw_max))


##======= Question 2 =======##

# Build sentiment classifer, ignoring ratings of 3
sf_mod = sf[sf['rating'] != 3]

# Create binary value for rating - > 3 = 1, else 0
sf_mod['sentiment'] = sf_mod['rating'] >= 4

# Create train / test datasets
train_data, test_data = sf_mod.random_split(0.8,seed=0)

# Train logisitic regression model
selected_words_model    = gl.logistic_classifier.create(
                        dataset=train_data
                        ,target='sentiment'
                        ,features=selected_words
                        ,validation_set=test_data
                        ,max_iterations=100
                        )

# Extract coefficients and select highest and lowest coefficieny by feature
sw_coeff = selected_words_model['coefficients'].sort('value')['name','value']
coeffword = {}
for row in sw_coeff:
    if row.get('name') in selected_words:
        coeffword[row.get('name')] = row.get('value')

coeff_min = min(coeffword.values())
coeff_max = max(coeffword.values())

coeffword_min = coeffword.keys()[coeffword.values().index(coeff_min)]
coeffword_max = coeffword.keys()[coeffword.values().index(coeff_max)]

print("{}: {} = {} {} = {}.".format("QUESTION 2","Lowest Coeff Sel Wrd",coeffword_min,"and Highest Coeff Sel Wrd",coeffword_max))


##======= Question 3 =======##

# Create sentiment model using word counts as features
sentiment_model         = gl.logistic_classifier.create(
                        dataset=train_data
                        ,target='sentiment'
                        ,features=['word_count']
                        ,validation_set=test_data
                        ,max_iterations=100
                        )

# Evaluate models - 1. word count model
sentiment_model.evaluate(test_data, metric='roc_curve')
sentiment_model.show(view='Evaluation')
sm_pred = sentiment_model.evaluate(test_data, metric='accuracy')['accuracy']

# Evaluate models - 2. selected words model
selected_words_model.evaluate(test_data, metric='roc_curve')
selected_words_model.show(view='Evaluation')
sw_pred = selected_words_model.evaluate(test_data, metric='accuracy')['accuracy']

# Evaluate models - 3. majority class classifier
maj_cls = sf_mod.groupby('sentiment',agg.COUNT)
mc_pred = float(maj_cls.select_column('Count')[1])/maj_cls.select_column('Count').sum()

print("{}: {} = {:.3f} {} = {:.3f} {} = {:.3f}.".format("QUESTION 3","Sentiment Model Accuracy",sm_pred,"Selected Wrd Model Accuracy",sw_pred,"Majority Class Accuracy",mc_pred))


##======= Question 4 =======##
# Subselect review for Baby Trend Diaper Champ product
diaper_champ_reviews = sf_mod[sf_mod['name'] == 'Baby Trend Diaper Champ']
diaper_champ_reviews.shape
diaper_champ_reviews = diaper_champ_reviews.sort('rating',ascending=False)

# Predict sentiment using "sentiment" model
diaper_champ_reviews['predicted_sentiment'] =   sentiment_model.predict(
                                                dataset=diaper_champ_reviews
                                                ,output_type='probability'
                                                )
# For top review, return actual review, word_count vector and predicted sentiment
diaper_champ_reviews[0]['review']
diaper_champ_reviews[0]['word_count']
dc_tp_sent = diaper_champ_reviews['predicted_sentiment'][0]

# Predict sentiment using "selected words" model
diaper_champ_reviews['predicted_selwrds'] =     selected_words_model.predict(
                                                dataset=diaper_champ_reviews
                                                ,output_type='probability'
                                                )
# For top review, return actual review, word_count vector and predicted sentiment
diaper_champ_reviews[0]['review']
diaper_champ_reviews[0]['word_count']
dc_tp_sw = diaper_champ_reviews['predicted_selwrds'][0]

print("{}: {} = {:.9f} {} = {:.9f}.".format("QUESTION 4","Sentiment Model Top Rating Pred Prob",dc_tp_sent,"Selected Wrd Model Top Rating Pred Prob",dc_tp_sw))

## In short, the selected words list is parsimonious and a lot of explanatory information is lost
