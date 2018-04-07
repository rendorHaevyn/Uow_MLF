# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 14:01:29 2018
About: Answers to University of Washington Machine Learning, Course 1, Week 2
@author: Admin
"""

from __future__ import print_function
import graphlab as gl
import graphlab.aggregate as agg
import os
os.chdir('c:/users/admin/documents/github/uow/uow_mlf/week 2/')


# GL Setup
gl.product_key.set_product_key('B6B8-4812-53F1-8F29-26F6-2A42-F48C-488D')
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS',4)

# Load data
sf = gl.SFrame('c:/users/admin/documents/github/uow/uow_mlf/week 2/home_data.gl/')

## Question 1:
zip_prc_dict    = sf.groupby(
                    key_columns='zipcode'
                    ,operations={'avg_price':agg.AVG('price')}
                    ).sort('avg_price',ascending=False)[0]
zip_high        = zip_prc_dict['zipcode']
avgprc_high        = zip_prc_dict['avg_price']
print("{}: {} = {} {} ${:,.2f}.".format("QUESTION 1","Highest Price ZipCode",zip_high,"and Avg Price",avgprc_high))

## Question 2:
sf_sl2k     = sf[(sf['sqft_living'] >= 2000) & (sf['sqft_living'] <= 4000)]
sf_sl2k_tp2 = sf[sf['sqft_living'].apply(lambda x: 2000 <= x <= 4000)]
sf_sl2k_tp3 = sf[sf.apply(lambda x: 2000 <= x['sqft_living'] <= 4000)]

sf_slrngpct = sf_sl2k.num_rows() / float(sf.num_rows())

print("{0:}: {1:} = {2:.2%}.".format("QUESTION 2","Pct Homes with Square Ft b/w 2k and 4k",sf_slrngpct))

## Question 3:
advanced_features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
    'condition',        # condition of house				
    'grade',            # measure of quality of construction				
    'waterfront',       # waterfront property				
    'view',             # type of view				
    'sqft_above',       # square feet above ground				
    'sqft_basement',    # square feet in basement				
    'yr_built',         # the year built				
    'yr_renovated',     # the year renovated				
    'lat', 'long',      # the lat-long of the parcel				
    'sqft_living15',    # average sq.ft. of 15 nearest neighbors 				
    'sqft_lot15',       # average lot size of 15 nearest neighbors 
    ]
my_features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode'
    ]

train_data,test_data = sf.random_split(0.8,seed=0)

model_basic     = gl.linear_regression.create(
                    dataset=train_data
                    ,target='price'
                    ,features=my_features
                    ,validation_set=None)
                
model_advanced  = gl.linear_regression.create(
                    dataset=train_data
                    ,target='price'
                    ,features=advanced_features
                    ,validation_set=None)

basic_rmse      = model_basic.evaluate(test_data)['rmse']
advanced_rmse   = model_advanced.evaluate(test_data)['rmse']

print("{}: {} = ${:,.2f}.".format(
                    "QUESTION 3"
                    ,"The RMSE Difference beween advanced and basic models"
                    ,basic_rmse - advanced_rmse))