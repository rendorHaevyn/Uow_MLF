# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import graphlab as gl
import matplotlib.pyplot as plt

os.chdir('c:/users/admin/documents/github/uow/uow_mlf/week 2/')


gl.product_key.set_product_key('B6B8-4812-53F1-8F29-26F6-2A42-F48C-488D')
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS',4)
sf = gl.SFrame('c:/users/admin/documents/github/uow/uow_mlf/week 2/home_data.gl/')
sf

gl.canvas.set_target('browser')
sf.show(view="Scatter Plot", x="sqft_living", y="price")
sf.show
train,test = sf.random_split(0.75,seed=0)
train.shape
test.shape

model_0 = gl.linear_regression.create(train,'price',['sqft_living'])

features = ['sqft_living','bathrooms','bedrooms','floors','sqft_lot','zipcode']
sf[features].show()
model_1 = gl.linear_regression.create(train,target='price',features=features,validation_set=None)
sf.show(view='BoxWhisker Plot',x='zipcode',y='price')
print model_0.evaluate(test)
print model_1.evaluate(test)

pred_1 = model_1.predict(test)
resid_1 = pred_1 - test['price']
plt.plot(resid_1.sort(),'.')

plt.plot(test['floors'],test['price'],',',
         test['floors'],pred_1,',')


vals = model_1.predict(test) - test['price']
type(vals)
vals.show()

model_2 = gl.linear_regression.create(train,'price',
         [
         'bedrooms',
         'bathrooms',
         'sqft_living',
         'sqft_lot',
         'floors',
         'waterfront',
         'view',
         'condition',
         'grade',
         'sqft_above',
         'sqft_basement',
         'yr_built',
         'yr_renovated',
         'zipcode',
         'long',
         'sqft_living15',
         'sqft_lot15'])
model_2.evaluate(test)

plt.plot(test['sqft_living'],test['price'],'.',
         test['sqft_living'],model_0.predict(test),'-')
         
model_0.list_fields()
model_0.get('coefficients')
model_0.get('num_features')