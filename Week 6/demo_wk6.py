# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 14:01:29 2018
About: Demo / practice - deep neural networks
@author: Admin
"""

from __future__ import print_function
import graphlab as gl
import graphlab.aggregate as agg
import os
os.chdir('c:/users/admin/documents/github/uow/uow_mlf/week 6/')


# GL Setup
pk = open('c:/users/admin/documents/github/gl_product_key.txt','r').read()
gl.product_key.set_product_key(pk)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS',4)

# Load CIFAR-10 dataset
img_train   = gl.SFrame('image_train_data/')
img_test    = gl.SFrame('image_test_data/')

# Explore data
img_train['image'].show()

# Train classifier
nn_model = gl.neuralnet_classifier.create(img_train
                                            ,target='label'
                                            ,features=['deep_features']
                                            ,max_iterations=100
                                            )
                                            
logistic_model = gl.logistic_classifier.create(img_train
                                            ,target='label'
                                            ,features=['image_array']
                                            ,max_iterations=100
                                            )
                        
df_log_model = gl.logistic_classifier.create(img_train
                                            ,target='label'
                                            ,features=['deep_features']
                                            ,max_iterations=100
                                            )

# Test classifer
img_test[0:3]['image'].show() 
img_test[0:3]['label']                  # cat / automobile / cat
logistic_model.predict(img_test[0:3])   # dog / dog / automobile
nn_model.predict(img_test[0:3])         # cat / automobile / cat
df_log_model.predict(img_test[0:3])     # cat / automobile / cat

# Evaluate models
logistic_model.evaluate(img_test)
nn_model.evaluate(img_test)
df_log_model.evaluate(img_test)


# Train Nearest neighbour model for image retrieval with deep features
knn_model = gl.nearest_neighbors.create(img_train
                                                ,features=['deep_features']
                                                ,label='id'
                                                )

cat = img_train[18:19]
cat['image'].show()
knn_model.query(cat)

def get_image(query_result):
    return img_train.filter_by(query_result['reference_label'],'id')

cat_neighbours = get_image(knn_model.query(cat))
cat_neighbours['image'].show()

car = img_train[8:9]
car_neighbours = get_image(knn_model.query(car))['image'].show()
# car_neighbours['image'].show()

# Lambda version
show_neighbours = lambda i: get_image(knn_model.query(img_train[i:i+1]))['image'].show()
show_neighbours(100)

