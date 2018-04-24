# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 14:01:29 2018
About: Answers to University of Washington Machine Learning, Foundations, Week 6
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



##======= Question 1 =======##
# Get sketch summary for 'label' column:
ss = img_train['label'].sketch_summary()

# Find lowest value class
min_freq = min(ss.frequent_items().values())
result = [key for key, value in ss.frequent_items().iteritems() if value == min_freq]

print("{}: {} = {} {} = {}.".format("QUESTION 1"
      ,"\nLowest Observation Class",result[0]
      ,"\nWith Obs equal to",min_freq
      ))


##======= Question 2 =======##
# iterate classes - well, couldnt work that bit out WRT creating SFrames
dog_train   = img_train[img_train['label'] == 'dog']
cat_train   = img_train[img_train['label'] == 'cat']
bird_train  = img_train[img_train['label'] == 'bird']
car_train   = img_train[img_train['label'] == 'automobile']

# Train Nearest neighbour model for image retrieval with deep features
dog_knn_mod     = gl.nearest_neighbors.create(dog_train
                                            ,features=['deep_features']
                                            ,label='id'
                                            )
cat_knn_mod     = gl.nearest_neighbors.create(cat_train
                                            ,features=['deep_features']
                                            ,label='id'
                                            )
bird_knn_mod     = gl.nearest_neighbors.create(bird_train
                                            ,features=['deep_features']
                                            ,label='id'
                                            )
car_knn_mod     = gl.nearest_neighbors.create(car_train
                                            ,features=['deep_features']
                                            ,label='id'
                                            )

# Get nearest cat for first test image
test_img_1 = img_test[0:1]
cat_t1_nn = cat_knn_mod.query(test_img_1)['reference_label'][0]
cat_train[cat_train['id']==cat_t1_nn]['image'].show()

# Get nearest dog for first test image
dog_t1_nn = dog_knn_mod.query(test_img_1)['reference_label'][0]
dog_train[dog_train['id']==dog_t1_nn]['image'].show()


print("{}: {} = {} {} = {}.".format("QUESTION 2"
      ,"\nCat closest Nrst Nbr to first test img",cat_t1_nn
      ,"\nDog closest Nrst Nbr to first test img",dog_t1_nn
      ))


##======= Question 3 =======##
# Calculate average distance for cat model nearest neighbours to first image of test dataset
cat_t1_nns = cat_knn_mod.query(test_img_1)
cat_t1_nns_avgdist = cat_t1_nns.groupby(key_columns='query_label',operations={'avg_dist': agg.AVG('distance')})['avg_dist'][0]

# Calculate average distance for dog model nearest neighbours to first image of test dataset
dog_t1_nns = dog_knn_mod.query(test_img_1)
dog_t1_nns_avgdist = dog_t1_nns.groupby(key_columns='query_label',operations={'avg_dist': agg.AVG('distance')})['avg_dist'][0]

print("{}: {} = {:2f} {} = {:2f}.".format("QUESTION 3"
      ,"\nCat closest 1st 5 Nrst Nbrs avg dist to 1st test img",cat_t1_nns_avgdist
      ,"\nDog closest 1st 5 Nrst Nbrs avg dist to 1st test igm",dog_t1_nns_avgdist
      ))


##======= Question 4 =======##
# iterate classes - well, couldnt work that bit out WRT creating SFrames
dog_test    = img_test[img_test['label'] == 'dog']
cat_test    = img_test[img_test['label'] == 'cat']
bird_test   = img_test[img_test['label'] == 'bird']
car_test    = img_test[img_test['label'] == 'automobile']

dog_mod_test_k1 = cat_mod_test_k1 = bird_mod_test_k1 = car_mod_test_k1 = gl.SFrame()
model_lst   = [dog_mod_test_k1,cat_mod_test_k1,bird_mod_test_k1,car_mod_test_k1]
data_lst    = [dog_test,cat_test,bird_test,car_test]
name_lst    = ['dog','cat','bird','car']

# well, I fucked around on this one for a while...and it didnt fucking well work.
for model,data,name in zip(model_lst,data_lst,name_lst):
    print(model,data,name)


for model,data,name in zip(model_lst,data_lst,name_lst):
    model = gl.SFrame({'{}-dog'.format(name):dog_knn_mod.query(data,k=1,label='id')['distance']
                    ,'{}-cat'.format(name):cat_knn_mod.query(data,k=1,label='id')['distance']
                    ,'{}-bird'.format(name):bird_knn_mod.query(data,k=1,label='id')['distance']
                    ,'{}-car'.format(name):car_knn_mod.query(data,k=1,label='id')['distance']
                    })

# dog data
dog_mod_test_k1 = gl.SFrame({'dog':dog_knn_mod.query(dog_test,k=1,label='id')['distance']
                            ,'cat':cat_knn_mod.query(dog_test,k=1,label='id')['distance']
                            ,'bird':bird_knn_mod.query(dog_test,k=1,label='id')['distance']
                            ,'car':car_knn_mod.query(dog_test,k=1,label='id')['distance']
                            })
                            
# cat data
cat_mod_test_k1 = gl.SFrame({'dog':dog_knn_mod.query(cat_test,k=1,label='id')['distance']
                            ,'cat':cat_knn_mod.query(cat_test,k=1,label='id')['distance']
                            ,'bird':bird_knn_mod.query(cat_test,k=1,label='id')['distance']
                            ,'car':car_knn_mod.query(cat_test,k=1,label='id')['distance']
                            })

# bird data
bird_mod_test_k1 = gl.SFrame({'dog':dog_knn_mod.query(bird_test,k=1,label='id')['distance']
                            ,'cat':cat_knn_mod.query(bird_test,k=1,label='id')['distance']
                            ,'bird':bird_knn_mod.query(bird_test,k=1,label='id')['distance']
                            ,'car':car_knn_mod.query(bird_test,k=1,label='id')['distance']
                            })

# car data
car_mod_test_k1 = gl.SFrame({'dog':dog_knn_mod.query(car_test,k=1,label='id')['distance']
                            ,'cat':cat_knn_mod.query(car_test,k=1,label='id')['distance']
                            ,'bird':bird_knn_mod.query(car_test,k=1,label='id')['distance']
                            ,'car':car_knn_mod.query(car_test,k=1,label='id')['distance']
                            })

# dog correct                            
def is_dog_correct(row):
    if row['dog'] < row['cat'] and row['dog'] < row['bird'] and row['dog'] < row['car']:
        return 1
    else:
        return 0
dog_mod_test_k1['verify'] = dog_mod_test_k1.apply(is_dog_correct)
dog_true = dog_mod_test_k1['verify'].sum()

# cat correct
def is_cat_correct(row):
    if row['cat'] < row['dog'] and row['cat'] < row['bird'] and row['cat'] < row['car']:
        return 1
    else:
        return 0
cat_mod_test_k1['verify'] = cat_mod_test_k1.apply(is_cat_correct)
cat_true = cat_mod_test_k1['verify'].sum()

# bird correct
def is_bird_correct(row):
    if row['bird'] < row['cat'] and row['bird'] < row['dog'] and row['bird'] < row['car']:
        return 1
    else:
        return 0
bird_mod_test_k1['verify'] = bird_mod_test_k1.apply(is_bird_correct)
bird_true = bird_mod_test_k1['verify'].sum()

# car correct
def is_car_correct(row):
    if row['car'] < row['cat'] and row['car'] < row['bird'] and row['car'] < row['dog']:
        return 1
    else:
        return 0
car_mod_test_k1['verify'] = car_mod_test_k1.apply(is_car_correct)
car_true = car_mod_test_k1['verify'].sum()


print("{}: {} = {:2%} {} = {:2%} {} = {:2%} {} = {:2%}.".format("QUESTION 4"
      ,"\n% Dog Trues of Dog KNN Model 1K on Dog Test data verus other models",float(dog_true)/dog_test.shape[0]
      ,"\n% Cat Trues of Cat KNN Model 1K on Cat Test data verus other models",float(cat_true)/cat_test.shape[0]
      ,"\n% Bird Trues of Bird KNN Model 1K on Bird Test data verus other models",float(bird_true)/bird_test.shape[0]
      ,"\n% Car Trues of Car KNN Model 1K on Car Test data verus other models",float(car_true)/car_test.shape[0]
      ))
