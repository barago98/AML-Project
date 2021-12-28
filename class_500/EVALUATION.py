# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:23:40 2021

@author: debod

The EVALUATION file is primarily used for the models to be fetched and run. 
We divide this file into two parts. 1) With GeM pooling and 2) Without GeM pooling.
In each of the situations, we also introduce an ensemble model with which we see a steady
increase in the accuracy.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from gempool import *
import seaborn as sns
import random
import tqdm


    
test_dir = 'testing'
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(120, 120),
    batch_size = 1,
    class_mode='categorical',
    seed=42)

def get_label(arr):
    for i in range(len(arr)):
        if arr[i] == max(arr):
            return test_label[i]
        
test_label = list(test_generator.class_indices.keys())

# Label from original data
true_labels=[]
for i in tqdm.tqdm(range(19640)):
    labels=get_label(test_generator[i][1][0])
    labels=int(labels)
    true_labels.append(labels)

test_label = list(test_generator.class_indices.keys())
len(test_label)
####################### 1st Model: ResNet50 ################################

resnet500 = tf.keras.models.load_model('checkpoint_resnet_500.h5', 
                                       custom_objects={'GeMPooling2D': GeMPooling2D})

# Predicting the label 
predict_labels_rn=[]
for i in tqdm.tqdm(range(19640)):
    pred_labels_1=get_label(np.array(resnet500.predict(test_generator[i][0])[0]))
    #pred_labels_2=np.array(model_mobile.predict(test_generator[i][0])[0])
    predict_labels_rn.append(pred_labels_1)

true_labels=[int(i) for i in true_labels]
predict_labels_rn=[int(i) for i in predict_labels_rn]   

# resnet accuracy  
res_acc=accuracy_score(true_labels, predict_labels_rn)
print('ResNEt 50 Accuracy:',res_acc)

######################## 2nd Model: MobileNetV2 #############################

MobileNet500 = tf.keras.models.load_model("checkpoint_mobilenet.h5", 
                                       custom_objects={'GeMPooling2D': GeMPooling2D})

    
# Predicting the label 
predict_labels_mn=[]
for i in tqdm.tqdm(range(19640)):
    pred_labels_1=get_label(np.array(MobileNet500.predict(test_generator[i][0])[0]))
    #pred_labels_2=np.array(model_mobile.predict(test_generator[i][0])[0])
    predict_labels_mn.append(pred_labels_1)

true_labels=[int(i) for i in true_labels]
predict_labels_mn=[int(i) for i in predict_labels_mn]    

# mobilenet accuracy 
mn_acc=accuracy_score(true_labels, predict_labels_mn)
print('MobileNetV2 Accuracy:',mn_acc)

######################## 3rd Model: DenseNet201 #############################

DenseNet500 = tf.keras.models.load_model("checkpoint_densenet.h5", 
                                       custom_objects={'GeMPooling2D': GeMPooling2D})

    
# Predicting the label 
predict_labels_dn=[]
for i in tqdm.tqdm(range(19640)):
    pred_labels_1=get_label(np.array(DenseNet500.predict(test_generator[i][0])[0]))
    #pred_labels_2=np.array(model_mobile.predict(test_generator[i][0])[0])
    predict_labels_dn.append(pred_labels_1)

true_labels=[int(i) for i in true_labels]
predict_labels_dn=[int(i) for i in predict_labels_dn]    

# densenet accuracy 
dn_acc=accuracy_score(true_labels, predict_labels_dn)
print('Densenet Accuracy:',dn_acc)

######################## 4th Model: VGG16 #############################

Vgg500 = tf.keras.models.load_model("checkpoint_vgg16.h5", 
                                       custom_objects={'GeMPooling2D': GeMPooling2D})

    
# Predicting the label 
predict_labels_vgg=[]
for i in tqdm.tqdm(range(19640)):
    pred_labels_1=get_label(np.array(Vgg500.predict(test_generator[i][0])[0]))
    #pred_labels_2=np.array(model_mobile.predict(test_generator[i][0])[0])
    predict_labels_vgg.append(pred_labels_1)

true_labels=[int(i) for i in true_labels]
predict_labels_vgg=[int(i) for i in predict_labels_vgg]    

# densenet accuracy 
vgg_acc=accuracy_score(true_labels, predict_labels_vgg)
print('Vgg Accuracy:',vgg_acc)

######################### Ensemble model ####################################

predict_labels_ensemble=[]
for i in tqdm.tqdm(range(19640)):
    pred_labels_1=np.array(resnet500.predict(test_generator[i][0])[0])
    pred_labels_2=np.array(MobileNet500.predict(test_generator[i][0])[0])
    pred_labels_3=np.array(DenseNet500.predict(test_generator[i][0])[0])
    pred_labels_4=np.array(Vgg500.predict(test_generator[i][0])[0])
    pred_labels=get_label(np.sum([pred_labels_1,pred_labels_2, pred_labels_3, pred_labels_4], axis=0))
    predict_labels_ensemble.append(pred_labels)
    
true_labels=[int(i) for i in true_labels]
predict_labels_ensemble=[int(i) for i in predict_labels_ensemble]   
ens_acc=accuracy_score(true_labels, predict_labels_ensemble) 
print('Ensemble Accuracy', ens_acc)

######################### Visual Comparison ###########################
acc_list=[res_acc,mn_acc,dn_acc,vgg_acc, ens_acc ]
acc_names=['ResNet50', 'MobileNetV2', 'DenseNet', 'VGG16', 'Ensemble']
sns.set_theme(style="whitegrid", palette="husl")
fig=sns.barplot(acc_names, acc_list)
fig.set(xlabel='Models',
        ylabel='Accuracy',
        title='Comparision of accuracies')

                   ############## Testing images ##################
# Test evaluation for image index 20

def predictor():
    '''
    The predictor function takes no argument. It initially chooses a random number
    between 0 to 7999, the number of images in the test data. Based on the index number, 
    we will check whether the original label and the predicted label matches.
    Finally, the function returns the randomly chosen index number and the predicted label.
    '''
    
    image_index=random.randint(0,19639)
    print('__________________________________________________\n')
    print('The index number is:\n',image_index)
    print('__________________________________________________\n')
    #image_index = 7999 # choose a image (0-8000)
    image = test_generator[image_index][0] 
    image = image.reshape((120,120,3))
    plt.imshow(image)
    plt.show()
    print('__________________________________________________\n')
    #predict_labels_ensemble=[]
    pred_labels_1=np.array(resnet500.predict(test_generator[image_index][0])[0])
    pred_labels_2=np.array(MobileNet500.predict(test_generator[image_index][0])[0])
    pred_labels_3=np.array(DenseNet500.predict(test_generator[image_index][0])[0])
    pred_labels_4=np.array(Vgg500.predict(test_generator[image_index][0])[0])
    pred_labels=get_label(np.sum([pred_labels_1,pred_labels_2, pred_labels_3, pred_labels_4], axis=0))
    #predict_labels_ensemble.append(pred_labels)
    
    true_label=get_label(test_generator[image_index][1][0])
    print('__________________________________________________\n')
    print('Original label', true_label)
    print('Predicted label', pred_labels)
    print('__________________________________________________\n')
    if true_label==pred_labels:
        print('INDEX MATCHED: CORRECT PREDICTION')
    else:
        print('INDEX DID NOT MATCH: WRONG PREDICTION')
    print('__________________________________________________\n')
    return [pred_labels, image_index]
    

def display_related_images(image_index):
    print('Creating the image list with same label...')
    image_list = [] # image with the same label
    for i in tqdm.tqdm(range(19640)):
        if get_label(test_generator[i][1][0]) == label[0]:
            image_list.append(i)
            
    if len(image_list) > 10:
        for i in (range(10)):
            if i != image_index:
                image = test_generator[image_list[i]][0]
                image = image.reshape((120,120,3))
                plt.imshow(image)
                plt.show()
            
    else:
        for i in range(len(image_list)):
            if i != image_index:
                image = test_generator[image_list[i]][0]
                image = image.reshape((120,120,3))
                plt.imshow(image)
                plt.show() 
 
label = predictor() 
display_related_images(label[1])
