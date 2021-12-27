# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 17:23:40 2021

@author: debod

The EVALUATION file is primarily used for the models to be fetched and run. 
We devide this file into two parts. 1) We compare the accuracy of the individual models 
and the ensemble model. 2) We pick the model with highest accuracy and use it to 
predict the class of an randomly picked image.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import tqdm
from gp2d import *
import seaborn as sns

test_dir = 'testing'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size = 1,
    class_mode='categorical',
    seed=00)

test_label = list(test_generator.class_indices.keys())
len(test_label)

def get_label(arr):
    for i in range(len(arr)):
        if arr[i] == max(arr):
            return test_label[i]

# Label from original data
true_labels=[]
for i in tqdm.tqdm(range(8000)):
    labels=get_label(test_generator[i][1][0])
    labels=int(labels)
    true_labels.append(labels)

####################### 1st Model: ResNet50 ################################

resnet200 = tf.keras.models.load_model('checkpoint_resnet200.h5', 
                                       custom_objects={'GeMPooling2D': GeMPooling2D})

# Predicting the label 
predict_labels_rn=[]
for i in tqdm.tqdm(range(8000)):
    pred_labels_1=get_label(np.array(resnet200.predict(test_generator[i][0])[0]))
    #pred_labels_2=np.array(model_mobile.predict(test_generator[i][0])[0])
    predict_labels_rn.append(pred_labels_1)

true_labels=[int(i) for i in true_labels]
predict_labels_rn=[int(i) for i in predict_labels_rn]   

# resnet accuracy  
res_acc=accuracy_score(true_labels, predict_labels_rn)
print('ResNEt 50 Accuracy:',res_acc)


######################## 2nd Model: MobileNetV2 #############################

MobileNet200 = tf.keras.models.load_model("checkpoint_mobilenet.h5", 
                                       custom_objects={'GeMPooling2D': GeMPooling2D})

    
# Predicting the label 
predict_labels_mn=[]
for i in tqdm.tqdm(range(8000)):
    pred_labels_1=get_label(np.array(MobileNet200.predict(test_generator[i][0])[0]))
    #pred_labels_2=np.array(model_mobile.predict(test_generator[i][0])[0])
    predict_labels_mn.append(pred_labels_1)

true_labels=[int(i) for i in true_labels]
predict_labels_mn=[int(i) for i in predict_labels_mn]    
mn_acc=accuracy_score(true_labels, predict_labels_mn)
print('MobileNetV2 Accuracy:',accuracy_score(true_labels, predict_labels_mn))


######################## 3rd Model: DenseNet121 #############################

DenseNet200 = tf.keras.models.load_model("checkpoint_densenet.h5", 
                                       custom_objects={'GeMPooling2D': GeMPooling2D})

    
# Predicting the label 
predict_labels_dn=[]
for i in tqdm.tqdm(range(8000)):
    pred_labels_1=get_label(np.array(DenseNet200.predict(test_generator[i][0])[0]))
    #pred_labels_2=np.array(model_mobile.predict(test_generator[i][0])[0])
    predict_labels_dn.append(pred_labels_1)

true_labels=[int(i) for i in true_labels]
predict_labels_dn=[int(i) for i in predict_labels_dn]    
dn_acc=accuracy_score(true_labels, predict_labels_dn)
print('DenseNet Accuracy:',accuracy_score(true_labels, predict_labels_dn))


######################## 4th Model: VGG16 #################################

vgg16 = tf.keras.models.load_model("checkpoint_VGG16.h5", 
                                       custom_objects={'GeMPooling2D': GeMPooling2D})

    
# Predicting the label 
predict_labels_vgg=[]
for i in tqdm.tqdm(range(8000)):
    pred_labels_1=get_label(np.array(vgg16.predict(test_generator[i][0])[0]))
    #pred_labels_2=np.array(model_mobile.predict(test_generator[i][0])[0])
    predict_labels_vgg.append(pred_labels_1)

true_labels=[int(i) for i in true_labels]
predict_labels_vgg=[int(i) for i in predict_labels_vgg]    
vgg_acc=accuracy_score(true_labels, predict_labels_vgg)
print('VGG16 Accuracy:',accuracy_score(true_labels, predict_labels_vgg))



######################### Ensemble model ####################################

predict_labels_ensemble=[]
for i in tqdm.tqdm(range(8000)):
    clear_output(wait=True)
    pred_labels_1=np.array(resnet200.predict(test_generator[i][0])[0])
    pred_labels_2=np.array(MobileNet200.predict(test_generator[i][0])[0])
    pred_labels_3=np.array(DenseNet200.predict(test_generator[i][0])[0])
    pred_labels_4=np.array(vgg16.predict(test_generator[i][0])[0])
    pred_labels=get_label(np.sum([pred_labels_1,pred_labels_2, pred_labels_3, pred_labels_4], axis=0))
    predict_labels_ensemble.append(pred_labels)
    
true_labels=[int(i) for i in true_labels]
predict_labels_ensemble=[int(i) for i in predict_labels_ensemble]   
ens_acc=accuracy_score(true_labels, predict_labels_ensemble) 
print(accuracy_score(true_labels, predict_labels_ensemble))

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


#def call_func():
 #   image_index=random.randint(0,7999)
def predictor():
    '''
    The predictor function takes no argument. It initially chooses a random number
    between 0 to 7999, the number of images in the test data. Based on the index number, 
    we will check whether the original label and the predicted label matches.
    Finally, the function returns the randomly chosen index number and the predicted label.
    '''
    
    image_index=random.randint(0,7999)
    print('__________________________________________________\n')
    print('The index number is:\n',image_index)
    print('__________________________________________________\n')
    #image_index = 7999 # choose a image (0-8000)
    image = test_generator[image_index][0] 
    image = image.reshape((128,128,3))
    plt.imshow(image)
    plt.show()
    print('__________________________________________________\n')
    #predict_labels_ensemble=[]
    pred_labels_1=np.array(resnet200.predict(test_generator[image_index][0])[0])
    pred_labels_2=np.array(MobileNet200.predict(test_generator[image_index][0])[0])
    pred_labels_3=np.array(DenseNet200.predict(test_generator[image_index][0])[0])
    pred_labels_4=np.array(vgg16.predict(test_generator[image_index][0])[0])
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
    for i in tqdm.tqdm(range(8000)):
        if get_label(test_generator[i][1][0]) == label[0]:
            image_list.append(i)
            
    if len(image_list) > 10:
        for i in (range(10)):
            if i != image_index:
                image = test_generator[image_list[i]][0]
                image = image.reshape((128,128,3))
                plt.imshow(image)
                plt.show()
            
    else:
        for i in range(len(image_list)):
            if i != image_index:
                image = test_generator[image_list[i]][0]
                image = image.reshape((128,128,3))
                plt.imshow(image)
                plt.show() 
 
label = predictor() 
display_related_images(label[1])



