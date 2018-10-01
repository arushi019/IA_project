import os
import skimage.data
import sys
from PIL import Image
from statistics import mean
import numpy
from sklearn import preprocessing
import tensorflow as tf
#from PIL.Image import fromArray,shape
def traffic_sign_dataset(paths):
    folders=[di for di in os.listdir(paths) if os.path.isdir(os.path.join(paths,di))]
    image=[]
    label=[]
    for f in folders:
        f_label=os.path.join(paths,f)
        files=[os.path.join(f_label,temp) for temp in os.listdir(f_label) if temp.endswith(".ppm")]
        for temp in files:
            image.append(skimage.data.imread(temp))
            label.append(int(f))
    return image,label
def traffic_sign_resize_images(images):
    resized_image=[]
    for img in images:
        img_object=Image.fromarray(img).convert('LA')
        resized_image.append(img_object.resize((32,32),Image.ANTIALIAS))
    #preprocessing.normalize(resized_image)
    return resized_image
def traffic_sign_normalise(images):
    normalised_image=[]
    for img in images:
        temp=numpy.array(img,dtype=numpy.float)
        normalised_image.append(temp)
    preprocessing.normalize(normalised_image)
    for i in range(len(normalised_image)):
        images[i]=Image.fromArray(normalised_image[i])
    return images
def traffic_sign_mean_image(images):
    mean_image=numpy.zeros((32,32,3),numpy.float)
    for temp in images:
        temp_arr=numpy.array(temp,dtype=numpy.float)
        mean_image=mean_image+temp_arr/len(images)
    mean_image=numpy.array(numpy.round(mean_image),dtype=numpy.uint8)
    for i in range(len(images)):
        images[i]=images[i]-mean_image
    return images
def traffic_sign_model_training(images,labels,test_images,test_labels):
    image=numpy.array(images)
    label=numpy.array(labels)
    test_image=numpy.array(test_images)
    test_label=numpy.array(test_labels)
    print("labels: ",label.shape,"images: ",image.shape)
    model=tf.Graph()
    with model.as_default():
        p_image=tf.placeholder(tf.float32,[None,32,32,3])
        p_label=tf.placeholder(tf.int32,[None])
        flattened_images=tf.contrib.layers.flatten(p_image)
        bits=tf.contrib.layers.fully_connected(flattened_images,62,tf.nn.relu)
        outcome=tf.argmax(bits,1)
        error=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=bits,labels=p_label))
        model_train=tf.train.AdamOptimizer(learning_rate=0.001).minimize(error)
        init=tf.global_variables_initializer()
    session=tf.Session(graph=model)
    _=session.run([init])
    for i in range(201):
        _,error_val=session.run([model_train,error],feed_dict={p_image:image,p_label:label})
        if i%10==0:
            print("Error: ",error_val) 
    outcome_l=session.run([outcome],feed_dict={p_image:test_image})[0]
    ct=0
    for i in range(len(test_label)):
        if outcome_l[i]==test_label[i]:
            ct+=1
    acc=ct/len(test_label)
    print(acc) 
    session.close()
traffic_sign_image,traffic_sign_label=traffic_sign_dataset("/home/iiitd/Desktop/IA_project/BelgiumTSC_Training/Training")
#print(traffic_sign_image)
traffic_sign_image=[skimage.transform.resize(temp,(32,32),mode='constant') for temp in traffic_sign_image]
#resized_image=traffic_sign_resize_images(traffic_sign_image)
normalised_image=traffic_sign_mean_image(traffic_sign_image)
#traffic_sign_model_training(normalised_image,traffic_sign_label)
#print(mean_image)
test_image,test_label=traffic_sign_dataset("/home/iiitd/Desktop/IA_project/BelgiumTSC_Testing/Testing")
test_image=[skimage.transform.resize(img,(32,32),mode='constant') for img in test_image]
normalised_image_test=traffic_sign_mean_image(test_image)
traffic_sign_model_training(normalised_image,traffic_sign_label,normalised_image_test,test_label)