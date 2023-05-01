import tensorflow as tf
from tensorflow import keras
import keras_vggface 
from keras_vggface import VGGFace
# from tensorflow.keras.applications.vggface import preprocess_input
import mtcnn
import numpy as np
from keras.utils.data_utils import get_file
import keras_vggface . utils
import PIL
from PIL import Image
import os
from keras. layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
import os. path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas as pd
import collectdata as cod
# from keras.engine.topology import network

dir_for_train = 'train/'

def list_dir(path):
    # specify the folder path
    # folder_path = 'train2/'
    folder_path = path
    # get a list of all the directories in the folder
    directories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    return directories

def predict(img_path):

    # img = mpimg.imread('2.jpg')
    img = mpimg.imread(img_path)

    face_detector = mtcnn.MTCNN()
    face_roi = face_detector.detect_faces(img)

    no_people = len(face_roi)
    total_people = []

    for i in range(no_people):

        x1, y1, width, height = face_roi[i]["box"]
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]

        resized_face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)
        # resized_face = face.resize((224, 224))
        input_image = np.array(resized_face)
        input_image = np.expand_dims(input_image, axis=0)
        print(resized_face.shape)
        # input_image = preprocess_input(input_image)

        prob_model = keras.Sequential ([
        custom_vgg_model,
        tf.keras.layers.Softmax ()
        ])
        predictions = prob_model.predict(resized_face[None, ...])

        class_names = list_dir('train2/')

        # assume `predictions` is a numpy array containing the predicted class indices
        predicted_class_indices = np.argmax(predictions, axis=1)

        # create a dictionary mapping the integer class indices to their string labels
        class_index_to_label = {i: class_names[i] for i in range(len(class_names))}

        # map the predicted class indices to their string labels
        predicted_labels = [class_index_to_label[i] for i in predicted_class_indices]

        total_people.append(predicted_labels[0])        

    return total_people

# cod.get_dataset_ready()

class_list = list_dir(dir_for_train)
train_dataset = keras.utils.image_dataset_from_directory(dir_for_train,class_names=class_list,shuffle = True,batch_size = 8,image_size = (224,224))
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.2),
    ])

vggface_resnet_base = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3))


path = "train/" # specify the path here
folders = 0

for _, dirnames, _ in os.walk(path):
    folders += len(dirnames)

nb_class = folders # Number of new people + 1 for unknown/Inva lid
# Freeze the base model
vggface_resnet_base.trainable = False
last_layer = vggface_resnet_base.get_layer('avg_pool') .output
# Build up the new model
inputs = tf.keras.Input (shape= (224, 224, 3))
x = data_augmentation (inputs)
x = vggface_resnet_base(x)
x = Flatten(name= 'flatten')(x)
out =  Dense(nb_class, name='classifier')(x)

custom_vgg_model = keras.Model(inputs, out)
base_learning_rate = 0.0001
custom_vgg_model.compile (optimizer=tf. keras.optimizers .Adam(learning_rate=base_learning_rate),loss=keras.losses .SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
custom_vgg_model.fit(train_dataset,epochs = 20)
custom_vgg_model.save('saved_models/v_2_2_13_04_23.h5')

