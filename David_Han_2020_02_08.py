
# Check sys configuration 
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras
#AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
print('Tensorflow: '+tf.__version__);print('OpenCV: '+cv2.__version__);print('Keras: '+keras.__version__);print('Python: '+sys.version)
print("Current version, Tensorflow = 2.0.0, OpenCV = 4.1.2, Keras = 2.3.1, Python = 3.6.9")

# Check the number of the jpegs from the specific url 
import pathlib
data_dir = pathlib.Path('C:/RACE/RiverBot/Garbage classification')
image_count = len(list(data_dir.glob('*/*.jpg')))
image_count

# Display names of subfolders under the main folders. Names of the subfolders would displayed the number of classes.
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
CLASS_NAMES

# list the first 3 images under plastic
plastic = list(data_dir.glob('plastic/*'))

for image_path in plastic[:6]:
    display.display(Image.open(str(image_path)))

# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 64
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))


# Display 5 x 5 images of a plot of images.
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)


# Check the time to run 1000 batches
import time 
default_timeit_steps = 1000

def timeit(ds, steps=default_timeit_steps):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
    batch = next(it)
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))


# `keras.preprocessing`
timeit(train_data_gen)
#Create subfolders consisting of test & train, where test consist of 25% and train set consists of 75% of images selected randomly.
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = 'C:/RACE/RiverBot/Garbage classification/'
subdirs = ['train/', 'test/']
labeldirs = ['no_plastic','plastic']
for subdir in subdirs:
    for labdir in labeldirs:
    # create label subdirectories
        newdir = dataset_home + subdir + labdir
        makedirs(newdir, exist_ok=True)
# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = 'C:/RACE/RiverBot/Garbage classification/'+str(labeldirs[0])
for file in listdir(src_directory):
    src = src_directory  + '/' + file
    dst_dir = 'train/'
    dst = dataset_home + dst_dir + str(labeldirs[0]) + '/' + file
    if random() < val_ratio:
        dst_dir = 'test/'
    #if file.startswith(str(labeldirs[0])):
        dst = dataset_home + dst_dir  + str(labeldirs[0]) + '/' + file
    copyfile(src, dst)
src_directory = 'C:/RACE/RiverBot/Garbage classification/'+str(labeldirs[1])
for file in listdir(src_directory):
    src = src_directory  + '/' + file
    dst_dir = 'train/'
    dst = dataset_home + dst_dir + str(labeldirs[1]) + '/' + file
    if random() < val_ratio:
        dst_dir = 'test/'
    #if file.startswith(str(labeldirs[1])):
        dst = dataset_home + dst_dir  + str(labeldirs[1]) + '/' + file
    copyfile(src, dst)

#Code for training the model.
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file in folder
	filename = 'C:/RACE/RiverBot/Garbage classification/' 
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 
# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterator
	train_it = datagen.flow_from_directory('C:/RACE/RiverBot/Garbage classification/train/',
		class_mode='binary', batch_size=32, target_size=(200, 200))
	test_it = datagen.flow_from_directory('C:/RACE/RiverBot/Garbage classification/test/',
		class_mode='binary', batch_size=32, target_size=(200, 200))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=100, verbose=0)
    # save model
    # Model is saved  in c:/Users/david
	model.save('final_model.h5')
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
 
# entry point, run the test harness
run_test_harness()
print('done')

# Print the inputs & outputs of the model.
from keras.models import load_model
model = load_model('c:/Users/david/final_model.h5')
print(model.outputs)
print(model.inputs)

# This code is needed in the riverbot including modules like keras, opencv, numpy
from keras.models import load_model 
from keras.preprocessing import image
import cv2
import numpy as np
#model = load_model('c:/Users/david/final_model.h5')
model = load_model('/home/pi/Desktop/Riverbot/final_model.h5')
model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
print("Ok")

# This code is needed in the riverbot
# compare the model with a test image in the same path. 1 is correct & o is incorrect. for single image
img = cv2.imread('C:/RACE/RiverBot/istrash/download (1).jpg')
img = cv2.resize(img,(200,200))
img = np.reshape(img,[1,200,200,3])
classes = model.predict_classes(img)
print (classes)

# This code is needed in the riverbot
# Compare the model with a test image in the same path. 1 is correct & o is incorrect. for multiple images
import glob
import pathlib
import numpy as np

istrash = pathlib.Path('/home/pi/Desktop/Riverbot/istrash')
display_istrash = list(istrash.glob('*.jpg'))
istrash_T = np.transpose(display_istrash)
print(istrash_T,end="")
print("")
cv_img = []

only_1 = 0
only_0 = 0
for img in glob.glob("/home/pi/Desktop/Riverbot/istrash/*.jpg"): #"C:/RACE/RiverBot/istrash/*.jpg"
    n= cv2.imread(img)
    n = cv2.resize(n,(200,200))
    n = np.reshape(n,[1,200,200,3])
    classes = model.predict_classes(n)
    print (int(classes),end = '')
    only_1 =  only_1 + int(classes)
    if int(classes) < 1:
        a = int(classes) + 1
        only_0 = only_0 + a
        cv_img.append(n)
print(" ")
print("No. of 1 found: "+str(only_1)+"/"+str(only_1+only_0))
print("No. of 1 found: "+str((only_1)/(only_1+only_0)))

# This code is needed in the riverbot
# compare the model with a test image in the same path. 1 is correct & o is incorrect. for multiple images
import glob
import pathlib
import numpy as np

istrash = pathlib.Path('C:/RACE/RiverBot/!istrash')
display_istrash = list(istrash.glob('*.jpg'))
istrash_T = np.transpose(display_istrash)
print(istrash_T,end="")
print("")
cv_img = []

only_1  = 0
only_0 = 0
for img in glob.glob("C:/RACE/RiverBot/!istrash/*.jpg"):
    n= cv2.imread(img)
    n = cv2.resize(n,(200,200))
    n = np.reshape(n,[1,200,200,3])
    classes = model.predict_classes(n)
    print (int(classes),end = '')
    only_1 =  only_1 + int(classes)
    if int(classes) < 1:
        a = int(classes) + 1
        only_0 = only_0 + a
        cv_img.append(n)
print(" ")
print("No. of 0 found: "+str(only_0)+"/"+str(only_1+only_0))
print("No. of 0 found: "+str((only_0)/(only_1+only_0)))

# Code used to create multiple images via translation from a folder when there are limited images
import glob
from random import seed
from random import random
import random
ctr = 0
for img in glob.glob("C:/RACE/RiverBot/Create_Image/*.jpg"):
    image = cv2.imread(img)
    rows,cols = image.shape[:2]
    radnumx  = random.randrange(50, 300)
    radnumy = random.randrange(50, 100)
    M = np.float32([[1,0,radnumx],[0,1,radnumy]]) 
    img_translation = cv2.warpAffine(image,M,(cols,rows))
    ctr += 1
    status = cv2.imwrite('C:/RACE/RiverBot/Edit_Image/'+str(ctr)+'.jpg',img_translation)
    print(ctr)
    print(radnumx,radnumy)
    
#Resources:
#https://www.tensorflow.org/tutorials/load_data/images#basic_methods_for_training
#https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/




