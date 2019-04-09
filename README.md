# Research Idea

## Project Description:
#### Goals and objectives
The aim of this project is to develop a system to help common people to better communicate with those individuals who can’t speak. The idea is to develop a mobile application by harnessing the power of computer vision and machine learning to translate hand gestures to human understandable text.

#### Problems to be addressed
The major problem which needs to be addressed include collecting data of hand gestures for training our system. This includes data taken from different geographic locations, during different times of the day including people from different age groups and in different ambient conditions.

#### Potential pitfalls & challenges
The system will be difficult to predict or translate gestures in very bright light conditions or in extremely dark environments.

#### Resources
In terms of resources, we will require an appreciable amount of cloud computing power. This is because of the fact that training a vast set of hand gestures images will require significant space and compute power. Also, the processing involved in the later end of the project will require real time processing and a cloud service will be of great use there.

#### People and Organizations Involved
For our project, we have taken help from the Disability Center and Center for American Sign Language for helping us in interpreting the sign language and for giving us valuable insights while deciphering the signs.
Developers: Nikunj Lad, Johail Sheriff


# Background Research
[American Sign Language](https://en.wikipedia.org/wiki/American_Sign_Language) recognition is not a recent computer vision problem. Over the past couple of decades, researchers have used different approaches to detect static hand gestures using linear classifiers, neural networks, Artificial Neural Network, Fuzzy Logic, Genetic Algorithm, Hidden Markov Model, Support Vector Machines etc. ASL focuses more on the hand gestures than the facial expressions. Often people use the base dataset of ASL that includes just the 26 english characters. Sign Language Recognition [1] has made a Flemish Sign Language recognition system using Convolutional Neural Networks with the use of dataset containing 50 different gestures and giving out an error of 2.5%. Unfortunately, this work is limited in the sense that it considers only a single person’s hand gesture. ASL fingerspelling [3] translator works based on a convolutional neural network. They make use of the pre-trained GoogLeNet architecture containing the A to K alphabets dataset. The above systems focus only on limited person’s gesture, also might not work well with the data taken in an unrestricted environment.

[Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) having been extremely successful in solving the real world problems with the help of image recognition and classification, and repeatedly implemented to recognize static hand gesture in recent years. In particular, there is work done in the recognition and classification of american sign language using deep CNNs, with input-recognition of the hand images to be converted to pixels of the image. The most relevant work to date is L. Pigou et al’s application [2] of CNN’s to classify 20 Italian Sign Language gestures on a considerably small amount of dataset. Previously, there was only basic camera technology used to generate datasets of images, which excluded depth or contour information while specifically considering pixel values. Attempts at using CNNs to handle the task of classifying images of ASL hand gestures have made some success [3], by using the pre-trained GoogLeNet architecture.

Neural networks have been used to tackle ASL translation in recent events. Neural networks provide a significant advantage to learn the important classification features of the dataset and they require considerably more time to train a larger dataset. The Intelligent Sign Language recognition has classified the video of ASL letters into text using advanced feature extraction [4] in the real time. The system extracts features in two categories: hand position and movement. Prior to ASL classification, with help of matlab to analyze the gesture in a graphical method by mapping it to the particular alphabet. While they claim to be able to correctly classify the images with an accuracy of 100%, there is no mention of whether this result was achieved during the training, validation or test set.

# Datasets
We explored numerous datasets of sign language recognition. We also approched the American Sign Language association at Northeastern University for help related to sign languages

1. [Rutgers University Sign Language Dataset](http://secrets.rutgers.edu/dai/queryPages/signSummary.php?signName=Search+all...&all_signs=all_signs_egcl&sign%5BLEX%5D=on&sign%5BIX%5D=on&sign%5BFS%5D=on&sign%5BNS%5D=on&sign%5BLS%5D=on&hand=dom&all_pos=on&pos%5B10%5D=on&pos%5B11%5D=on&pos%5B19%5D=on&pos%5B15%5D=on&pos%5B34%5D=on&pos%5B16%5D=on&pos%5B28%5D=on&pos%5B4%5D=on&pos%5B18%5D=on&pos%5B13%5D=on&pos%5B22%5D=on&pos%5B8%5D=on&pos%5B30%5D=on&pos%5B26%5D=on&pos%5B1%5D=on&pos%5B7%5D=on&pos%5B17%5D=on&pos%5B20%5D=on&pos%5B12%5D=on&pos%5B3%5D=on&pos%5B2%5D=on&pos%5B5%5D=on&pos%5B24%5D=on&pos%5B27%5D=on&pos%5B14%5D=on&pos%5B29%5D=on&pos%5B23%5D=on&pos%5B9%5D=on&pos%5B21%5D=on&pos%5B6%5D=on&pos%5B25%5D=on&minOccur=-1&data_source=&participant=&color=noCare&sleeves=noCare&glasses=noCare&allBox=noCare&results_view=front&resultsPerPage=25)
2. [Boston University Dataset](http://www.bu.edu/av/asllrp/dai-asllvd.html)

The dataset which we are using is [ASL Language Dataset](https://www.kaggle.com/grassknoted/asl-alphabet)

# Existing Methods used
According to our research, CovNets are used for detecting and classifying sign languages. In order to understand the performance and efficiency of this method, we decided to implement this with the sign language dataset. Here we simply input the images without any pre processing and image filtering to see how well the model performs.

Importing modules
```javascript
import os
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import random
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

```
Setting initial parameters
```javascript
# our data directory  where our sign language images are stored
DataDir = r"/home/nikunj/Northeastern/ml/ml_project/data/asl_alphabet_train/"
test_dir = r"/home/nikunj/Northeastern/ml/ml_project/data/asl_alphabet_test/"
# list of sub directory names
categories = ['A','B'] #'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

global img_array  # declaring global variable name for storing read image file

# image size is 50 * 50
Img_Size = 50

training_data = []
test_data_x = []
test_data_y = []
no_of_classes = 10
class_num = np.identity(no_of_classes)
```

&nbsp;&nbsp;![A](/images/A1.jpg) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![B](/images/B1.jpg)<br/>
**Figure 1** - Sign Representation A &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Figure 2** - Sign Representation B


Create Training and Testing data
```javascript
def create_training_data():
    for index, Category in enumerate(categories):
        path = os.path.join(DataDir, Category)  # path to cats or dogs Category
        class_num = categories.index(Category)  # 0 = Dog, 1 = Cat
        #selected_row = class_num[:, index]

        # for an img in
        for img in os.listdir(path):
            try:
                # reading the image in gray scale mode and resizing it
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (Img_Size, Img_Size))
                training_data.append([new_array, class_num])  # creating training data by appending list of new sized images and class it correponds to
            except Exception as e:
                pass

def create_testing_data():
    for index, Category in enumerate(categories):
        path = os.path.join(test_dir, Category)
        target_class = categories.index(Category)

        try:
            img_name = path + '_test.jpg'
            img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (Img_Size, Img_Size))
            test_data_x.append(new_array)
            test_data_y.append(target_class)
        except Exception as e:
            pass


create_training_data()   # function to create training data
create_testing_data()
```
Randomly shuffling and pickling data
```javascript
random.shuffle(training_data)    # randomly shuffling training data

# for every sample (image, class pair) present in the training data,  print the class labels
for sample in training_data[0:10]:
    print(sample[1])
# create variables to hold training data samples and target values
x = [] #feature
y = [] #label

# seperate out the training data into features and label
for features, label in training_data:
    x.append(features)
    y.append(label)

print(len(x))
print(len(y))
# reshaping the images
x = np.array(x).reshape(-1, Img_Size, Img_Size, 1)  # -1 how many shapes we have


pickle_out  = open("x.pickle", 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out  = open("y.pickle", 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("x.pickle", 'rb')
x = pickle.load(pickle_in)

# finding the tensorflow version
print(tf.__version__)

pickle_in = open("x.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
print(y)

X = X/255.0
```

Creating CNN Model using Keras
```javascript
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'mse', 'mae', 'cosine', 'mape'])
```

Model Evaluation
```javascript
history = model.fit(X, y, batch_size=30, epochs=4, validation_split=0.3)
print("\nTraining accuracy: ", history.history['acc'])
print("\nCross Validation accuracy: ", history.history['val_acc'])

print(history.history.keys())
plt.plot(history.history['mean_squared_error'])
plt.show()
plt.plot(history.history['mean_absolute_error'])
plt.show()
plt.plot(history.history['mean_absolute_percentage_error'])
plt.show()
plt.plot(history.history['cosine_proximity'])
plt.show()
plt.plot(history.history['cosine_proximity'])
plt.show()
plt.plot(history.history['val_mean_absolute_error'])
plt.show()
plt.plot(history.history['val_mean_squared_error'])
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

test_data_x = np.array(test_data_x).reshape(-1, Img_Size, Img_Size, 1)
ypred = model.predict(x=test_data_x, verbose=1)

hits = [i for i, j in zip(ypred, test_data_y) if i == j]
test_accuracy = len(hits) / len(test_data_y) * 100
print("\nTest data accuracy: ", test_accuracy)
```
Output using just CNN
```
4200/4200 [==============================] - 10s 2ms/step - loss: 0.5123 - acc: 0.7283 - mean_squared_error: 0.1735 - mean_absolute_error: 0.3732 - cosine_proximity: -0.4990 - mean_absolute_percentage_error: 186247017.6571 - val_loss: 0.3283 - val_acc: 0.8650 - val_mean_squared_error: 0.1037 - val_mean_absolute_error: 0.2335 - val_cosine_proximity: -0.5022 - val_mean_absolute_percentage_error: 74763885.0667
Epoch 2/4
4200/4200 [==============================] - 9s 2ms/step - loss: 0.2276 - acc: 0.9074 - mean_squared_error: 0.0693 - mean_absolute_error: 0.1652 - cosine_proximity: -0.4990 - mean_absolute_percentage_error: 80535795.6893 - val_loss: 0.1779 - val_acc: 0.9289 - val_mean_squared_error: 0.0534 - val_mean_absolute_error: 0.1205 - val_cosine_proximity: -0.5022 - val_mean_absolute_percentage_error: 84005310.9000
Epoch 3/4
4200/4200 [==============================] - 10s 2ms/step - loss: 0.1297 - acc: 0.9571 - mean_squared_error: 0.0358 - mean_absolute_error: 0.0945 - cosine_proximity: -0.4990 - mean_absolute_percentage_error: 47370335.2946 - val_loss: 0.1120 - val_acc: 0.9633 - val_mean_squared_error: 0.0315 - val_mean_absolute_error: 0.0778 - val_cosine_proximity: -0.5022 - val_mean_absolute_percentage_error: 53106005.4167
Epoch 4/4
4200/4200 [==============================] - 15s 4ms/step - loss: 0.0811 - acc: 0.9738 - mean_squared_error: 0.0212 - mean_absolute_error: 0.0605 - cosine_proximity: -0.4990 - mean_absolute_percentage_error: 30264694.1054 - val_loss: 0.0839 - val_acc: 0.9711 - val_mean_squared_error: 0.0232 - val_mean_absolute_error: 0.0556 - val_cosine_proximity: -0.5022 - val_mean_absolute_percentage_error: 44999270.4667

Training accuracy:  [0.7283333337732724, 0.9073809453419277, 0.9571428464991706, 0.9738095143011638]

Cross Validation accuracy:  [0.8649999926487605, 0.9288888802131017, 0.9633333226044972, 0.9711111048857372]
dict_keys(['val_loss', 'val_acc', 'val_mean_squared_error', 'val_mean_absolute_error', 'val_cosine_proximity', 'val_mean_absolute_percentage_error', 'loss', 'acc', 'mean_squared_error', 'mean_absolute_error', 'cosine_proximity', 'mean_absolute_percentage_error'])
2/2 [==============================] - 0s 40ms/step

Test data accuracy:  100.0
```
Performance Curves

![Train Val Acc](/images/cnntva.png)
**Figure 3** - Training-Validation Accuracy curve
![Train Val Loss](/images/cnntvl.png)
**Figure 4** - Training-Validation Loss curve

---
# What are we proposing and researching

## Filtering and Edge Detection

Filtering is a technique for modifying or enhancing an image. For example, you can filter an image to emphasize certain features or remove other features. Image processing operations implemented with filtering include smoothing, sharpening, and edge enhancement.

Edge detection is an image processing technique for finding the boundaries of objects within images. It works by detecting discontinuities in brightness between the object pixels and the background pixels. You can easily notice that in an edge, the pixel intensity changes in a notorious way. A good way to express changes is by using derivatives. A high change in gradient indicates a major change in the image.


## 1. Gabor Filters


Gabor filters have been adapted in edge detection and feature extraction. Gabor filters are special classes of bandpass filter and they allow only a certain band of sinusoidal planes with frequencies and orientation, and reject the others which is modulated by a Gaussian envelope. The filter responds well in the edges and texture changes as it distinguishes a particular feature via the filter and the filter has a good distinguishing value at the spatial location of each feature as it can modleded to find at any angle. If the Gabor filter is oriented at a particular direction and a strong response is given at the  locations of the target images that have structures in the same direction of the garbor filter. For example, if the target image is made up of edges in the diagonal direction, a Gabor filter set will give a strong response only if its direction matches with the direction of the edges of the target image. To start with, Gabor filters work pretty much the same way as the conventional filters used in detecting the edge pixels of the image. By masking the image we are getting more precise details of each pixel and each pixel is assigned a value called it as ‘weight’ stored in an array. This array is slid over every pixel of the image same as the convolution operation kernel. When a Gabor filter is applied to an image, it gives the highest response at edges and at points where texture changes completely with respective to the background.

![Gabor Mathematical Formula](/images/gabor-filter.jpg)
**Figure 5** - Gabor Mathematical Operation

Gabor filter is a linear filter whose impulse response is the multiplication of a harmonic function with a Gaussian function. As per convolution theorem, the convolution of Fourier Transformation (FT) of harmonic function and FT of Gaussian function is nothing but FT of a Gabor filter impulse response [ FT(Gabor) = FT(Harmonic) * FT(Gaussian) ]. The filter consists of a real and an imaginary component, which represent the orthogonal directions. The two components are used individually or in a complex form. 

The tuning frequency f, or tuning period P, or λ establish to what kind of sinus wave the filter will respond best. (f=1/P=1/λ ; f=1/P=1/λ or f=π/λ )depending on the specific implementation.


![Filter Examples](/images/OpenCV-edge-detection-on-blackandwhite-5.jpg )
**Figure 6** - Comparison of Laplacian, Sobel-x, Sobel-y with Original Image 

### Mathematical Concept of Gabor Filters

![Math Operation Gabor Filter](/images/Gabor_filter.jpg )
**Figure 7** - Math Operation For Gabor 


### Using CNN with Gabor Filters

The initial implementation will be as mentioned above, the only difference being 
Training and Testing code will now be generated using Gabor fitered images
```javascript
def create_training_data():
    for index, Category in enumerate(categories):
        path = os.path.join(DataDir, Category)  # path to cats or dogs Category
        class_num = categories.index(Category)  # 0 = Dog, 1 = Cat
        #selected_row = class_num[:, index]
        count = 0

        # for an img in
        for img in os.listdir(path):
            try:
                # reading the image in gray scale mode and resizing it
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (Img_Size, Img_Size))
                
                filt_real, filt_imag = gabor(new_array, frequency=0.9)

                if(count == 0):
                    plt.subplot(121), plt.imshow(new_array, cmap='gray'), plt.title('Original')
                    plt.xticks([]), plt.yticks([])
                    plt.subplot(122), plt.imshow(filt_real), plt.title('Garbor Filter with Real Response')
                    plt.xticks([]), plt.yticks([])
                    plt.show()

                training_data.append([filt_real, class_num])  # creating training data by appending list of new sized images and class it correponds to
                count+=1
            except Exception as e:
                pass

def create_testing_data():
    for index, Category in enumerate(categories):
        path = os.path.join(test_dir, Category)
        target_class = categories.index(Category)

        try:
            img_name = path + '_test.jpg'
            img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (Img_Size, Img_Size))
            filt_real, filt_imag = gabor(new_array, frequency=0.9)
            
            test_data_x.append(filt_real)
            test_data_y.append(target_class)
        except Exception as e:
            pass


create_training_data()   # function to create training data
create_testing_data()
```

### Observations and Evaluations

Using Gabor filter with frequency as 0.6
![0.6a](/images/Gabor%20filter/gabor0.6a.png)
**Figure 8** - Sign Representation A 
![0.6b](/images/Gabor%20filter/gabor0.6b.png)
**Figure 9** - Sign Representation B 

Performance with frequency as 0.6
![0.6a](/images/Gabor%20filter/gabor0.6tv.png)
**Figure 10** - Training-Validation Accuracy curve 
![0.6b](/images/Gabor%20filter/gabor0.6tvl.png)
**Figure 11** - Training-Validation Loss curve 

Using Gabor filter with frequency as 0.9
![0.9a](/images/Gabor%20filter/gabor0.9a.png)
**Figure 12** - Sign Representation A 
![0.9b](/images/Gabor%20filter/gabor0.9b.png)
**Figure 13** - Sign Representation B 

Performance with frequency as 0.9
![0.9a](/images/Gabor%20filter/gabor0.9tva.png)
**Figure 14** - Training-Validation Accuracy curve
![0.9b](/images/Gabor%20filter/gabor0.9tvl.png)
**Figure 15** - Training-Validation Loss curve

---
## 2. Sobel Filters
It is a gradient(Sobel - first order derivatives) filter that  uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives – one for horizontal changes, and one for vertical. We define A as the source image, and Gx and Gy are two images which at each point contain the vertical and horizontal derivative approximations respectively

The first thing we are going to do is find the gradient of the grayscale image, allowing us to find edge-like regions in the x and y direction. As we apply this mask on the image it gives prominent vertical edges. It simply works like as first order derivative and calculates the difference of pixel intensities in a edge region. As the center column is of zero so it does not include the original values of an image but rather it calculates the difference of right and left pixel values around that edge. Also the center values of G(x) in the first and third column is 2 and -2 respectively for vertical mask. Where as the horizontal map works with the center values of G(y) as 2 and -2 in the first and third column.This give more weightage to the pixel values around the edge region and increases the edge intensity giving it enhanced look when compared to the original image.

![Filter Examples](/images/Sobel%20filter/sobel_filter.jpg )

**Figure 16** - Sobel filter on the original Image

### Mathematical Concept of Sobel Filters
![Filter Examples](/images/Sobel%20filter/sobel_maths.jpg )
**Figure 17** - Sobel Mathematical operation


### Using CNN with Sobel Filters

```javascript

def create_training_data():
    for index, Category in enumerate(categories):
        path = os.path.join(DataDir, Category)  # path to cats or dogs Category
        class_num = categories.index(Category)  # 0 = Dog, 1 = Cat
        #selected_row = class_num[:, index]
        count = 0
        # for an img in
        for img in os.listdir(path):
            try:
                # reading the image in gray scale mode and resizing it
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (Img_Size, Img_Size))

                filt = skimage.filters.sobel(img)

                if(count == 0):
                    plt.subplot(121), plt.imshow(new_array, cmap='gray'), plt.title('Original')
                    plt.xticks([]), plt.yticks([])
                    plt.subplot(122), plt.imshow(filt_x), plt.title('Sobel filter with derivative of X-axis')
                    plt.xticks([]), plt.yticks([])
                    plt.show()

                # plt.subplot(121), plt.imshow(img_array, cmap='gray'), plt.title('Original')
                # plt.xticks([]), plt.yticks([])
                # plt.subplot(122), plt.imshow(filt), plt.title('Sobel filter with derivative of Y-axis')
                # plt.xticks([]), plt.yticks([])
                # plt.show()
                training_data.append([filt_x, class_num])  # creating training data by appending list of new sized images and class it correponds to
                count+=1
            except Exception as e:
                pass

def create_testing_data():
    for index, Category in enumerate(categories):
        path = os.path.join(test_dir, Category)
        target_class = categories.index(Category)

        try:
            img_name = path + '_test.jpg'
            img_array = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (Img_Size, Img_Size))
            filt = skimage.filters.sobel(img)

            test_data_x.append(filt)
            test_data_y.append(target_class)
        except Exception as e:
            pass


create_training_data()   # function to create training data
create_testing_data()
```

### Observations and Evaluations

Using Sobel filter with kernel size as 3x3
![3a](/images/Sobel%20filter/sobel3a.png)
**Figure 18** - Sign Representation A 
![3b](/images/Sobel%20filter/sobel3b.png)
**Figure 19** - Sign Representation B

Performance of Sobel Filter with kernel size as 3x3
![3a](/images/Sobel%20filter/sobel3tva.png)
**Figure 20** - Training-Validation Accuracy curve
![3b](/images/Sobel%20filter/sobel3tvl.png)
**Figure 21** - Training-Validation Loss curve

Using Sobel filter with kernel size as 5x5
![5a](/images/Sobel%20filter/sobel5a.png)
**Figure 22** - Sign Representation A 
![5b](/images/Sobel%20filter/sobel5b.png)
**Figure 23** - Sign Representation B

Performance of Sobel Filter with kernel size as 5x5
![5a](/images/Sobel%20filter/sobel5tva.png)
**Figure 24** - Training-Validation Accuracy curve
![5b](/images/Sobel%20filter/sobel5tvl.png)
**Figure 25** - Training-Validation Loss curve


## 3. Watershed Algorithm:
A watershed is an area of land that feeds all the water running under it and draining off of it into a body of water. It combines with other watersheds to form a network of rivers and streams that progressively drain into larger water areas. As for topography it  determines where and how water flows. As watershed is also used for image processing techniques where the topographic surface with high intensity denotes peaks and hills with low intensity denotes valleys. First you start filling every isolated valleys (local minima) with different colored water labels. As the water rises, depending on the peaks (gradients) nearby, water from different valleys, obviously with different colors will start to merge. To avoid that, you start to build barriers in the locations where water merges. You continue the work of filling water and building barriers until all the peaks are under water. Then the barriers you created gives you the segmentation result. This is the "philosophy" behind the watershed. Watershed algorithm is used in image processing which primarily focuses on segmentation of the object from the background.

Before using any image we first remove the noises in the data, when converting the image into grayscale we get to see the noises present in the image. Morphological opening serves a good role in removing the white noises from the grayscale image

```javascript

ret, thresh = cv2.threshold(gray, 12,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # Otsu’s binarization gives a approximate estimate of the hand
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
```
![WaterShed](/images/Water_algo_thres-0/A-morph_open.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![Watershed](/images/Water_algo_thres-0/B-morph_open.png)<br/>
**Figure 26** - Morphology Opening for A and B

Now we get to know for that the region near to the center of objects are foreground and the region far away from the object are background of the image. To extract the properties of the image that are sure to be object is present we use erosion technique to remove the boundary pixels. And whatever is remaining, we can be sure it is an object in the image. That would work if objects were not touching each other. If they are touching each other, another good option would be to find the distance transform from the center of the objects and apply a proper threshold to get the accurate result of each object even if it is in contact with another object. 

```javasrcipt

sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
```
![Watershed](/images/Water_algo_thres-0/A_sure_bg_and_fg.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![Watershed](/images/Water_algo_thres-0/B_sure_bg_and_fg.png)<br/>
**Figure 27** - Sure Background and Foreground for A and B

Next we decide to find the area which are not of the object but the background of the image. For this, we will dilate the result of the morphological opening to increase the object boundary near to the background. This way, we can make sure that whatever region is in the background is really a background as the boundary regions have been removed

The remaining regions are those which we aren't sure if it is an object or the background so we name it as unknown region. Watershed algorithm should able the find what the unknown object field is it a background or the object. These areas are normally around the boundaries of objects where foreground and background also called as edges or border. This can be obtained from subtracting sure_fg area from sure_bg area of the image resulting in the unknown area. Now we know for sure which regions are the objects and the background as well. So we create marker (an array of same size as that of original image and with int32 data type) and label the regions inside it. The regions we know for sure (whether foreground or background) are labelled with any positive integers, but different integers, and the area we don't know for sure are just left as zero. For this we use cv2.connectedComponents(). It labels background of the image with 0, then other objects are labelled with integers starting from 1.
```Javascript

unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
```
![Watershed](/images/Water_algo_thres-0/A-markers.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![Watershed](/images/Water_algo_thres-0/B-markers.png)<br/>
**Figure 28** - Watershed Markers for A and B


But we know that if background is marked with 0, watershed will consider it as unknown area. Instead, we will mark unknown region, defined by unknown, with 0. Now our marker is ready and it is time for the final step to apply watershed. The original image is marked with the outline of the markers we got from the watershed algorithm.
```Javascript

img_array[markers == -1] = [255, 0, 0]
```
[Watershed](/images/Water_algo_thres-0/A-image.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![Watershed](/images/Water_algo_thres-0/B-image.png)<br/>
**Figure 29** - Watershed Markers with Original Image for A and B

Performance of Watershed Algorithm with threshold of 0, 255
![PlotWater](/images/Water_algo_thres-0/watershed_model_acc.png)
**Figure 30** - Training-Validation Accuracy curve
![PlotWater](/images/Water_algo_thres-0/watershed_algo_model_loss.png)
**Figure 31** - Training-Validation Loss curve

Performance of Watershed Algorithm with threshold of 125, 255
![PlotWater](/images/Water_algo_thres_125/water_shed_thre_acc_125.png)
**Figure 32** - Training-Validation Accuracy curve
![PlotWater](/images/Water_algo_thres_125/water_shed_thre_loss_125.png)
**Figure 33** - Training-Validation Loss curve

Performance of Watershed Algorithm with threshold of 12, 255
![PlotWater](/images/Water_algo_thres_12/water_thres_acc_12.png)
**Figure 34** - Training-Validation Accuracy curve
![PlotWater](/images/Water_algo_thres_12/watershed_thres_loss_12.png)
**Figure 35** - Training-Validation Loss curve


## 4. Adaptive Histogram Equalization

Adaptive Histogram Equalization is a image processing technique that handles the contrast adjustment using the image's histogram. It differs from ordinary histogram equalization in respect to the adaptive method computes several histograms for each pixels, it does not increases the contrast for the pixels but uses them to redistribute the lightness values of the image. This method helps in improving the local contrast and therefore enhancing the definitions of edges in the image.

The first histogram equalization considers only the global contrast of the whole image, which in many cases it is not a good idea. As the image increases in contrast the pixe with moderate contrast would get more contrast might be possible to lose some of the information needed by the image. The reason behind this method is because the adaptive histogram is not just confined to a particular region but to whole image.

Adaptive histogram equalization comes into play when the image has more darker pixels than the brighter pixels but should also maintain the state of the pixels which are already bright enough. So first the image is divided into small blocks called “tiles” ( default size is 8*8). Then each of these blocks or tiles in the histogram equalized as usual where in a small area the histogram could be confined to a small region. We use contrast limiting to get all the noises amplified in the image which then is sent to histogram bin. If any of the pixel from the image falls in the histogram bin within the specified contrast limit, those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization of the pixels the bilinear interpolation is applied to remove artifacts from the tile borders giving you an exact information of how the image should be.

![Adaptive Histogram Equalization](/images/AdaptiveHistogram_Clp_5.0/A_Adaptive_Histogram-image.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![Adaptive Histogram Equalization](//images/AdaptiveHistogram_Clp_5.0/A_Adaptive_Histogram-image.png)<br/>
**Figure 36** - Adaptive Histogram Equalization for A and B

Performance of Adaptive Histogram Equalization with clip rate of 2.0
![PlotHisto](/images/AdaptiveHistogramClip_2.0/adapt_histo_2.0clp_acc.png)
**Figure 37** - Training-Validation Accuracy curve
![PlotHisto](/images/AdaptiveHistogramClip_2.0/adapt_histo_2.0clp_loss.png)
**Figure 38** - Training-Validation Loss curve of clip rate 2.0

Performance of Adaptive Histogram Equalization with clip rate of 5.0
![PlotHisto](/images/AdaptiveHistogram_Clp_5.0/adaptive_histo_5.0clp_acc.png)
**Figure 37** - Training-Validation Accuracy curve
![PlotHisto](/images/AdaptiveHistogram_Clp_5.0/adapt_histo_5.0clp_loss.png)
**Figure 38** - Training-Validation Loss curve of clip rate 2.0

Performance of Adaptive Histogram Equalization with clip rate of 15.0
![PlotHisto](/images/AdaptiveHistogramClp_15.0/adapt_histo_15.0Clp_acc.png)
**Figure 37** - Training-Validation Accuracy curve
![PlotHisto](/images/AdaptiveHistogramClp_15.0/adapt_histo_clp_15.0_loss.png)
**Figure 38** - Training-Validation Loss curve of clip rate 2.0


---
# References

1. [Roel Verschaeren, Gesture Recognition using the Microsoft Kinect, 2012](https://www.microsoft.com/en-us/research/publication/robust-part-based-hand-gesture-recognition-using-kinect-sensor-2/)
2. [L. Pigou et al, Sign Language Recognition Using Convolutional Neural Networks. European Conference on Computer Vision 6-12, September 2014](https://www.semanticscholar.org/paper/Sign-Language-Recognition-Using-Convolutional-Pigou-Dieleman/ee72744bea3c77a8f490d766d87517b1a450d44b)
3. [Garcia, Brandon and Viesca, Sigberto. Real-time American Sign Language Recognition with Convolutional Neural Networks. In Convolutional Neural Networks for Visual Recognition at Stanford University, 2016](http://cs231n.stanford.edu/reports/2016/pdfs/214_Report.pdf)
4. [Sawant Pramada, Deshpande Saylee, Nale Pranita, Nerkar Samiksha, Mrs.Archana S. Vaidya: Intelligent Sign Language Recognition Using Image Processing, IOSR Journal of Engineering (IOSRJEN), Vol. 3, Issue 2, February 2013](https://pdfs.semanticscholar.org/ae17/d08cf42fd3ee2c66cfc52b7fddeb5ea6e55a.pdf)

# Project scope

We intend to create a mobile application which can be used to detect sign languages in real time.

# Next Steps to be taken

Nextly we intend to stack different filters together so as to have a combined effect on the images before they are sent to the convolutional neural network. We intend to implement ensemble methods while simultaneously exploring watershed algorithm and other segmentation results.

# Licensing
MIT License

Copyright (c) 2018 Nikunj Lad & Johail Sherieff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
