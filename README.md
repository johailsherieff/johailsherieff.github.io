# Research Idea

## Project Description:
#### Goals and objectives
The aim of this research project is to explore Capsule Networks (CN) and compare its classification performance with respect to Convolutional Neural Networks (CNNs) by identifying famous monuments from around the world whose images are taken from any orientation. The idea is to exploit the disadvantage of CNNs i.e. it’s incapability to identify images aligned or positioned differently spatially. We intend to reject the Null hypothesis that CNNs perform great for image classification by proposing an alternate hypothesis of using Capsule Networks which can capture the positional and relative location of elements of an image and help us identify objects with higher accuracy.

#### Problems to be addressed
The major problem which needs to be addressed includes collecting data of famous monuments around the world from different angles for training our system. This includes data taken from different geographic locations, during different times of the day and in different ambient conditions. Next challenge for us is setting up a cloud cluster for training our models. Based on our research it takes about 1.5 hours to train the IMAGENET dataset on GPU enabled system with i7 processor, 16GB of RAM @3.6 GHz clock speed.

#### Potential pitfalls & challenges
The system will be difficult to predict or translate monuments in very bright light conditions or in extremely dark environments. 

#### Resources
In terms of resources, we will require an appreciable amount of cloud computing power. This is because of the fact that training a vast set of hand gestures images will require significant space and compute power. Also, the processing involved in the later end of the project will require real time processing and a cloud service will be of great use there.

#### People and Organizations Involved
For our project, we have taken help from the Disability Center and Center for American Sign Language for helping us in interpreting the sign language and for giving us valuable insights while deciphering the signs.
Developers: Nikunj Lad, Johail Sheriff


# Background Research
[Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) having been successful in solving the real world problems of image recognition and classification, and are repeatedly implemented to recognize objects in recent years. However, CNNs have a hard time classifying images if the relative position of elements in the image are changed spatially by adding a rotation factor or a revolutional view of the image. This is due to the fact that the CNNs involve a max-pooling layer which takes away a lot of important information away since we are reducing the image dimensions and keeping only locally important features. So while the images retain the information likes edges, colors, depth etc in different layers, it fails to capture the positional and relative locations of these features with respect to the entire image. This drawback is addressed by capsule networks. Capsule Networks will help in maintaining these positional variances while increasing the classification accuracy of the system.  


## Image Classification using CNNs

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance to various aspects/objects in the image and be able to differentiate one from the other. 

![CNN:](/images/Images/CNN.png)<br/>
&nbsp;&nbsp;

```javascript
# A common Conv2D model
input_image = Input(shape=(None, None, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_image)
x = Conv2D(64, (3, 3), activation='relu')(x)
#x = Conv2D(128, (3, 3), activation='relu')(x)
x = AveragePooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
#x = Conv2D(512, (3, 3), activation='relu')(x)
x = Dense((512))(x)
```

![CNN Accuracy](/images/Images/cnn_accuracy.png)
<br/><br/>
![CNN Loss](/images/Images/cnn_loss.png)
<br/><br/>

Image recognition, is the ability of a machine learning model to identify objects, places, people, writing and actions in images. Computers can use computer vision technologies in combination with a camera and artificial intelligence software to achieve image recognition


## Capsule Networks

![Face Neuron](/images/Images/face_neuron.jpg)
<br/><br/>
![Inverse_graphics](/images/Images/inverse_graphics.png)
<br/><br/>
![Equivariance](/images/Images/equivariance.png)
<br/><br/>
![Capsules](/images/Images/capsules.png)
<br/><br/>
![Dynamic Routing](/images/Images/routing.png)
<br/><br/>
### Network performace on CIFAR10 dataset
![Capsule Accuracy](/images/Images/capsule_accuracy.png)
<br/><br/>
![Capsule Loss](/images/Images/capsule_loss.png)
<br/><br/>
### Network performance on COIL100 dataset
![Accuracy for CIFAR:](/images/Images/accuracy_coil100)<br/>
&nbsp;&nbsp;
![Loss for CIFAR:](/images/Images/loss_coil100.png)<br/>

```javascript

"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.
the output of final model is the lengths of 10 Capsule, whose dim=16.
the length of Capsule is the proba,
so the problem becomes a 10 two-classification problem.
"""

x = Reshape((-1, 128))(x)
capsule = Capsule(10, 32, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = Model(inputs=input_image, outputs=output)

# we use a margin loss
#adam = K.optimizers.Adam(lr=0.001)
model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
model.summary()
```

## Google Cloud Platform

![Google Architecture:](/images/Images/Cloud_computing.png)<br/>
&nbsp;&nbsp;
![GCP Benefits:](/images/Images/GCP_benifits.png)<br/>
&nbsp;&nbsp;
![ML Architecture:](/images/Images/ml_architecture.png)<br/>


```javascript
Cloud SQL Instance for Database(SQL Standard):
M/C type : db-n1-standard-4
RAM(GB) : 15
Max Storage Capacity: 50 GB
Max Connections: 4000

We will require Cloud type as : My SQL and Second Generation.

Cloud Storage as a Common database as of now are using our google drive to store and retrieve the data.

Compute Engine Instance for the Capsule Network project:
M/C type : n1-standard-8
Virtual CPU: 8
Memory: 50 GB
```




Cloud computing is the delivery of computing services for servers, storage, databases, networking, software, analytics, artificial intelligence and moreover deployed in the Internet (“the cloud”) to offer faster innovation and flexible resources.

# Datasets
We explored numerous datasets of sign language recognition. We also approched the American Sign Language association at Northeastern University for help related to sign languages

1. https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/ This database is intended for experiments in 3D object reocgnition from shape.
2. The dataset we are building contain nearly 1000 images of each monument from a different angle, time of day, lighting conditions.
3. There will be total 50K images for training and 10K images will be for testing the training model.


# References

1. [Geoffrey E. Hinton, Sara Sabour, Nicholas Frosst; Dynamic Routing Between Capsules; 26, October 2017](https://arxiv.org/pdf/1710.09829v1.pdf)
2. [Capsule Networks: The New Deep Learning Network](https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8)
3. [GitHub Link for CapsNet Keras](https://github.com/XifengGuo/CapsNet-Keras)

# Project scope

We intend to create a mobile application which can be used to detect sign languages in real time.

# Next Steps to be taken

Nextly we intend to stack different filters together so as to have a combined effect on the images before they are sent to the convolutional neural network. We intend to implement ensemble methods while simultaneously exploring watershed algorithm and other segmentation results.

# License
MIT License

Copyright (c) 2019 Johail Sherieff, Nikunj Lad and Parag Bhingarkar

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
