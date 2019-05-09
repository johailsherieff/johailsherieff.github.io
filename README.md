# Research Idea

## Project Description:

#### Goals and objectives
The aim of this research project is to explore Capsule Networks (CN) and compare its classification performance with respect to Convolutional Neural Networks (CNNs) by identifying 3D images of various objects which are having spatial variances. The idea is to exploit the disadvantage of CNNs i.e. it’s incapability to identify images aligned or positioned differently spatially. We intend to reject the Null hypothesis that CNNs perform great for image classification by proposing an alternate hypothesis of using Capsule Networks which can capture the positional and relative location of elements of an image and help us identify objects with higher accuracy. Lastly, we intend to come up with the scenarios and cases when it is ideal to use CNN and when it is a good choice to use Capsule Networks.

#### Problems to be addressed
The major problem which needs to be addressed includes collecting data of famous monuments around the world from different angles for training our system. This includes data taken from different geographic locations, during different times of the day and in different ambient conditions. Next challenge for us is setting up a cloud cluster for training our models. Based on our research it takes about 1.5 hours to train the IMAGENET dataset on GPU enabled system with i7 processor, 16GB of RAM @3.6 GHz clock speed.

#### Potential pitfalls & challenges
The system will be difficult to predict or translate 3D objects in very bright light conditions or in extremely dark environments. This is just an assumption and further conclusions can be derived after running the algorithms on the same.

#### Resources
In terms of resources, we will require an appreciable amount of cloud computing power. This is because of the fact that training a vast set of 3D images will require significant space and compute power. 

#### People and Organizations Involved
For our project, we have taken help from the students of the INFO 6210 class. The team of 11 students are involved in developing and researching on various methods of Maya 3D image scripting along with the database creation for the same. 

### Developers: 

Nikunj Lad, Johail Sherieff, Parag Bhingarkar

## Background Research

[Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) having been successful in solving the real world problems of image recognition and classification, and are repeatedly implemented to recognize objects in recent years. However, CNNs have a hard time classifying images if the relative position of elements in the image are changed spatially by adding a rotation factor or a revolutional view of the image. This is due to the fact that the CNNs involve a max-pooling layer which takes away a lot of important information away since we are reducing the image dimensions and keeping only locally important features. So while the images retain the information likes edges, colors, depth etc in different layers, it fails to capture the positional and relative locations of these features with respect to the entire image. This drawback is addressed by capsule networks. Capsule Networks will help in maintaining these positional variances while increasing the classification accuracy of the system.  


## Image Classification using CNNs

Image recognition using CNN is the ability of a machine learning model to identify objects, places, people, writing and actions in images. Computers can use computer vision technologies in combination with a camera and artificial intelligence software to achieve image recognition.

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance to different aspects/objects within the image and be able to differentiate one from the other. CNN is used as the default model to deal with images and handles images in differen ways however still it follows the general concept of Neural Networks whenever the neurons are made up of learnable weights and biases. Each neuron takes the image pixel as the input and performs a dot product operation so that each element of the same height/width is multiplied with the same weight and they are summed together. CNN works based on the hidden layers and the fully connected layers.

![CNN:](/images/Images/CNN.png)<br/>
&nbsp;&nbsp;

First of all, hidden layer in artificial neural networks a layer of neurons, whose output is connected to the inputs of other neurons and therefore is not visible as a network output. The hidden layers' job is to transform the inputs into something that the output layer can use and use can apply any function to the layers. The output layer transforms the hidden layer activations into whatever scale you wanted your output to be on.

Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset. This is a totally general purpose connection pattern and makes no assumptions about the features in the data. It's also very expensive in terms of memory (weights) and computation (connections).

```javascript
batch_size = 16
num_classes = 100
epochs = 50

# A common Conv2D model
input_image = Input(shape=(128,128, 3))
x = Conv2D(64, (5, 5), activation='relu')(input_image)
#x= MaxPooling2D((2,2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Dropout(0.2)(x)
#x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((3, 3))(x)
#x = Conv2D(256, (3, 3), activation='relu')(x)
x = Dropout(0.2)(x)
#x = MaxPooling2D((2, 2))(x)
#x = Conv2D(512, (3, 3), activation='relu')(x)
#x = Dense((128))(x)
x = Dense((512))(x)

```

The dataset for our project would be 3D images and the CNN architecture would look like the above code snippet where we are using the Relu Activation Layer, Average Pooling and a Dense Layer. There are many different layers but this what our CNN architecture looks like and we would not go in depth of each layer as we will explain the layers we have used for our project.

## Capsule Networks

Capsule Network is made of capsules rather than neurons. A capsule is a small group of neurons that learns to detect a particular object within a given region of the image, and it outputs a vector whose length represents the estimated probability that the object is present in that region, and the pose parameters of objects are encoded from orientation of object. If the object is changed slightly then the capsule will output a vector of the same length, but oriented slightly differently.

![Face Features](/images/Images/face_image.png)
<br/><br/>


### Network performace of ConvNets on Coil100 dataset

![Accuracy for COIL100](/images/Images/accuracy_coil.png)
<br/><br/>
![ConvNets for COIL100](/images/Images/conv_net_loss.png)
<br/><br/>

### Network performance on COIL100 dataset

![Accuracy for COIL100:](/images/Images/accuracy_coil100.png)
<br/><br/>
![Loss for COIL100:](/images/Images/loss_coil100.png)
<br/><br/>


```javascript

"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.

the output of final model is the lengths of 10 Capsule, whose dim=16.

the length of Capsule is the proba,
so the problem becomes a 10 two-classification problem.
"""

x = Reshape((-1, 128))(x)
capsule = Capsule(num_classes, 32, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = Model(inputs=input_image, outputs=output)

# we use a margin loss
#adam = K.optimizers.Adam(lr=0.001)
model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
model.summary()
```

## Contributions

We explored numerous datasets of sign language recognition. We also approched the American Sign Language association at Northeastern University for help related to sign languages

1. [This database is intended for experiments in 3D object reocgnition from shape.](http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)
2. The dataset we are building contain nearly 500-600 images (taken from x,y,z angles) of differenc real-world objects.
3. We have managed a group of 11 students to create database of 3D images with the help of Maya. [Link](https://sindhurakolli.github.io/DMDD_portfolio/)

## References

1. [Geoffrey E. Hinton, Sara Sabour, Nicholas Frosst; Dynamic Routing Between Capsules; 26, October 2017](https://arxiv.org/pdf/1710.09829v1.pdf)
2. [Capsule Networks: The New Deep Learning Network](https://towardsdatascience.com/capsule-networks-the-new-deep-learning-network-bd917e6818e8)
3. [GitHub Link for CapsNet Keras](https://github.com/XifengGuo/CapsNet-Keras)

## Result

![Result Table:](/images/Images/table.PNG)<br/>

## Conclusion

The research typically helped us understand the effect of Convolutional Neural Networks on images with positional variances and how such variances can be preserved using Capsule Networks. Convolutional Neural Networks works well on large datasets and takes less computational time whereas Capsule Networks although requiring less data are computationally expensive. 

## Project scope

The project can be extended over the summer term to modify and experiment over the architecture of capsule networks for the database created by the students as most of the features are not evenly distributed among the image. Preprocessing the image before giving it to the CNN and Capsule will prove better result to the dataset and experimenting on the dataset will require time.

## License
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
