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

# Datasets
We explored numerous datasets of sign language recognition. We also approched the American Sign Language association at Northeastern University for help related to sign languages

1. https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/ This database is intended for experiments in 3D object reocgnition from shape.
2. The dataset we are building contain nearly 1000 images of each monument from a different angle, time of day, lighting conditions.
3. There will be total 50K images for training and 10K images will be for testing the training model.


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
