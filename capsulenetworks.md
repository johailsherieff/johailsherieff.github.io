## Capsule Networks

Capsule Network is made of capsules rather than neurons. A capsule is a small group of neurons that learns to detect a particular object within a given region of the image, and it outputs a vector whose length represents the estimated probability that the object is present in that region, and the pose parameters of objects are encoded from orientation of object. If the object is changed slightly then the capsule will output a vector of the same length, but oriented slightly differently.

![Face Neuron](/images/Images/face_neuron.jpg)
<br/><br/>

### Network performace on CIFAR10 dataset

![Capsule Accuracy CIFAR](/images/Images/Accuracy.png)
<br/><br/>
![Capsule Loss CIFAR](/images/Images/Loss.png)
<br/><br/>

### Network performance on COIL100 dataset
![Accuracy for Coid100:](/images/Images/accuracy_coil100.png)<br/>
&nbsp;&nbsp;
![Loss for Coil100:](/images/Images/loss_coil100.png)<br/>

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
