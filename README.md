# ⏳ Intro to Deep Learning with Keras

## Summary

1. Today you will get an intro to deep learning and run a neural network with Keras 📊 
1. You will see that getting started is *accessible* and you don't have to know everything 💭


## Context
- In classical programming, answers are the product of input data and the explicit rules that programmers program.
- In machine learning, input data and example answers yield the rules we can use on future input data.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://dpzbhybb2pdcj.cloudfront.net/chollet/Figures/01fig02.jpg" height=150 >
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://dpzbhybb2pdcj.cloudfront.net/chollet/Figures/01fig01.jpg" height=150 >

### Identify meaningful representations of the data

This figure shows the original data and a new representation. We obtain new representations of the data by changing our axes to fit the story the data tells. Here we get a new representation of the data in such a way that we can describe data with a simple rule:
<br>
> “Black points are such that x > 0,” or “White points are such that x < 0.”

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://dpzbhybb2pdcj.cloudfront.net/chollet/Figures/01fig04.jpg" >


## What is Deep Learning? 
1. Applied linear algebra w/ the latest in hardware. It's just math.
2. The "deep" refers to the number of layers of data transformation used to sift for meaningful representations
3. Deep learning is the application of many different representations of data to identify meaningful relationships in data.
3. The key component of deep learning is comparing predictions vs. true targets, then sending that data through an optimizer that updates weights and runs the inputs with the updated weights. 
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://dpzbhybb2pdcj.cloudfront.net/chollet/Figures/01fig09.jpg">


- In deep learning, we perform many data representations and transformations to compare predictions vs. true targets. The accuracy of the prediction is fed into an optimizer, and the optimizer updates the weights on different layers or features. Then, we run the data through the same layers with updated weights.
- This is the "learning" process: the feeding back of the result of predictions (backpropogation) through an optimization function that updates weights on layers that run on the data again, making for more effective predictions.
- Each layer is the application of geometric transformations of the tensor data

### What Deep Learning is *not*
- How brains function in living things
- Inaccessible mathematics or beyond your understanding
- Free from bias and prejudice. (Biased inputs mean biased outputs)

## What is Deep Learning good for?
- Categorization tasks
- Regressions
- Clustering 
- Dimensionality reduction

### Vocabulary
- Tensor: a basic data structure in machine learning, a container for data can hold n dimensional matrices, usually numerical data
- Scalar: A tensor that holds only one number  (a zero dimensional tensor)
- Vector: A one dimensional array of numbers (a one dimensional tensor)
- Matrices: An array of vectors. (a two dimensional tensor)
- Tensors are defined by 3 key attributes:
    - Number of axes, rank, or .ndim property. A 3D tensor has 3 axes and a matrix has 2 axes. 
    - Shape
    - Data type (usually dtype)

### Real World Tensor Data
- Vector data —2D tensors of shape (samples, features)
- Timeseries data or sequence data are 3D tensors of shape (samples, timesteps,
features)
- Images—4D tensors of shape(samples,height,width,channels)or(samples,
channels, height, width)
- Video —5D tensors of shape (samples, frames, height, width, channels) or
(samples, frames, channels, height, width)


## Show me the code!
1. Install pre-requisites
    - Python 2 or 3, the pip package manager (`brew install pip` for Macs)
    - `pip install tensorflow`
    - `pip install keras`

Pull in our training data and test data
    
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

Explore the data a bit to understand its shape

    len(train_labels)
    train_images.shape
    train_images[0].shape

    train_labels

    test_images.shape
    len(test_labels)
    test_labels

Import the keras models and create the network
    
    from keras import models
    from keras import layers

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))


    network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

Take a moment to reexplore the shape and dimension of the test_images

    from keras.utils import to_categorical
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

Check loss score, send data through optimizer

    network.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    print('test_loss:', test_loss)
