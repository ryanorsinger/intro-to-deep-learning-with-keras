# ⏳ Intro to Deep Learning with Keras

## Summary

1. Today you will get an intro to deep learning and run a neural network with Keras 📊 
1. You will see that getting started is *accessible* and you don't have to know everything 💭
1. We will import a data set, explore the shape of the data, and create a deep learning model


## Context
- In classical programming, answers are the product of input data and the explicit rules that programmers manually program.

- In machine learning, input data and example answers yield the rules we can use on future input data

- The output is a function of the inputs. Deep learning attempts to approximate that applied function 

  <img src="https://dpzbhybb2pdcj.cloudfront.net/chollet/Figures/01fig02.jpg" height=150 >

  

### What is Deep Learning? 
1. Applied linear algebra and calculus w/ the latest in hardware. Yay, math!
2. The "deep" refers to the number of layers of data transformation used to sift for meaningful representations
3. Deep learning is the application of many different representations of data to identify meaningful relationships in data.
4. The key component of deep learning is comparing predictions vs. true targets, then sending that data through an optimizer that updates weights and runs the inputs with the updated weights. 





## Anatomy of a Deep Learning Network

<img src="https://dpzbhybb2pdcj.cloudfront.net/chollet/Figures/01fig09.jpg" height="300px">


- In deep learning, we perform many data representations and transformations to compare predictions vs. true targets. The accuracy of the prediction is fed into an optimizer, and the optimizer updates the weights on different layers or features. Then, we run the data through the same layers with updated weights.
- This is the "learning" process: the feeding back of the result of predictions (backpropogation) through an optimization function that updates weights on layers that run on the data again, making for more effective predictions.
- Each layer is the application of many geometric transformations of the tensor data

### Identify meaningful representations of the data

This figure shows the original data and a new representation. We obtain new representations of the data by changing our axes to fit the story the data tells. Here we get a new representation of the data in such a way that we can describe data with a simple rule:

> “Black points are such that x > 0,” or “White points are such that x < 0.”

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://dpzbhybb2pdcj.cloudfront.net/chollet/Figures/01fig04.jpg" >



### What Deep Learning is *not*
- Cognition or thinking.
- Inaccessible mathematics
- Beyond your understanding
- Free from bias and prejudice. (Biased inputs mean biased outputs) 

### Vocabulary
- Tensor: a basic data structure in machine learning, a container for data can hold n dimensional matrices, usually numerical data

- Scalar: A tensor that holds only one number  (a zero dimensional tensor)

- Vector: A one dimensional array of numbers (a one dimensional tensor)

- Matrices: An array of vectors. (a two dimensional tensor)

    

### Real World Tensor Data
- Vector data —2D tensors (samples, **features**)
- Timeseries data or sequence data are 3D tensors of shape (samples, timesteps,
**features**)
- Images—4D tensors of shape(samples,height,width,channels)or(samples,
channels, height, width)
- Video —5D tensors of shape (samples, frames, height, width, channels) or
(samples, frames, channels, height, width)


## Show me the code!
1. Install pre-requisites
    - Python 2 or 3, the pip package manager (`brew install pip` for Macs)
    - `pip install tensorflow`
    - `pip install keras`
    - `pip install matplotlib`

#### Import in our demo data

    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()





### Let's explore the data to understand its shape.

    len(train_labels)
    train_images.shape
    train_images[0].shape
    train_images[0][0]
    
    test_images.shape
    test_images[0].shape

#### Data for people (thousands of 28px by 28px greyscale images)

We'll use `matplotlib` to visualize the training and test data. Grab a digit and we can see a good representation for people.

```
# grab any train or test image
digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
```

#### Get the data ready for the model

The deep learning model will need the data represented as a tensor of values between 0 and 1, so we need to do the following.

```
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
```

Review the data's shape`train_images.shape`, `train_images[0]` again

Import the keras models and create the network

    from keras import models
    from keras import layers
    
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    from keras.utils import to_categorical
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

Check loss score, send data through optimizer

    model.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    print('test_loss:', test_loss)



### Resources for continued learning

#### Video Recommendations

- MIT's Linear Algebra `http://web.mit.edu/18.06/www/videos.shtml`
- If you're a Bexar County resident, get your library card from `https://bexarbibliotech.org/` to get free access to `lynda.com`, a massive library of high quality video courses for professionals.
- Recommend `https://www.lynda.com/Google-TensorFlow-tutorials/Building-Deep-Learning-Applications-Keras-2-0/601801-2.html`

#### Selected readings

*Deep Learning With Python* by Francois Chollet (first 3 chapters are free) `https://livebook.manning.com/#!/book/deep-learning-with-python/about-this-book/`
