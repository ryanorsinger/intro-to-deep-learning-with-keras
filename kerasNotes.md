Unlike statistics, ML is better w/ tends to deal with large, complex datasets (such as a dataset of millions of images, each consisting of tens of thousands of pixels) for which classical statistical analysis such as Bayesian analysis would be impractical. 


So, to do machine learning, we need three things:
    - Input data points—For instance, if the task is speech recognition, these data points could be sound files of people speaking. If the task is image tagging, they could be pictures.
    - Examples of the expected output—In a speech-recognition task, these could be human-generated transcripts of sound files. In an image etask, expected outputs could be tags such as “dog,” “cat,” and so on.
    - A way to measure whether the algorithm is doing a good job—This is necessary in order to determine the distance between the algorithm’s current output and its expected output. The measurement is used as a feedback signal to adjust the way the algorithm works. This adjustment step is what we call learning.


Some tasks that may be difficult with one representation can become easy with another.

Machine-learning models are all about finding appropriate representations for their input data—transformations of the data that make it more amena- ble to the task at hand, such as a classification task.

Learning, in the context of machine learning, describes an automatic search process for better representations.

So that’s what machine learning is, technically: searching for useful representa- tions of some input data, within a predefined space of possibilities, using guidance from a feedback signal. 

Deep means sucessive layers of increasingly meaningful representations
Deep learning is a multistage way to learn data representations.


Where we've been
- Logistic regression (logreg) is probibilistic regression (Naive Bayes). It's the data scientist's first classification algorithm and very useful, even if it's not 'deep'.
- Kernel methods like the Support Vector Machine find decision boundaries, "maximizing the margin" means how correct your hyperplane is
- "Kernel trick" 
- SVM requires "feature engineering"
- Random forests
- Gradient boosting machine

applying gradient boosting to decision trees is "the best" for non-perceptual data.

convolutional neural nets are the goto for computer vision and good for perceptual tasks

deep learning completely automates humans doing manual feature engineering
gradient boosting good for structured data

In technical terms, this means you’ll need to be familiar with XGBoost and Keras.

