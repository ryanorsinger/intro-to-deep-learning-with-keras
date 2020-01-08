import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Model


# Specify the number of digits. We're using base 10, so this is 10.
NUM_DIGITS = 10

# Binary encode the inputs
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


## Setup the training data for 101-1024,
raw_training_data = np.array(range(101, 2 ** NUM_DIGITS))

x_train = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])



# One hot encoding of the training data
def fizz_buzz_encode(i):

    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

y_train = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])


# model
model = Sequential()
model.add(Dense(1000, input_dim=10, activation="relu"))
model.add(Dense(1000, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=["accuracy"])
model.fit(x_train, y_train, nb_epoch=100, batch_size=128)


# fizzbuzz to binary
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


# y_train fizzbuzz for prime numbers from 1 to 100
numbers = np.arange(1, 101)
x_test = np.transpose(binary_encode(numbers, NUM_DIGITS))
y_test = model.predict_classes(x_test)
output = np.vectorize(fizz_buzz)(numbers, y_test)
print (output)


# answer
answer = np.array([])
for i in numbers:
    if i % 15 == 0: answer = np.append(answer, "fizzbuzz")
    elif i % 5 == 0: answer = np.append(answer, "buzz")
    elif i % 3 == 0: answer = np.append(answer, "fizz")
    else: answer = np.append(answer, str(i))
print (answer)


# evaluate
evaluate = np.array(answer == output)
print (np.count_nonzero(evaluate == True) / 100)

