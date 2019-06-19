# Batch Size  

Neural networks are trained using gradient descent where the estimate of the error used to update the weights is calculated based on a subset of the training dataset.

The number of examples from the training dataset used in the estimate of the error gradient is called the batch size and is an important hyperparameter that influences the dynamics of the learning algorithm.

The choice of batch size controls how quickly the algorithm learns, for example:
- **Batch Gradient Descent.** Batch size is set to the number of examples in the training dataset, more accurate estimate of error but longer time between weight updates.
- **Stochastic Gradient Descent.** Batch size is set to 1, noisy estimate of error but frequent updates to weights.
- **Minibatch Gradient Descent.** Batch size is set to a value more than 1 and less than the number of training examples, trade-off between batch and stochastic gradient descent.  

Keras allows you to configure the batch size via the batch_size argument to the fit() function, for example:

```python
# fit model
history = model.fit(trainX, trainy, epochs=1000, batch_size=len(trainX))
```
The example below demonstrates a Multilayer Perceptron with batch gradient descent on a binary classification problem.  

```python
# example of batch gradient descent
from sklearn.datasets import make_circles
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
# generate dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, batch_size=len(trainX), verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss learning curves
pyplot.subplot(211)
pyplot.title('Cross-Entropy Loss', pad=-40)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy learning curves
pyplot.subplot(212)
pyplot.title('Accuracy', pad=-40)
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()​
```

Your Task  
For this lesson, you must run the code example with each type of gradient descent (batch, minibatch, and stochastic) and describe the effect that it has on the learning curves during training.
  
Info via: jason@MachineLearningMastery.com 
