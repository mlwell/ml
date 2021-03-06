from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import RMSprop

from qsdata import qsdata
import numpy as np

#env = qsdata(20)
#x_train, y_train, x_test, y_test = env.get_cat_data(10)
#x_train = np.expand_dims(x_train, axis=2)
#x_test = np.expand_dims(x_test, axis=2)

env = qsdata(1)
x_train, y_train, x_test, y_test = env.get_candlestick_data(10)


model = Sequential()

model.add(Dense(128, activation='relu', input_dim=4))
model.add(Dense(64, activation='relu'))


# model.add(Conv1D(32, 5, activation='relu', input_shape=(20, 1)))
# model.add(Conv1D(64, 2, activation='relu'))
# #model.add(MaxPooling1D(3))
# #model.add(Conv1D(128, 3, activation='relu'))
# #model.add(Conv1D(128, 3, activation='relu'))
# model.add(GlobalAveragePooling1D())
# #model.add(Dropout(0.5))
#
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))

model.add(Dense(2, activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(epsilon=1e-10, rho=0.9, lr=0.001),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=64)

print ("score", score)

