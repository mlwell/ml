from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import RMSprop

from qsdata import qsdata
import numpy as np

# Load iris data
# Tensorflow iris test
import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                        )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)

(train_feature, train_label), (test_feature, test_label) = load_data()
from keras.utils import to_categorical
#train_label = to_categorical(train_label)
#test_label = to_categorical(test_label)

#train_label = np.eye(3)[train_label]
#test_label = np.eye(3)[test_label]


model = Sequential()

model.add(Dense(32, activation='relu', input_dim=4))
model.add(Dense(8, activation='relu'))

model.add(Dense(3, activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(epsilon=1e-10, rho=0.9, lr=0.001),
              metrics=['accuracy'])

model.fit(train_feature, train_label, batch_size=8, epochs=100)
score = model.evaluate(test_feature, test_label, batch_size=30)

print ("score", score)


