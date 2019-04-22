#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

data = pd.read_csv('./data/sudoku.csv')
number_column = r'(\d)' * 81

X = data.quizzes.str.extract(number_column, expand=False)
X.iloc[0:] = X.iloc[0:].astype('int8')
X = X.values / 9

y = data.solutions.str.extract(number_column, expand=False)
y.iloc[0:] = y.iloc[0:].astype('int8')
y = y.values / 9

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(81, activation='relu', input_shape=(81,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(81, activation='relu')
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=128)
print()
model.evaluate(X_test, y_test)
