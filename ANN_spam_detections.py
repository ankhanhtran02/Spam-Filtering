# Import the libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
# File "Spam.csv" contains some invalid characters
# An error may be thrown, so we need to encode it
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# After encoding, there are some redundant columns
# Drop unnecessary columns
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# rename columns
df.columns = ['labels', 'data']

# create binary labels
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
y = df['b_labels'].values

# try multiple ways of calculating features
# decode_error: ignore any invalid UTF character
tfidf = TfidfVectorizer(decode_error='ignore')
X = tfidf.fit_transform(df['data'])

# count_vectorizer = CountVectorizer(decode_error='ignore')
# X = count_vectorizer.fit_transform(df['data'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Define the Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train.todense(), y_train, epochs=10)

train_loss, train_acc = model.evaluate(X_train.todense(), y_train, verbose=2)
test_loss, test_acc = model.evaluate(X_test.todense(),  y_test, verbose=2)

print('\nTrain accuracy:', train_acc)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# Test the result
test_string = input()

test_string = tfidf.transform([test_string])

predictions = probability_model.predict(test_string.todense())

print(np.argmax(predictions[0]))
