"""
trains a multi-class text classifier from a CSV
CSV data should be like: "text to train", "Label1;Label2"
"""

import csv
import re

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import set_maxsize


def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


set_maxsize.set_csv_max_length()
STOPWORDS = set(stopwords.words('english'))

vocab_size = 10000
embedding_dim = 256
max_length = 5000
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

articles = []
labels = []

# Set memory limit for your GPU if necessary
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)]
# )

with open("data.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    pattern = re.compile('[\W_]+')
    for row in reader:
        article_labels = row[1].split(';')
        for label in article_labels:
            label = pattern.sub(' ', label)
            label = label.replace(' ', '')
            labels.append(label)

        article = row[0]

        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')

        for _ in article_labels:
            articles.append(article)

# optional data visualization
# y = np.array(labels)
# plt.hist(y)
# plt.show()

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token=oov_tok,
)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.summary()

model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

num_epochs = 100
history = model.fit(
    train_padded,
    training_label_seq,
    batch_size=16,
    epochs=num_epochs,
    validation_data=(validation_padded, validation_label_seq),
    verbose=2
)


def plot_graphs(plot_history, string):
    plt.plot(plot_history.history[string])
    plt.plot(plot_history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
