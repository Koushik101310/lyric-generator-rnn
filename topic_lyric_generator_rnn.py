import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import pandas as pd
import zipfile
import urllib.request
import os
import re
import pickle
import json

MAX_SONGS    = 500
EPOCHS       = 50


url       = "https://github.com/datasigntist/datasetsForTraining/raw/master/songlyrics.zip"
local_zip = "/tmp/songlyrics.zip"

if not os.path.exists(local_zip):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, local_zip)

with zipfile.ZipFile(local_zip, "r") as zip_ref:
    zip_ref.extractall("/tmp")

print("Dataset downloaded and extracted.")


songdata = pd.read_csv("/tmp/songdata.csv")

print(f"Total overall songs: {len(songdata)}")

topic_songs = songdata.head(MAX_SONGS)
print(f"Using {len(topic_songs)} songs for training.")


def clean_line(line):
    line = line.lower()
    line = re.sub(r"\[.*?\]", "", line)
    line = re.sub(r"[^a-z\s']", "", line)
    line = re.sub(r" +", " ", line).strip()
    return line

raw_lines = "\n".join(topic_songs["text"]).split("\n")
corpus    = [clean_line(line) for line in raw_lines]
corpus    = [line for line in corpus if line]

print(f"Total number of lines: {len(corpus)}")


tokenizer = Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(f"Total unique words: {total_words}")

reverseWordIndex = {v: k for k, v in tokenizer.word_index.items()}


input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[: i + 1])

print(f"Total sequences: {len(input_sequences)}")


max_len_sequence = max(len(x) for x in input_sequences)
print(f"Maximum sequence length: {max_len_sequence}")

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_len_sequence, padding="pre")
)


xs     = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys     = tf.keras.utils.to_categorical(labels, num_classes=total_words)


embedding_size = 250
node_count     = 150
dropout_rate   = 0.3

model = Sequential([
    Embedding(total_words, embedding_size),
    Bidirectional(LSTM(node_count, return_sequences=True)),
    Dropout(dropout_rate),
    Bidirectional(LSTM(node_count)),
    Dropout(dropout_rate),
    Dense(total_words, activation="softmax"),
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
model.summary()


lr_scheduler = ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1,
)

history = model.fit(xs, ys, epochs=EPOCHS, callbacks=[lr_scheduler], verbose=1)

print("Saving model and tokenizer...")
model.save("lyric_model.keras")

with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("sequence_length.json", "w") as jf:
    json.dump({"max_len_sequence": int(max_len_sequence)}, jf)

print("Training complete and model saved.")
