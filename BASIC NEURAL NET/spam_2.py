import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub

cls = lambda: os.system("cls")
df0 = pd.read_csv("spam.csv")
df = df0.copy()

train, valid, test = np.split(
    df.sample(frac=1), [int(0.8 * len(df)), int(0.9 * len(df))]
)


def df_to_dataset(dataframe, shuffle=True, batch_size=500):
    df = dataframe.copy()
    labels = df.pop("class")
    df = df["message"]
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train = df_to_dataset(train)
test = df_to_dataset(test)
valid = df_to_dataset(valid)
# convert text to numeric values
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)
hub_layer(list(train)[0][0])
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)
history = model.fit(train, epochs=10, validation_data=valid)


def predict_class(text):
    text = [text]
    prediction = model.predict(text)
    if prediction[0] >= 0.5:
        return "spam"
    else:
        return "ham"


while True:
    text_to_predict = input("Enter text to predict: ")
    predicted_class = predict_class(text_to_predict)
    print("Predicted class:", predicted_class)
