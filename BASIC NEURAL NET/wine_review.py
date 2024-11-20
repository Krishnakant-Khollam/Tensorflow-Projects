import os
import numpy as np
import pandas as pd

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import tensorflow_hub as hub

cls = lambda: os.system("cls")
path = r"C:\Users\ACER\Desktop\projects\data\wine_reviews.csv"
df0 = pd.read_csv(path)
df = df0.copy()
df["class"] = (df.points >= 90).astype(int)
columns = ["description", "class"]
df = df[columns]

train, valid, test = np.split(
    df.sample(frac=1), [int(0.8 * len(df)), int(0.9 * len(df))]
)


def df_to_dataset(dataframe, shuffle=True, batch_size=1024):
    df = dataframe.copy()
    labels = df.pop("class")
    df = df["description"]
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
model.evaluate(train)
model.evaluate(valid)
history = model.fit(train, epochs=10, validation_data=valid)
