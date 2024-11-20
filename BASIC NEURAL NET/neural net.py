import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

cls = lambda: os.system("cls")
path = r"C:\Users\ACER\Desktop\projects\data\magic.data"
df0 = pd.read_csv(path)
df = df0.copy()
df = df.drop_duplicates()
df["class"] = (df["class"] == "g").astype(int)
columns = [a for a in df.columns]
columns.remove("class")
x = df[columns].values
y = df["class"].values
ros = RandomOverSampler()
x, y = ros.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
nn_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = nn_model.fit(
    x_train, y_train, epochs=100, batch_size=32, validation_split=0.2
)


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history["loss"], label="loss")
    ax1.plot(history.history["val_loss"], label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary crossentropy")
    ax1.grid(True)

    ax2.plot(history.history["accuracy"], label="accuracy")
    ax2.plot(history.history["val_accuracy"], label="val_accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    plt.show()


plot_history(history)
