# The following code was taken from the GitHub repository of https://github.com/mlech26l/ncps, and only serves as a run-through of the code that is listed there. This program is not meant to be a copy, 
# but for learning

from ncps.wirings import AutoNCP
from ncps.tf import LTC
import tensorflow as tf
height, width, channels = (78, 200, 3)

ncp = LTC(AutoNCP(32, output_size=8), return_sequences=True)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(None, height, width, channels)),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(32, (5, 5), activation="relu")
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(64, (5, 5), activation="relu")
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation="relu")),
        ncp,
        tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax")),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='sparse_categorical_crossentropy',
)
