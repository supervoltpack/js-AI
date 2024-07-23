# Library Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Data loading
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
# optional
# print(info)
# print(dataset)
# print(dataset["train])
