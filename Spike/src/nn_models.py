from typing import Tuple
import tensorflow.keras.layers as tf_layer
import tensorflow.keras.initializers as tf_init
from tensorflow.keras.models import Sequential, Model

import matplotlib.pyplot as plt

def build_model_regression_v0(num_features: int, num_classes: int) -> Sequential:
    # Einfaches Modell für die Klassifikation
    model = Sequential()
    nHiddenLayer = [50, 50]

    # Hidden Layer #1 with Input
    model.add(tf_layer.Dense(units=nHiddenLayer[0], input_shape=(num_features,)))
    model.add(tf_layer.Activation("relu"))

    # Hidden Layer #2
    model.add(tf_layer.Dense(units=nHiddenLayer[1], input_shape=(num_features,)))
    model.add(tf_layer.Activation("relu"))

    # Output Layer
    model.add(tf_layer.Dense(units=num_classes))
    model.add(tf_layer.Activation("softmax"))
    # model.summary()

    return model

def build_dnn_class_v0(num_features: int, num_classes: int) -> Sequential:
    init_w = tf_init.RandomUniform(minval=-0.05, maxval=0.05)
    init_b = tf_init.Constant(value=0.0)
    # Einfaches Modell für die Klassifikation
    model = Sequential()
    nHiddenLayer = [100, 100]

    # Hidden Layer #1 with Input
    model.add(tf_layer.Dense(units=nHiddenLayer[0], kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,)))
    model.add(tf_layer.Activation("relu"))

    # Hidden Layer #2
    model.add(tf_layer.Dense(units=nHiddenLayer[1], kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,)))
    model.add(tf_layer.Activation("relu"))

    # Output Layer
    model.add(tf_layer.Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(tf_layer.Activation("softmax"))
    #model.summary()

    return model

def build_dnn_class_v1(num_features: int, num_classes: int) -> Sequential:
    init_w = tf_init.RandomUniform(minval=-0.10, maxval=0.1)
    init_b = tf_init.Constant(value=0.0)
    # Einfaches Modell für die Klassifikation
    model = Sequential()
    nHiddenLayer = [20, 50]

    # Hidden Layer #1 with Input
    model.add(tf_layer.Dense(units=nHiddenLayer[0], kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features,)))
    model.add(tf_layer.Activation("relu"))

    # Hidden Layer #2
    model.add(tf_layer.Dense(units=nHiddenLayer[1], kernel_initializer=init_w, bias_initializer=init_b))
    model.add(tf_layer.Activation("relu"))

    # Output Layer
    model.add(tf_layer.Dense(units=num_classes, kernel_initializer=init_w, bias_initializer=init_b, activation="softmax"))
    #model.summary()

    return model

def build_cnn_class_v0(img_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    filter_size = [32, 40, 16]

    model = Sequential()
    model.add(tf_layer.Conv2D(filters=filter_size[0], kernel_size=(3, 3), padding="same", input_shape=img_shape))
    #model.add(tf_layer.Dropout(0.25))
    model.add(tf_layer.Activation("relu"))
    model.add(tf_layer.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf_layer.Conv2D(filters=filter_size[1], kernel_size=(3, 3), padding="same"))
    model.add(tf_layer.Activation("relu"))
    model.add(tf_layer.MaxPooling2D(pool_size=(2, 2)))

    # --- überführung ins DNN zur Klassifikation
    model.add(tf_layer.Flatten())
    model.add(tf_layer.Dense(units=num_classes, activation="softmax"))
    return model

def build_cnn_class_v1(img_shape: Tuple[int, int, int], num_classes: int) -> Model:
    filter_size = [32, 40, 16]

    input_data = tf_layer.Input(shape=img_shape)
    x0 = tf_layer.Conv2D(filters=filter_size[0], kernel_size=(3, 3), padding="same")(input_data)
    x1 = tf_layer.Activation("relu")(x0)
    x2 = tf_layer.Conv2D(filters=filter_size[1], kernel_size=(3, 3), padding="same")(x1)
    x3 = tf_layer.Activation("relu")(x2)
    x4 = tf_layer.MaxPooling2D()(x3)

    # --- überführung ins DNN zur Klassifikation (Fully Connected)
    x5 = tf_layer.Flatten()(x4)
    x6 = tf_layer.Dense(units=num_classes)(x5)
    y_pred = tf_layer.Activation("softmax")(x6)

    model = Model(
        inputs = [input_data],
        outputs = [y_pred]
    )

    return model


def build_autoencoder_v0(img_shape: Tuple[int, int, int]) -> Sequential:
    filter_size = 16

    encoder = Sequential()
    encoder.add(tf_layer.Conv2D(filters=filter_size, kernel_size=(3, 3), input_shape=img_shape))
    encoder.add(tf_layer.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(tf_layer.Conv2D(filters=filter_size, kernel_size=(3, 3)))
    encoder.add(tf_layer.MaxPooling2D(pool_size=(2, 2)))

    decoder = Sequential()
    decoder.add(tf_layer.Conv2D(filters=filter_size, kernel_size=(3, 3), input_shape=img_shape))
    decoder.add(tf_layer.UpSampling2D(pool_size=(2, 2)))
    decoder.add(tf_layer.Conv2D(filters=filter_size, kernel_size=(3, 3)))
    decoder.add(tf_layer.UpSampling2D(pool_size=(2, 2)))

    model = Sequential()
    model.add(encoder)
    model.add(decoder)

    return model

def plot_weights(model: Model):
    first_conv_layer = model.layers[1]
    layer_weights = first_conv_layer.get_weights()
    kernels = layer_weights[0]

    num_filters = kernels.shape[3]
    subplot_grid = (num_filters // 4, 4)

    (fig, ax) = plt.subplots(subplot_grid[0], subplot_grid[1], figsize=(20, 20))
    ax = ax.reshape(num_filters)

    for idx in range(num_filters):
        ax[idx].imshow(kernels[:, :, 0, idx], cmap="gray")

    ax = ax.reshape(subplot_grid)
    fig.subplots_adjust(hspace=0.5)
    plt.show()



