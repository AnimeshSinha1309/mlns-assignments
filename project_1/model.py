import tensorflow as tf

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.InputLayer(
            input_shape=(
                28,
                28,
            ),
            name="Block0-Input",
        ),
        tf.keras.layers.Reshape((28, 28, 1), name="Block0-Reshape"),
        # Pad, Convolve, Batch Normalize, ReLU
        tf.keras.layers.ZeroPadding2D((3, 3), name="Block1-Padding"),
        tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), name="Block1-Convolution"),
        tf.keras.layers.BatchNormalization(axis=3, name="Block1-Normalization"),
        tf.keras.layers.Activation("relu", name="Block1-Activation"),
        # Convolve, Batch Normalize, ReLU, Pool
        tf.keras.layers.Conv2D(32, (7, 7), strides=(1, 1), name="Block2-Convolution"),
        tf.keras.layers.BatchNormalization(axis=3, name="Block2-Normalization"),
        tf.keras.layers.Activation("relu", name="Block2-Activation"),
        tf.keras.layers.MaxPooling2D((2, 2), name="Block2-MaxPool"),
        # Flatten, Fully Connected, Sigmoid
        tf.keras.layers.Flatten(name="Block3-Flatten"),
        tf.keras.layers.Dense(10, activation="softmax", name="Block3-Dense"),
    ]
)


def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        x=x_train,
        y=y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath="weights/mnist.ckpt", save_weights_only=True, verbose=1
            ),
        ],
    )


if __name__ == "__main__":
    train()
