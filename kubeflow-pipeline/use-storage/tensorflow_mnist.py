from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import tensorflow as tf


def train():
    print("TensorFlow version: ", tf.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./model', type=str)
    args = parser.parse_args()

    version = 1
    export_path = os.path.join(args.model_path, str(version))

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training...")
    training_history = model.fit(x_train, y_train, batch_size=64, epochs=10,
                                 validation_split=0.2)

    print('\\nEvaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    model.save(export_path)
    print('"Saved model to {}'.format(export_path))


if __name__ == '__main__':
    train()