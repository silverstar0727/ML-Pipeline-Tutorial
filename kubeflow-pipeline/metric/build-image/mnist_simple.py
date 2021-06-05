from __future__ import absolute_import, division, print_function, unicode_literals
import json
import tensorflow as tf
from tensorflow.python.lib.io import file_io

def train():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    training_history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)
    loss = results[0]
    accuracy = results[1]
    metrics = {
        'metrics': [{
            'name': 'accuracy',
            'numberValue': float(accuracy),
            'format': "PERCENTAGE",
        }, {
            'name': 'loss',
            'numberValue': float(loss),
            'format': "RAW",
        }]
    }
    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    train()