import kfp
from kfp import dsl

KUBEFLOW_HOST = "https://1057f88936d72de2-dot-us-central1.pipelines.googleusercontent.com"

def train() -> NamedTuple('output', [('mlpipeline_metrics', 'metrics')]):
    from collections import namedtuple
    import json
    import tensorflow as tf

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

    output = namedtuple('output', ['mlpipeline_metrics'])
    return output(json.dumps(metrics))

train_component = kfp.components.func_to_container_op(train, base_image='tensorflow/tensorflow:2.1.0-py3')

def calc_pipeline():
    training = train_component()
    
if __name__ == "__main__":
    kfp.Client(host=KUBEFLOW_HOST).create_run_from_pipeline_func(
        calc_pipeline,
        arguments={},
        experiment_name='metrics')