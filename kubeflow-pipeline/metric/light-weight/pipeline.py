from typing import Collection, NamedTuple

import kfp
from kfp.components import func_to_container_op

KUBEFLOW_HOST = "http://5815e2f0459871a1-dot-us-east1.pipelines.googleusercontent.com"

@func_to_container_op
def train() -> NamedTuple('output', [('mlpipeline_metrics', 'metrics')]):
    import json
    loss = 0.812345
    accuracy = 0.9712345
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
    from collections import namedtuple

    output = namedtuple('output', ['mlpipeline_metrics'])
    return output(json.dumps(metrics))


def pipeline_metrics_fn_pipeline():
    train()


if __name__ == '__main__':
    arguments = {}
    client = kfp.Client(host = KUBEFLOW_HOST)
    run = client.create_run_from_pipeline_func(pipeline_metrics_fn_pipeline, arguments=arguments, experiment_name='Sample Experiment')