import kfp
from kfp import dsl


def flip_coin_op():
    import random

    if random.randint(0,1) == 0:
        result = "heads" 
    else:
        result = 'tails'

    return result


def print_op(msg):
    """Print a message."""
    return dsl.ContainerOp(
        name='Print',
        image='alpine:3.6',
        command=['echo', msg],
    )


@dsl.pipeline(
    name='Sequential pipeline',
    description='A pipeline with two sequential steps.'
)
def sequential_pipeline():
    """A pipeline with two sequential steps."""

    flip = kfp.components.func_to_container_op(flip_coin_op)
    print_op(flip.output)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sequential_pipeline, 'sequential.zip')

    client = kfp.Client(host = KUBEFLOW_HOST)
    my_experiment = client.create_experiment(name='Basic Experiment')
    my_run = client.run_pipeline(my_experiment.id, 'Sequential pipeline', 'sequential.zip')