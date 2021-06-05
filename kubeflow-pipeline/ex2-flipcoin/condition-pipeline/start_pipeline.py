import kfp
from kfp import dsl


def flip_coin_op():
    return dsl.ContainerOp(
        name='Flip coin',
        image='silverstar456/kubeflow:1',
        file_outputs={'output': '/tmp/output'}
    )


def print_op(msg):
    """Print a message."""
    return dsl.ContainerOp(
        name='Print',
        image='alpine:3.6',
        command=['echo', msg],
    )


@dsl.pipeline(
    name='Conditional execution pipeline',
    description='Shows how to use dsl.Condition().'
)
def condition_pipeline():
    flip = flip_coin_op()
    with dsl.Condition(flip.output == 'heads'):
        print_op('YOUT WIN')

    with dsl.Condition(flip.output == 'tails'):
        print_op('YOU LOSE')


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(condition_pipeline, 'condition' + '.zip')

    client = kfp.Client(host=KUBEFLOW_HOST)
    my_experiment = client.create_experiment(name='Basic Experiment')
    my_run = client.run_pipeline(my_experiment.id, 'Condition pipeline', 'condition' + '.zip')