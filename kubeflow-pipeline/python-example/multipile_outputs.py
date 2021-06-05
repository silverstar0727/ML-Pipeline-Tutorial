import kfp
from kfp import dsl
import kfp.components as comp
from typing import NamedTuple

KUBEFLOW_HOST = "http://5815e2f0459871a1-dot-us-east1.pipelines.googleusercontent.com"

def print_op(msg):
    return dsl.ContainerOp(
        name = "print",
        image = "alpine:3.6",
        command = ['echo', msg]
    )

@comp.func_to_container_op
def add_multiply_two_nums(a: float, b: float) -> NamedTuple(
    'Outputs', [('sum', float), ('product', float)]
):
    return (a + b, a*b)

@dsl.pipeline(
    name = "Multiple outputs pipeline",
    description = "A pipeline to showcase"
)
def multiple_outputs_pipeline(a = "10", b = "20"):
    add_multiply_task = add_multiply_two_nums(a, b)
    print_op(f"sum={add_multiply_task.outputs['sum']}, \
        product={add_multiply_task.outputs['product']}")

if __name__ == "__main__":
    arguments = {'a': '3', 'b': '4'}
    kfp.compiler.Compiler().compile(multiple_outputs_pipeline, 'multiple-ouptuts.zip')

    client = kfp.Client(host=KUBEFLOW_HOST)
    my_experiment = client.create_experiment(name='Basic Experiment')
    my_run = client.run_pipeline(my_experiment.id, 'multiple-output', 'multiple-outputs.zip')