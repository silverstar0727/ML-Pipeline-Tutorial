import kfp
from kfp import dsl
from kfp import components
from kfp.components import func_to_container_op, InputPath, OutputPaht

KUBEFLOW_HOST = "http://5815e2f0459871a1-dot-us-east1.pipelines.googleusercontent.com"

@func_to_container_op
def print_text(text_path: InputPath()):
    with open(text_path, 'r') as reader:
        for line in reader:
            print(line, end = '')

@func_to_container_op
def add(a: float, b: float) -> float:
    return a + b

@dsl.pipeline(
    name = "lightweight-components",
    description = "lightweight components with python"
)
def pipeline(a = '10', b = '20'):
    add_task = add_op(a, b)
    print_text(add_task.output)

if __name__ == "__main__":
    arguments = {'a': '1000', 'b': '4'}
    kfp.compiler.Compiler().compile(pipeline, 'lightweight-component.zip')

    client = kfp.Client(host=KUBEFLOW_HOST)
    my_experiment = client.create_experiment(name='Basic Experiment')
    my_run = client.run_pipeline(my_experiment.id, 'lightweight-components', 'lightweight-component.zip')