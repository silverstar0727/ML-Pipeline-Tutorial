import kfp
from kfp.components import func_to_container_op, InputPath, OutputPath

KUBEFLOW_HOST = "http://5815e2f0459871a1-dot-us-east1.pipelines.googleusercontent.com"

@func_to_container_op
def write_numbers(numbers_path: OutputPath(str), start: int = 1, count: int = 10):
    with open(numbers_path, 'w') as writer:
        for i in range(start, start + count):
            writer.write(str(i) + '\\n')


@func_to_container_op
def print_text(text_path: InputPath()):
    with open(text_path, 'r') as reader:
        for line in reader:
            print(line, end = '')


@func_to_container_op
def sum_multiply_numbers(
        numbers_path: InputPath(str),
        sum_path: OutputPath(str),
        product_path: OutputPath(str)):

    sum = 0
    product = 1
    with open(numbers_path, 'r') as reader:
        for line in reader:
            sum = sum + int(line)
            product = product * int(line)
    with open(sum_path, 'w') as writer:
        writer.write(str(sum))
    with open(product_path, 'w') as writer:
        writer.write(str(product))


def python_input_output_pipeline(count='10'):
    numbers_task = write_numbers(count=count)
    sum_multiply_task = sum_multiply_numbers(numbers_task.output)

    print_text(sum_multiply_task.outputs['sum'])
    print_text(sum_multiply_task.outputs['product'])


if __name__ == '__main__':
    arguments = {'count': '10'}
    kfp.compiler.Compiler().compile(python_input_output_pipeline, 'multiple-outputs.zip')

    client = kfp.Client(host=KUBEFLOW_HOST)
    my_experiment = client.create_experiment(name='Basic Experiment')
    my_run = client.run_pipeline(my_experiment.id, 'multiple-output', 'multiple-outputs.zip')
