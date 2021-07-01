import kfp
from kfp import dsl

KUBEFLOW_HOST = "https://1057f88936d72de2-dot-us-central1.pipelines.googleusercontent.com"

@kfp.components.func_to_container_op
def add(a: float, b: float) -> float:
    print(a, '+', b, '=', a + b)
    return a + b
    
def calc_pipeline(a: float = 0, b: float = 7):
    add_task = add(a, 4)
    add_2_task = add(a, b)
    add_3_task = add(add_task.output, add_2_task.output)


if __name__ == "__main__":
    kfp.Client(host=KUBEFLOW_HOST).create_run_from_pipeline_func(
        calc_pipeline,
        arguments={},
        experiment_name='add number')