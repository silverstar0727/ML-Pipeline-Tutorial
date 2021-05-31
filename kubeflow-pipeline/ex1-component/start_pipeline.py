import kfp
from kfp import dsl

KUBEFLOW_HOST = "http://5815e2f0459871a1-dot-us-east1.pipelines.googleusercontent.com"

@kfp.dsl.component
def train_component_op():
    return kfp.dsl.ContainerOp(
        name='mnist-train',
        image='silverstar456/kubeflow'
    )


@dsl.pipeline(
    name='My pipeline',
    description='My machine learning pipeline'
)
def my_pipeline():
    train_task = train_component_op()


if __name__ == '__main__':
    # Compile
    pipeline_package_path = 'my_pipeline.zip'
    kfp.compiler.Compiler().compile(my_pipeline, pipeline_package_path)

    # Run
    client = kfp.Client(host = KUBEFLOW_HOST)
    my_experiment = client.create_experiment(name='Basic Experiment')
    my_run = client.run_pipeline(my_experiment.id, 'my_pipeline', pipeline_package_path)