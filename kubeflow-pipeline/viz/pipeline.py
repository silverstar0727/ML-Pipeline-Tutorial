import kfp
from kfp import dsl

KUBEFLOW_HOST = "http://5815e2f0459871a1-dot-us-east1.pipelines.googleusercontent.com"

def confusion_matrix_pipeline():
  dsl.ContainerOp(
    name='confusion-matrix',
    image='silverstar456/kubeflow:CM',
    output_artifact_paths={'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
  )


if __name__ == '__main__':
    arguments = {}
    client = kfp.Client(host = KUBEFLOW_HOST)
    run = client.create_run_from_pipeline_func(confusion_matrix_pipeline, arguments=arguments, experiment_name='Sample Experiment')