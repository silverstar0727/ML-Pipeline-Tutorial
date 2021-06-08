import kfp

KUBEFLOW_HOST = "http://5815e2f0459871a1-dot-us-east1.pipelines.googleusercontent.com"

@kfp.dsl.pipeline(
    name='Pipeline Metrics',
    description='Export and visualize pipeline metrics'
)
def pipeline_metrics_pipeline():
    kfp.dsl.ContainerOp(
        name='mnist-kfp-metrics',
        image='silverstar456/kubeflow:5',
        output_artifact_paths={'mlpipeline-metrics': '/mlpipeline-metrics.json'}
    )


pipeline_package_path = 'pipeline_metrics_pipeline.zip'
kfp.compiler.Compiler().compile(pipeline_metrics_pipeline, pipeline_package_path)

client = kfp.Client(host = KUBEFLOW_HOST)
my_experiment = client.create_experiment(name='Sample Experiment')
my_run = client.run_pipeline(my_experiment.id, 'pipeline_metrics_pipeline', pipeline_package_path)