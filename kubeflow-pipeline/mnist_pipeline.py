import kfp
import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.onprem as onprem

platform = 'onprem'
EXPERIMENT_NAME = 'MNIST pipeline'        # Name of the experiment in the UI
KUBEFLOW_HOST = "http://127.0.0.1:8080/pipeline"


@dsl.pipeline(
    name='MNIST',
    description='A pipeline to train and serve the MNIST example.'
)
def mnist_pipeline(model_export_dir='gs://your-bucket/export',
                   train_steps='200',
                   learning_rate='0.01',
                   batch_size='100',
                   pvc_name=''):
    """
    Pipeline with three stages:
      1. train an MNIST classifier
      2. deploy a tf-serving instance to the cluster
      3. deploy a web-ui to interact with it
    """
    train = dsl.ContainerOp(
        name='train',
        image='gcr.io/kubeflow-examples/mnist/model:v20190304-v0.2-176-g15d997b',
        arguments=[
            "/opt/model.py",
            "--tf-export-dir", model_export_dir,
            "--tf-train-steps", train_steps,
            "--tf-batch-size", batch_size,
            "--tf-learning-rate", learning_rate
        ]
    )

    serve_args = [
        '--model-export-path', model_export_dir,
        '--server-name', "mnist-service"
    ]
    if platform != 'GCP':
        serve_args.extend([
            '--cluster-name', "mnist-pipeline",
            '--pvc-name', pvc_name
        ])

    serve = dsl.ContainerOp(
        name='serve',
        image='gcr.io/ml-pipeline/ml-pipeline-kubeflow-deployer:'
              '7775692adf28d6f79098e76e839986c9ee55dd61',
        arguments=serve_args
    )
    serve.after(train)

    webui_args = [
        '--image', 'gcr.io/kubeflow-examples/mnist/web-ui:'
                   'v20190304-v0.2-176-g15d997b-pipelines',
        '--name', 'web-ui',
        '--container-port', '5000',
        '--service-port', '80',
        '--service-type', "LoadBalancer"
    ]
    if platform != 'GCP':
        webui_args.extend([
            '--cluster-name', "mnist-pipeline"
        ])

    web_ui = dsl.ContainerOp(
        name='web-ui',
        image='gcr.io/kubeflow-examples/mnist/deploy-service:latest',
        arguments=webui_args
    )
    web_ui.after(serve)

    steps = [train, serve, web_ui]
    for step in steps:
        if platform == 'GCP':
            step.apply(gcp.use_gcp_secret('user-gcp-sa'))
        else:
            step.apply(onprem.mount_pvc(pvc_name, 'task-pv-volume', '/mnt'))


if __name__ == "__main__":
    kfp.Client(host=KUBEFLOW_HOST).create_run_from_pipeline_func(
        mnist_pipeline,
        arguments={
            "model_export_dir": "/mnt/export",
            "pvc_name": "task-pv-claim"},
        experiment_name=EXPERIMENT_NAME)