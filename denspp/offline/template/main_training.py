from denspp.offline.dnn import PyTorchTrainer, DefaultSettingsTraining
from denspp.offline.dnn.models import mnist


if __name__ == "__main__":
    trainer = PyTorchTrainer(
        use_case='MNIST',
        settings=DefaultSettingsTraining,
        default_model=mnist.mnist_mlp_cl_v0.__name__,
        path2config='config',
        generate_configs=True
    )
    trainer.do_plot_dataset()
    results = trainer.do_training()
    trainer.do_plot_results(results)
