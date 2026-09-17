from backend.app.services.experiment_runner import ExperimentRunner


config = {
    "dataset_file": "data/input.txt",

    "model_id": "10k",

    "n_layer": 1,
    "n_head": 1,
    "n_embd": 32,
    "block_size": 64,

    "batch_size": 8,

    "learning_rate": 3e-4,

    "max_steps": 100,

    "device": "mps",
}


runner = ExperimentRunner(config)

result = runner.run()

print()
print("Final result:")
print("Status:", result["status"])
print("Parameters:", result["actual_parameters"])
print("Training time:", result["training_time"])