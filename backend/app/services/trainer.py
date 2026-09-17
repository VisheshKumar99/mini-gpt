from backend.app.services.experiment_runner import ExperimentRunner


def run_training(config: dict) -> dict:
    runner = ExperimentRunner(config)
    return runner.run()