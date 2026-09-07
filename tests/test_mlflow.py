from pathlib import Path

import mlflow

from src.lightning_module import PMFColorizerModule, _select_validation_visuals
from train import build_logger, log_mlflow_metadata


def test_validation_visual_selection_defaults_to_sorted_first_images():
    visuals = {"image-z": ("z",), "image-a": ("a",), "image-m": ("m",)}

    selected = _select_validation_visuals(visuals, set(), 2)

    assert list(selected) == ["image-a", "image-m"]


def test_validation_visual_selection_preserves_requested_ids():
    visuals = {"image-z": ("z",), "image-a": ("a",), "image-m": ("m",)}

    selected = _select_validation_visuals(visuals, {"image-m", "missing"}, 2)

    assert list(selected) == ["image-m"]


def test_local_mlflow_logs_run_data_and_artifact(tmp_path: Path):
    database = tmp_path / "mlflow.db"
    artifact_root = tmp_path / "mlartifacts"
    config = {
        "data": {
            "train_manifest": "train.txt",
            "val_manifest": "val.txt",
            "resolution": [16, 16],
            "batch_size": 2,
            "val_batch_size": 2,
        },
        "training": {
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "max_epochs": 1,
            "max_steps": None,
            "accumulate_grad_batches": 1,
            "time_sampling": {"p_mean": 0.8},
        },
        "logger": {
            "type": "mlflow",
            "tracking_uri": f"sqlite:///{database}",
            "artifact_location": f"file:{artifact_root}",
            "experiment_name": "tmp",
            "run_name": "local-test",
        },
    }
    logger = build_logger(
        {"log_dir": str(tmp_path / "logs"), "logger": config["logger"]}
    )
    module = PMFColorizerModule(
        model={
            "resolution": [16, 16],
            "patch_size": 8,
            "hidden_size": 16,
                "depth": 2,
                "heads": 4,
                "mlp_ratio": 2.0,
                "aux_head_depth": 1,
                "pca_channels": 8,
        }
    )
    logger.log_hyperparams(config)
    log_mlflow_metadata(logger, config, module)
    artifact = tmp_path / "sample.txt"
    artifact.write_text("qualitative output\n", encoding="utf-8")
    logger.experiment.log_artifact(
        logger.run_id, str(artifact), artifact_path="qualitative"
    )
    artifact.write_text("updated qualitative output\n", encoding="utf-8")
    logger.experiment.log_artifact(
        logger.run_id, str(artifact), artifact_path="qualitative"
    )
    logger.finalize("success")

    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{database}")
    experiment = client.get_experiment_by_name("tmp")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["training/max_steps"] == "None"
    assert run.data.tags["data/train_manifest"] == "train.txt"
    assert run.data.tags["model/parameter_report"]
    assert client.list_artifacts(run.info.run_id, "qualitative")[0].path == (
        "qualitative/sample.txt"
    )
    downloaded = Path(
        client.download_artifacts(
            run.info.run_id, "qualitative/sample.txt", str(tmp_path / "download")
        )
    )
    assert downloaded.read_text(encoding="utf-8") == "updated qualitative output\n"
