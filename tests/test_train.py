import pytest

from train import (
    checkpoint_callback_kwargs,
    logging_trainer_kwargs,
    validation_trainer_kwargs,
)


def test_validation_interval_is_converted_from_optimizer_steps_to_batches():
    assert validation_trainer_kwargs({
        "validation_check_interval_steps": 100,
        "accumulate_grad_batches": 64,
    }) == {
        "val_check_interval": 6400,
        "check_val_every_n_epoch": None,
    }


def test_validation_interval_defaults_to_end_of_epoch():
    assert validation_trainer_kwargs({"accumulate_grad_batches": 64}) == {}


def test_logging_interval_defaults_to_every_optimizer_step():
    assert logging_trainer_kwargs({}) == {"log_every_n_steps": 1}
    assert logging_trainer_kwargs({"log_every_n_steps": 10}) == {
        "log_every_n_steps": 10,
    }


def test_checkpoint_interval_defaults_to_every_epoch_and_is_configurable():
    assert checkpoint_callback_kwargs({}) == {"every_n_epochs": 1}
    assert checkpoint_callback_kwargs({"checkpoint_every_n_epochs": 5}) == {
        "every_n_epochs": 5,
    }


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_validation_interval_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="validation_check_interval_steps"):
        validation_trainer_kwargs({"validation_check_interval_steps": value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_logging_interval_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="log_every_n_steps"):
        logging_trainer_kwargs({"log_every_n_steps": value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5"])
def test_checkpoint_interval_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="checkpoint_every_n_epochs"):
        checkpoint_callback_kwargs({"checkpoint_every_n_epochs": value})