import pytest

from train import logging_trainer_kwargs, validation_trainer_kwargs


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


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_validation_interval_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="validation_check_interval_steps"):
        validation_trainer_kwargs({"validation_check_interval_steps": value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_logging_interval_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="log_every_n_steps"):
        logging_trainer_kwargs({"log_every_n_steps": value})