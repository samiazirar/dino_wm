from train import target_epoch_range


def test_resume_epochs_are_target_total():
    assert list(target_epoch_range(0, 3)) == [1, 2, 3]
    assert list(target_epoch_range(2, 3)) == [3]
    assert list(target_epoch_range(3, 3)) == []
    assert list(target_epoch_range(5, 3)) == []
