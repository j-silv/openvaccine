from openvaccine.common import early_stop


def test_empty():
    stop_training, save_loss = early_stop(
        [],
        [],
        1,
        0
    )

    assert (stop_training == False) and (save_loss == True)


def test_no_patience():
    stop_training, save_loss = early_stop(
        [3.0, 1.0],
        [],
        2.0,
        0
    )

    assert (stop_training == True) and (save_loss == False)

def test_one_patience_no_stop():
    losses = [3.0, 1.0]
    patience_losses = []

    stop_training, save_loss = early_stop(
        losses,
        patience_losses,
        2.0,
        1
    )
    
    assert (stop_training == False) and (save_loss == False)
    assert losses == [3.0, 1.0]
    assert patience_losses == [2.0]


def test_patience_stop():
    losses = [3.0, 1.0]
    patience_losses = [2.0, 1.9]

    stop_training, save_loss = early_stop(
        losses,
        patience_losses,
        2.5,
        2
    )
    
    assert (stop_training == True) and (save_loss == False)

def test_save_loss():
    losses = [3.0, 1.0]
    patience_losses = [2.0, 1.9]

    stop_training, save_loss = early_stop(
        losses,
        patience_losses,
        0.5,
        2
    )
    
    assert (stop_training == False) and (save_loss == True)
    assert losses == [3.0, 1.0, 0.5]
    assert patience_losses == []    
