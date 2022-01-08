def sqrt_decay_1(epoch, lr):
    decay_epochs = 100
    initial_lr = 1e-4
    end_lr = 1e-5
    epoch = min(epoch, decay_epochs)
    power=0.5
    return ((initial_lr-end_lr)*(1-epoch/decay_epochs)**(power))+end_lr