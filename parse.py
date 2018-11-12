from model import Model
import torch.nn as nn
import torch.optim as optim


def parse_config(config):
    model = Model()
    optimizer_config = config.get('optimizer')
    optimizer = get_optimizer(optimizer_config, model)
    loss_fn = get_loss_fn(config)
    return {
        "batch_size": config.get('batch_size'),
        "epochs": config.get('epochs'),
        "loss_fn": loss_fn,
        "model": model,
        "name": config.get('name'),
        "optimizer": optimizer,
        "training": config.get('training')
    }


def get_loss_fn(config):
    loss_fn = config.get('loss_fn')
    if loss_fn == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        raise Exception(f'Unknown loss function: {loss_fn}')


def get_optimizer(config, model):
    type = config.get('type')
    learning_rate = config.get('learning_rate')
    weight_decay = config.get('weight_decay', 0)
    if type == 'SGD':
        momentum = config.get('momentum', 0)
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif type == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise Exception(f'Unknown optimizer: {type}')
