from model import Encoder, Decoder
import torch.nn as nn
import torch.optim as optim


def parse_config(config):
    optimizer_config = config.get('optimizer')
    loss_fn = get_loss_fn(config)
    parsed_config = {
        "batch_size": config.get('batch_size'),
        "attention": config.get('attention'),
        "epochs": config.get('epochs'),
        "loss_fn": loss_fn,
        "name": config.get('name'),
        "rnn": config.get('rnn'),
        "source_vocabulary_size": config.get('source_vocabulary_size'),
        "target_vocabulary_size": config.get('target_vocabulary_size'),
        "training": config.get('training'),
    }
    encoder = Encoder(parsed_config)
    encoder_optimizer = get_optimizer(optimizer_config, encoder)
    decoder = Decoder(parsed_config)
    decoder_optimizer = get_optimizer(optimizer_config, decoder)
    parsed_config["model"] = encoder, decoder
    parsed_config["optimizer"] = encoder_optimizer, decoder_optimizer
    return parsed_config


def get_loss_fn(config):
    loss_fn = config.get('loss_fn')
    if loss_fn == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif loss_fn == 'NLLLoss':
        return nn.NLLLoss()
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
