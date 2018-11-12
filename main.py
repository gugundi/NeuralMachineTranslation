import json
from parse import parse_config
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    args = sys.argv
    if len(args) < 2:
        config_path = 'configs/default.json'
    else:
        config_path = args[1]
    with open(config_path, 'r') as f:
        config = json.load(f)
    parsed_config = parse_config(config)
    # assume we have data...
    train_iter = None
    val_iter = None
    train(train_iter, val_iter, parsed_config)


def train(train_iter, val_iter, parsed_config):
    epochs = parsed_config.get('epochs')
    loss_fn = parsed_config.get('loss_fn')
    model = parsed_config.get('model')
    optimizer = parsed_config.get('optimizer')
    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            model.train()
            output = model(batch.text)
            batch_loss = loss_fn(output, batch.label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()


def test(model, validationset):
    pass


if __name__ == '__main__':
    main()
