import json
import os
from parse import parse_config
import sys
from tensorboardX import SummaryWriter
from time import time
import torch
import torch.nn as nn
import torch.optim as optim


# generate random data for test script...
class Data:

    def __init__(self, text, label):
        self.text = text
        self.label = label


class Iterator:

    def __init__(self):
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= 1000:
            self.index = 0
            raise StopIteration
        self.index += 1
        X = torch.rand(100, 10)
        y = torch.zeros(100).long()
        X_sum = torch.sum(X, 1)
        ones = X_sum > 0.5
        y[ones] = 1
        data = Data(X, y)
        return data


def main():
    args = sys.argv
    if len(args) < 2:
        config_path = 'configs/default.json'
    else:
        config_path = args[1]
    with open(config_path, 'r') as f:
        config = json.load(f)
    parsed_config = parse_config(config)
    main_path = os.path.dirname(os.path.realpath(__file__))
    name = parsed_config.get('name')
    writer_path = get_or_create_dir(main_path, f'.logs/{name}')
    # TODO: call data loader here
    train_iter = Iterator()
    val_iter = Iterator()
    train(train_iter, val_iter, writer_path, parsed_config)


def train(train_iter, val_iter, writer_path, parsed_config):
    writer_train_path = get_or_create_dir(writer_path, 'train')
    writer_val_path = get_or_create_dir(writer_path, 'val')
    writer_train = SummaryWriter(log_dir=writer_train_path)
    writer_val = SummaryWriter(log_dir=writer_val_path)
    epochs = parsed_config.get('epochs')
    loss_fn = parsed_config.get('loss_fn')
    model = parsed_config.get('model')
    optimizer = parsed_config.get('optimizer')
    training = parsed_config.get('training')
    eval_every = training.get('eval_every')
    step = 1
    for epoch in range(epochs):
        for i, train_batch in enumerate(train_iter):
            model.train()
            output = model(train_batch.text)
            batch_loss = loss_fn(output, train_batch.label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            timestamp = time()
            writer_train.add_scalar('loss', batch_loss, step, timestamp)

            if (i + 1) % eval_every == 0:
                model.eval()
                val_losses = 0
                val_lengths = 0
                for val_batch in val_iter:
                    val_output = model(val_batch.text)
                    val_loss = loss_fn(val_output, val_batch.label)
                    val_losses += val_loss
                    val_lengths += 1

                val_loss = val_losses / val_lengths
                writer_val.add_scalar('loss', val_loss, step, timestamp)

            step += 1


def test(model, validationset):
    pass


def get_or_create_dir(base_path, dir_name):
    out_directory = os.path.join(base_path, dir_name)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    return out_directory


if __name__ == '__main__':
    main()
