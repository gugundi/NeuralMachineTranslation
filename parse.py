import argparse
from data_loader import load_debug, load_dummy_fixed_length, load_dummy_variable_length, load_iwslt, load_multi30k
import json
from model import Model
from model_without_attention import ModelWithoutAttention
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_or_create_dir


def get_config(use_gpu, device, device_idx, **kwargs):
    config_path = kwargs.get('config_path', None)
    load_weights = kwargs.get('load_weights', False)
    parse_args = kwargs.get('parse_args', True)
    if parse_args:
        args = parse_arguments()
    else:
        args = DummyArgs()
    if args.config is not None:
        config_path = args.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    EOS_token = '<eos>'
    PAD_token = '<pad>'
    SOS_token = '<sos>'
    config['EOS_token'] = EOS_token
    config['SOS_token'] = SOS_token
    config['PAD_token'] = PAD_token
    if args.name is not None:
        config['name'] = args.name
    name = config.get('name')
    file_path = os.path.dirname(os.path.realpath(__file__))
    config['weights_path'] = get_or_create_dir(file_path, f'.weights/{name}')
    if load_weights:
        model_data_path = get_or_create_dir(file_path, f'model-data/{name}')
        config['model_data_path'] = model_data_path
        language_path = f'{model_data_path}/language.json'
        with open(language_path, 'r') as f:
            language_data = json.load(f)
        src_language = Language(language_data.get('source'))
        trg_language = Language(language_data.get('target'))
    else:
        if args.debug:
            train_iter, val_iter, src_language, trg_language, val_dataset = load_debug(config, device)
        elif args.dummy_fixed_length:
            train_iter, val_iter, src_language, trg_language, val_dataset = load_dummy_fixed_length(config, device)
        elif args.dummy_variable_length:
            train_iter, val_iter, src_language, trg_language, val_dataset = load_dummy_variable_length(config, device)
        elif args.iwslt:
            train_iter, val_iter, src_language, trg_language, val_dataset = load_iwslt(config, device)
        else:
            train_iter, val_iter, src_language, trg_language, val_dataset = load_multi30k(config, device)
        config['writer_path'] = get_or_create_dir(file_path, f'.logs/{config.get("name")}')
        config['train_iter'] = train_iter
        config['val_iter'] = val_iter
        config['val_dataset'] = val_dataset
        config['teacher_forcing'] = config.get('teacher_forcing', 0)
    config['source_vocabulary_size'] = len(src_language.itos)
    config['target_vocabulary_size'] = len(trg_language.itos)
    config['EOS'] = trg_language.stoi[EOS_token]
    config['PAD_src'] = src_language.stoi[PAD_token]
    config['PAD_trg'] = trg_language.stoi[PAD_token]
    config['SOS'] = trg_language.stoi[SOS_token]
    config['src_language'] = src_language
    config['trg_language'] = trg_language
    config['window_size'] = config.get('attention').get('window_size')
    config['input_feeding'] = config.get('input_feeding', False)
    config['use_attention'] = config.get('attention').get('enabled', True)
    if config.get('use_attention'):
        config['model'] = Model(config, device)
    else:
        config['model'] = ModelWithoutAttention(config, device)
    if use_gpu:
        config["model"] = config["model"].to(device)
    if load_weights:
        model_path = f'{model_data_path}/model'
        config['model'].load_state_dict(torch.load(model_path, map_location=device))
    config['optimizer'] = get_optimizer(config.get('optimizer'), config['model'])
    config['loss_fn'] = nn.CrossEntropyLoss()
    return config


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    dummy_fixed_length_help = 'Dummy data with fixed length.'
    dummy_variable_length_help = 'Dummy data with variable length.'
    iwslt_help = 'IWSLT dataset.'
    parser = argparse.ArgumentParser(description='Train machine translation model.')
    parser.add_argument('--config', type=str, nargs='?', default='configs/default.json', help='Path to model configuration.')
    parser.add_argument('--debug', type=str2bool, default=False, const=True, nargs='?', help='Debug mode.')
    parser.add_argument('--dummy_fixed_length', type=str2bool, default=False, const=True, nargs='?', help=dummy_fixed_length_help)
    parser.add_argument('--dummy_variable_length', type=str2bool, default=False, const=True, nargs='?', help=dummy_variable_length_help)
    parser.add_argument('--iwslt', type=str2bool, default=False, const=True, nargs='?', help=iwslt_help)
    parser.add_argument('--name', default=None, type=str, help='Name used when writing to tensorboard.')
    return parser.parse_args()


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


class DummyArgs():
    config = None
    debug = False
    dummy_fixed_length = False
    dummy_variable_length = False
    iwslt = False
    name = None


class Language:

    def __init__(self, data):
        self.itos = data.get('itos')
        self.stoi = data.get('stoi')
