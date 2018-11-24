import argparse
from data_loader import load_debug, load_dummy_fixed_length, load_dummy_variable_length, load_iwslt
from device import select_device, with_cpu, with_gpu
import json
import os
from parse import parse_config
from random import random, sample
from tensorboardX import SummaryWriter
import torchtext
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_or_create_dir


# TODO: visualize attention
# TODO: merge parsing of config and command line args
# TODO: clean up code - for example make config load data
# TODO: reverse source sentence for better results


def main():
    args = parse_arguments()
    config_path = args.config
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpu, device, device_idx = select_device()
    parsed_config = parse_config(config, device)
    main_path = os.path.dirname(os.path.realpath(__file__))
    name = args.name
    if name is None:
        name = parsed_config.get('name')
    writer_path = get_or_create_dir(main_path, f'.logs/{name}')
    EOS_token = '<EOS>'
    if args.debug:
        train_iter, val_iter, src_language, trg_language, _, val_dataset = load_debug(parsed_config, EOS_token, device_idx)
    elif args.dummy_fixed_length:
        train_iter, val_iter, src_language, trg_language, _, val_dataset = load_dummy_fixed_length(parsed_config, EOS_token, device_idx)
    elif args.dummy_variable_length:
        train_iter, val_iter, src_language, trg_language, _, val_dataset = load_dummy_variable_length(parsed_config, EOS_token, device_idx)
    else:
        train_iter, val_iter, src_language, trg_language, _, val_dataset = load_iwslt(parsed_config, EOS_token, device_idx)
    parsed_config['source_vocabulary_size'] = len(src_language.itos)
    parsed_config['target_vocabulary_size'] = len(trg_language.itos)
    val_data = val_iter.data()

    def sample_batches_from_val_data(k):
        return [torchtext.data.Batch(sample(val_data, 1), val_dataset, device_idx) for _ in range(k)]

    if use_gpu:
        with torch.cuda.device(device_idx):
            encoder, decoder = parsed_config['model']
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            parsed_config['model'] = encoder, decoder
            train(train_iter, src_language, trg_language, EOS_token, writer_path, parsed_config, sample_batches_from_val_data)
    else:
        train(train_iter, src_language, trg_language, EOS_token, writer_path, parsed_config, sample_batches_from_val_data)


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
    parser = argparse.ArgumentParser(description='Train machine translation model.')
    parser.add_argument('--config', type=str, nargs='?', default='configs/default.json', help='Path to model configuration.')
    parser.add_argument('--debug', type=str2bool, default=False, const=True, nargs='?', help='Debug mode.')
    parser.add_argument('--dummy_fixed_length', type=str2bool, default=False, const=True, nargs='?', help=dummy_fixed_length_help)
    parser.add_argument('--dummy_variable_length', type=str2bool, default=False, const=True, nargs='?', help=dummy_variable_length_help)
    parser.add_argument('--name', default=None, type=str, help='Name used when writing to tensorboard.')
    return parser.parse_args()


def train(train_iter, source_language, target_language, EOS_token, writer_path, parsed_config, sample_batches_from_val_data):
    EOS = target_language.stoi[EOS_token]
    writer_train_path = get_or_create_dir(writer_path, 'train')
    writer_val_path = get_or_create_dir(writer_path, 'val')
    writer_train = SummaryWriter(log_dir=writer_train_path)
    writer_val = SummaryWriter(log_dir=writer_val_path)
    epochs = parsed_config.get('epochs')
    loss_fn = parsed_config.get('loss_fn')
    encoder, decoder = parsed_config['model']
    encoder_optimizer, decoder_optimizer = parsed_config['optimizer']
    training = parsed_config.get('training')
    eval_every = training.get('eval_every')
    sample_every = training.get('sample_every')
    step = 1
    for epoch in range(epochs):
        for i, train_pair in enumerate(train_iter):
            loss = train_sentence_pair(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn, EOS, train_pair)

            timestamp = time()
            writer_train.add_scalar('loss', loss, step, timestamp)

            if (i + 1) % eval_every == 0:
                val_losses = 0
                val_lengths = 64
                val_batches = sample_batches_from_val_data(val_lengths)
                for val_pair in val_batches:
                    val_loss, _ = evaluate_sentence_pair(encoder, decoder, loss_fn, EOS, val_pair)
                    val_losses += val_loss
                val_loss = val_losses / val_lengths
                writer_val.add_scalar('loss', val_loss, step, timestamp)

            if (i + 1) % sample_every == 0:
                val_pair = sample_batches_from_val_data(1)[0]
                _, translation = evaluate_sentence_pair(encoder, decoder, loss_fn, EOS, val_pair)
                text = get_text(source_language, target_language, val_pair.src, val_pair.trg, translation, EOS_token)
                writer_val.add_text('translation', text, step, timestamp)

            step += 1


def train_sentence_pair(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn, EOS, pair):
    encoder.train()
    decoder.train()

    source_sentence = pair.src
    target_sentence = pair.trg
    encoder_hidden = encoder.init_hidden()
    source_sentence_length = source_sentence.size(0)
    target_sentence_length = target_sentence.size(0)

    encoder_output, encoder_hidden = encoder(source_sentence, encoder_hidden)
    context = encoder_output[source_sentence_length - 1]
    encoder_output = encoder_output.view(source_sentence_length, encoder.hidden_size)
    decoder_input = with_gpu(torch.LongTensor([[EOS]]))
    decoder_hidden = encoder_hidden

    loss = with_gpu(torch.FloatTensor([0]))
    for i in range(target_sentence_length):
        context = context.unsqueeze(0)
        y, context, decoder_hidden = decoder(source_sentence_length, encoder_output, decoder_input, context, decoder_hidden)
        topv, topi = y.topk(1)
        decoder_input = topi.detach()
        target = target_sentence[i].view(1)
        loss += loss_fn(y, target)
        if decoder_input.item() == EOS:
            break

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return with_cpu(loss)


def evaluate_sentence_pair(encoder, decoder, loss_fn, EOS, pair):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        source_sentence = pair.src
        target_sentence = pair.trg
        encoder_hidden = encoder.init_hidden()
        source_sentence_length = source_sentence.size(0)
        target_sentence_length = target_sentence.size(0)

        encoder_output, encoder_hidden = encoder(source_sentence, encoder_hidden)
        context = encoder_output[source_sentence_length - 1]
        encoder_output = encoder_output.view(source_sentence_length, encoder.hidden_size)
        decoder_input = with_gpu(torch.LongTensor([[EOS]]))
        decoder_hidden = encoder_hidden

        decoded_words = []
        max_length = max(10, 2 * target_sentence_length)
        loss = with_gpu(torch.FloatTensor([0]))
        i = 0
        while True:
            context = context.unsqueeze(0)
            y, context, decoder_hidden = decoder(source_sentence_length, encoder_output, decoder_input, context, decoder_hidden)
            topv, topi = y.topk(1)
            decoder_input = topi
            decoded_word = topi.item()
            if i < target_sentence_length:
                target = target_sentence[i].view(1)
                loss += loss_fn(y, target).item()
            if decoded_word == EOS:
                break
            decoded_words.append(decoded_word)
            if (i + 1) > max_length:
                break
            i += 1

        return with_cpu(loss), decoded_words


def torch2text(language, sentence, EOS_token):
    sentence = sentence.squeeze()
    sentence = with_cpu(sentence)
    sentence = map(lambda idx: language.itos[idx], sentence)
    sentence = filter(lambda word: word != EOS_token, sentence)
    sentence = " ".join(sentence)
    return sentence


def get_text(source_language, target_language, source, target, translation, EOS_token):
    source = torch2text(source_language, source, EOS_token)
    target = torch2text(target_language, target, EOS_token)
    translation = get_sentence(target_language, translation)
    return f"""
    Source: \"{source}\"
    Target: \"{target}\"
    Translation: \"{translation}\"
    """


def get_sentence(language, words):
    return " ".join(map(lambda word: language.itos[word], words))


if __name__ == '__main__':
    main()
