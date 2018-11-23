from data_loader import load_iwslt
import json
import os
from parse import parse_config
from random import random
import sys
from tensorboardX import SummaryWriter
from time import time
from torch.nn.utils.rnn import pad_sequence
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
    main_path = os.path.dirname(os.path.realpath(__file__))
    name = parsed_config.get('name')
    writer_path = get_or_create_dir(main_path, f'.logs/{name}')
    SOS_token = '<SOS>'
    EOS_token = '<EOS>'
    train_iter, val_iter, source_language, target_language = load_iwslt(parsed_config, SOS_token, EOS_token)
    parsed_config['source_vocabulary_size'] = len(source_language.itos)
    parsed_config['target_vocabulary_size'] = len(target_language.itos)
    train(train_iter, val_iter, source_language, target_language, SOS_token, EOS_token, writer_path, parsed_config)


def train(train_iter, val_iter, source_language, target_language, SOS_token, EOS_token, writer_path, parsed_config):
    SOS = target_language.stoi[SOS_token]
    EOS = target_language.stoi[EOS_token]
    writer_train_path = get_or_create_dir(writer_path, 'train')
    writer_val_path = get_or_create_dir(writer_path, 'val')
    writer_train = SummaryWriter(log_dir=writer_train_path)
    writer_val = SummaryWriter(log_dir=writer_val_path)
    batch_size = parsed_config.get('batch_size')
    epochs = parsed_config.get('epochs')
    loss_fn = parsed_config.get('loss_fn')
    encoder, decoder = parsed_config['model']
    encoder_optimizer, decoder_optimizer = parsed_config['optimizer']
    training = parsed_config.get('training')
    eval_every = training.get('eval_every')
    sample_every = training.get('sample_every')
    step = 1
    for epoch in range(epochs):
        train_iter.init_epoch()
        val_iter.init_epoch()

        for i, train_batch in enumerate(train_iter):
            # TODO: Use torch.pad_sequence
            print(train_batch.size())
            pad_sequence(train_batch, batch_first = True)
            train_loss = train_sentence_batch(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn, SOS, EOS, train_batch)

            timestamp = time()
            writer_train.add_scalar('loss', train_loss, step, timestamp)

            if (i + 1) % eval_every == 0:
                val_losses = 0
                val_lengths = 0
                for val_batch in val_iter:
                    val_loss, _ = evaluate_sentence_pair(encoder, decoder, loss_fn, SOS, EOS, val_batch)
                    val_losses += val_loss
                    # Is this supposed to be batch_size?
                    # val_lengths += batch_size
                    val_lengths += 1
                val_loss = val_losses / val_lengths
                writer_val.add_scalar('loss', val_loss, step, timestamp)

            if (i + 1) % sample_every == 0:
                for j, val_batch in enumerate(val_iter):
                    if j == 5:
                        break
                    _, translation = evaluate_sentence_pair(encoder, decoder, loss_fn, SOS, EOS, val_batch)
                    text = get_text(source_language, target_language, val_batch.src, val_batch.trg, translation, SOS_token, EOS_token)
                    writer_val.add_text('translation', text, step, timestamp)

            step += 1


def train_sentence_batch(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn, SOS, EOS, batch):
    # TODO: Update train methods
    encoder.train()
    decoder.train()

    # TODO: everything in pair should come from batch
    source_batch = batch.src
    target_batch = batch.trg
    encoder_hidden = encoder.init_hidden()
    source_batch_length = source_batch.size(1)
    print('Sorce_batch size: ', source_batch.size())
    print('Sorce_batch_length length: ', source_batch_length)
    source_hiddens = torch.zeros(source_batch_length, encoder.hidden_size)
    target_sentence_length = target_batch.size(1)

    # Run encoder and get output and last hidden state
    for i in range(source_batch_length):
        encoder_output, encoder_hidden = encoder(source_batch[i], encoder_hidden)
        print('Encoder_output size: ', encoder_output.size())
        print('Encoder_hidden size: ', encoder_hidden.size())
        source_hiddens[i] = encoder_output[0, 0]

    decoder_input = torch.LongTensor([[SOS]])
    decoder_hidden = encoder_hidden
    context = encoder_hidden[0]

    loss = 0
    for i in range(target_sentence_length):
        y, context, decoder_hidden = decoder(source_sentence_length, source_hiddens, decoder_input, context, decoder_hidden)
        context = context.unsqueeze(0)
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

    return loss


def evaluate_sentence_pair(encoder, decoder, loss_fn, SOS, EOS, pair):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        source_sentence = pair.src
        target_sentence = pair.trg
        encoder_hidden = encoder.init_hidden()
        source_sentence_length = source_sentence.size(0)
        source_hiddens = torch.zeros(source_sentence_length, encoder.hidden_size)
        target_sentence_length = target_sentence.size(0)

        for i in range(source_sentence_length):
            encoder_output, encoder_hidden = encoder(source_sentence[i], encoder_hidden)
            source_hiddens[i] = encoder_output[0, 0]

        decoded_words = []
        decoder_input = torch.LongTensor([[SOS]])
        decoder_hidden = encoder_hidden
        context = encoder_hidden[0]

        max_length = max(10, 2 * target_sentence_length)
        loss = 0
        i = 0
        while True:
            y, context, decoder_hidden = decoder(source_sentence_length, source_hiddens, decoder_input, context, decoder_hidden)
            context = context.unsqueeze(0)
            topv, topi = y.topk(1)
            decoder_input = topi
            decoded_word = topi.item()
            if i < target_sentence_length:
                target = target_sentence[i].view(1)
                loss += loss_fn(y, target)
            if decoded_word == EOS:
                break
            decoded_words.append(decoded_word)
            if (i + 1) > max_length:
                break
            i += 1

        return loss, decoded_words


def torch2text(language, sentence, SOS_token, EOS_token):
    sentence = sentence.squeeze()
    sentence = map(lambda idx: language.itos[idx], sentence)
    sentence = filter(lambda word: word != SOS_token and word != EOS_token, sentence)
    sentence = " ".join(sentence)
    return sentence


def get_text(source_language, target_language, source, target, translation, SOS_token, EOS_token):
    source = torch2text(source_language, source, EOS_token, SOS_token)
    target = torch2text(target_language, target, EOS_token, SOS_token)
    translation = get_sentence(target_language, translation)
    return f"""
    Source: \"{source}\"'
    Target: \"{target}\"
    Translation: \"{translation}\"
    """


def get_sentence(language, words):
    return " ".join(map(lambda word: language.itos[word], words))


def get_or_create_dir(base_path, dir_name):
    out_directory = os.path.join(base_path, dir_name)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    return out_directory


if __name__ == '__main__':
    main()
