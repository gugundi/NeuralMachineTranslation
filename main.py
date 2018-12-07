from device import select_device, with_cpu, with_gpu
from parse import get_config
from random import sample
from tensorboardX import SummaryWriter
import torchtext
import torch
import torch.nn as nn
from utils import get_or_create_dir, get_text, list2words, torch2words
from visualize import visualize_attention


# TODO: reverse source sentence for better results
# TODO: replace NLLLoss with CrossEntropyLoss (also remove softmax from decoder)
# TODO: fix window start and end problems
# TODO: add simple progress to stdout (e.g. iteration x/n)


def main():
    use_gpu, device, device_idx = select_device()
    print(f'Using device: {device}')
    if use_gpu:
        with torch.cuda.device(device_idx):
            run(use_gpu, device, device_idx)
    else:
        run(use_gpu, device, device_idx)


def run(use_gpu, device, device_idx):
    config = get_config(use_gpu, device, device_idx)
    val_iter = config.get('val_iter')
    val_data = val_iter.data()
    val_dataset = config.get('val_dataset')

    def sample_validation_batches(k):
        return torchtext.data.Batch(sample(val_data, k), val_dataset, device_idx)

    train(config, sample_validation_batches)


def train(config, sample_validation_batches):
    source_language = config.get('src_language')
    target_language = config.get('trg_language')
    EOS_token = config.get('EOS_token')
    PAD_token = config.get('PAD_token')
    SOS_token = config.get('SOS_token')
    train_iter = config.get('train_iter')
    writer_path = config.get('writer_path')
    writer_train_path = get_or_create_dir(writer_path, 'train')
    writer_val_path = get_or_create_dir(writer_path, 'val')
    writer_train = SummaryWriter(log_dir=writer_train_path)
    writer_val = SummaryWriter(log_dir=writer_val_path)
    batch_size = config.get('batch_size')
    epochs = config.get('epochs')
    training = config.get('training')
    eval_every = training.get('eval_every')
    sample_every = training.get('sample_every')
    step = 1
    for epoch in range(epochs):
        for i, training_batch in enumerate(train_iter):
            loss = train_batch(config, training_batch)

            writer_train.add_scalar('loss', loss, step)

            should_evaluate = (i + 1) % eval_every == 0
            should_sample = (i + 1) % sample_every == 0
            if should_evaluate or should_sample:
                val_batch = sample_validation_batches(batch_size)
                val_loss, translation, attention_weights = evaluate_batch(config, val_batch)
                if should_evaluate:
                    writer_val.add_scalar('loss', val_loss, step)

                if should_sample:
                    source_words = torch2words(source_language, val_batch.src[:, 0])
                    target_words = torch2words(target_language, val_batch.trg[:, 0])
                    translation_words = list2words(target_language, translation)
                    attention_figure = visualize_attention(source_words, translation_words, attention_weights)
                    text = get_text(source_words, target_words, translation_words, SOS_token, EOS_token, PAD_token)
                    writer_val.add_figure('attention', attention_figure, step)
                    writer_val.add_text('translation', text, step)

            step += 1


def train_batch(config, batch):
    encoder, decoder = config.get('encoder'), config.get('decoder')
    encoder_optimizer, decoder_optimizer = config.get('encoder_optimizer'), config.get('decoder_optimizer')
    PAD = config.get('PAD')
    SOS = config.get('SOS')
    window_size = config.get('window_size')
    loss_fn = config.get('loss_fn')

    encoder.train()
    decoder.train()

    target_batch = batch.trg
    batch_size = target_batch.size()[1]
    mask, lengths = create_mask(batch)
    encoder_output, encoder_hidden, S, T = encode(encoder, batch)
    encoder_output_padded = pad_with_window_size(encoder_output, window_size, PAD)

    input = with_gpu(torch.LongTensor([[SOS] * batch_size]))
    hidden = encoder_hidden

    losses = with_gpu(torch.empty((batch_size, T), dtype=torch.float))
    for i in range(T):
        y, input, hidden, _ = decode_word(decoder, encoder_output_padded, input, hidden, S, batch_size)
        compute_word_loss(batch_size, losses, i, y, target_batch, loss_fn)
    loss = compute_batch_loss(losses, mask, lengths)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return with_cpu(loss)


def evaluate_batch(config, batch):
    encoder, decoder = config.get('encoder'), config.get('decoder')
    PAD = config.get('PAD')
    window_size = config.get('window_size')
    loss_fn = config.get('loss_fn')

    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        target_batch = batch.trg
        batch_size = target_batch.size()[1]
        mask, lengths = create_mask(batch)
        encoder_output, encoder_hidden, S, T = encode(encoder, batch)
        encoder_output_padded = pad_with_window_size(encoder_output, window_size, PAD)

        input = with_gpu(torch.LongTensor([[SOS] * batch_size]))
        hidden = encoder_hidden

        first_sentence_has_reached_eos = False
        decoded_words = []
        attention_weights = with_gpu(torch.zeros(0, S))
        losses = with_gpu(torch.empty((batch_size, T), dtype=torch.float))
        for i in range(T):
            y, input, hidden, attention = decode_word(decoder, encoder_output_padded, input, hidden, S, batch_size)
            compute_word_loss(batch_size, losses, i, y, target_batch, loss_fn)
            decoded_word = input

            if not first_sentence_has_reached_eos:
                # add attention weights of first sentence in batch
                weights_batch, window_start_batch, window_end_batch = attention
                weights, window_start, window_end = weights_batch[0], window_start_batch[0], window_end_batch[0]
                # make sure that [window_start; window_end] has same length as weights
                max_window_length = weights.shape[1]
                window_length = window_end - window_start + 1
                attention_weights_row = with_gpu(torch.zeros(1, source_sentence_length))
                attention_weights_row[0, window_start:window_end+1] = weights
                attention_weights = torch.cat((attention_weights, attention_weights_row))
                # add decoded word of first sentence in batch
                decoded_word_of_first_sentence = decoded_word[0, 0].item()
                decoded_words.append(decoded_word_of_first_sentence)
                first_sentence_has_reached_eos = decoded_word_of_first_sentence == EOS
        loss = compute_batch_loss(losses, mask, lengths)

        return with_cpu(loss), decoded_words, with_cpu(attention_weights)


def encode(encoder, batch):
    source_batch, _ = batch.src
    target_batch = batch.trg
    batch_size = source_batch.shape[1]
    encoder_hidden = encoder.init_hidden(batch_size)
    S = source_batch.size(0)
    T = target_batch.size(0)
    output, hidden = encoder(source_batch, encoder_hidden)
    return output, hidden, S, T


def decode_word(decoder, encoder_output, decoder_input, decoder_hidden, S, batch_size):
    y, _, decoder_hidden, attention = decoder(S, encoder_output, decoder_input, decoder_hidden, batch_size)
    _, topi = y.topk(1)
    decoder_input = topi.detach().view(1, batch_size)
    y = y.view(batch_size, -1)
    for j in range(batch_size):
        losses[j, ith_word] += loss_fn(y[j], target[j])
    return y, decoder_input, decoder_hidden, attention


def compute_word_loss(batch_size, losses, ith_word, y, target_batch, loss_fn):
    target = target_batch[ith_word]
    for j in range(batch_size):
        losses[j, ith_word] += loss_fn(y[j], target[j])


def pad_with_window_size(batch, window_size, pad):
    size = batch.size()
    n = len(size)
    if n == 2:
        length, batch_size = size
        padded_length = length + 2 * window_size
        padded = with_gpu(torch.empty((padded_length, batch_size)))
        padded[:window_size, :] = pad
        padded[window_size:window_size+length, :] = batch
        padded[-window_size:, :] = pad
    elif n == 3:
        length, batch_size, hidden = size
        padded_length = length + 2 * window_size
        padded = with_gpu(torch.empty((padded_length, batch_size, hidden)))
        padded[:window_size, :, :] = pad
        padded[window_size:window_size+length, :, :] = batch
        padded[-window_size:, :, :] = pad
    else:
        raise Exception(f'Cannot pad batch with {n} dimensions.')
    return padded


def create_mask(batch):
    _, lengths = batch.src
    max_length = lengths.max()
    batch_size = lengths.size()
    mask = with_gpu(torch.ones(max_length, batch_size[0]))
    for i, length in enumerate(lengths):
        mask[length:, i] = 0
    return mask, lengths


def compute_batch_loss(loss, mask, lengths):
    loss = loss * mask
    loss = torch.sum(loss, 1)
    loss = loss / lengths
    return loss


if __name__ == '__main__':
    main()
