from device import select_device, with_cpu, with_gpu
from parse import get_config
from random import sample
from tensorboardX import SummaryWriter
import torchtext
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from utils import get_bleu, get_or_create_dir, get_text, list2words, torch2words
from visualize import visualize_attention


# TODO: teacher forcing?
# TODO: look in to using torch pad function instead of pad_with_window_size
# TODO: save and load weights


def main():
    use_gpu, device, device_idx = select_device()
    if use_gpu:
        device_name = torch.cuda.get_device_name(device_idx)
        print(f'Using device: {device} ({device_name})')
        with torch.cuda.device(device_idx):
            run(use_gpu, device, device_idx)
    else:
        print(f'Using device: cpu')
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
        print(f'Epoch: {epoch+1}/{epochs}')
        for i, training_batch in enumerate(train_iter):
            loss = train_batch(config, training_batch)

            writer_train.add_scalar('loss', loss, step)

            should_evaluate = (i + 1) % eval_every == 0
            should_sample = (i + 1) % sample_every == 0
            if should_evaluate or should_sample:
                val_batch = sample_validation_batches(batch_size)
                val_batch_trg, _ = val_batch.trg
                val_loss, translations, attention_weights = evaluate_batch(config, val_batch)
                if should_evaluate:
                    bleu = get_bleu(source_language, target_language, val_batch_trg, batch_size, translations, PAD_token)
                    writer_val.add_scalar('bleu', bleu, step)
                    writer_val.add_scalar('loss', val_loss, step)
                if should_sample:
                    val_batch_src, val_lengths_src = val_batch.src
                    s0 = val_lengths_src[0].item()
                    source_words = torch2words(source_language, val_batch_src[:, 0])
                    target_words = torch2words(target_language, val_batch_trg[:, 0])
                    translation_words = list(filter(lambda word: word != PAD_token, list2words(target_language, translations[0])))
                    if sum(attention_weights.shape) != 0:
                        attention_figure = visualize_attention(source_words[:s0], translation_words, attention_weights)
                        writer_val.add_figure('attention', attention_figure, step)
                    text = get_text(source_words, target_words, translation_words, SOS_token, EOS_token, PAD_token)
                    writer_val.add_text('translation', text, step)

            step += 1


def train_batch(config, batch):
    encoder, decoder = config.get('encoder'), config.get('decoder')
    encoder_optimizer, decoder_optimizer = config.get('encoder_optimizer'), config.get('decoder_optimizer')
    PAD = config.get('PAD_src')
    SOS = config.get('SOS')
    window_size = config.get('window_size')
    loss_fn = config.get('loss_fn')

    encoder.train()
    decoder.train()

    _, source_lengths = batch.src
    target_batch, target_lengths = batch.trg
    _, batch_size = target_batch.size()
    mask = create_mask(batch.trg)
    encoder_output, encoder_hidden, S, T = encode(encoder, batch, window_size, PAD)
    hidden = encoder_hidden

    y, _, _, _ = decoder(encoder_output, target_batch, hidden, source_lengths)
    loss = loss_fn(y, target_batch)
    loss = compute_batch_loss(loss, mask, target_lengths)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(encoder.parameters(), 1)
    clip_grad_norm_(decoder.parameters(), 1)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return with_cpu(loss)


def evaluate_batch(config, batch):
    encoder, decoder = config.get('encoder'), config.get('decoder')
    EOS = config.get('EOS')
    PAD_src = config.get('PAD_src')
    PAD_trg = config.get('PAD_trg')
    SOS = config.get('SOS')
    window_size = config.get('attention').get('window_size')
    loss_fn = config.get('loss_fn')

    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        _, source_lengths = batch.src
        target_batch, target_lengths = batch.trg
        batch_size = target_batch.size()[1]
        mask = create_mask(batch.trg)
        encoder_output, encoder_hidden, S, T = encode(encoder, batch, window_size, PAD_src)

        input = with_gpu(torch.LongTensor([[SOS] * batch_size]))
        hidden = encoder_hidden

        first_sentence_has_reached_end = False
        translations = [[] for _ in range(batch_size)]
        attention_weights = with_gpu(torch.zeros(0, source_lengths[0]))
        losses = with_gpu(torch.empty((T, batch_size), dtype=torch.float))
        for i in range(T):
            y, input, hidden, attention = decode_word(decoder, encoder_output, input, hidden, source_lengths)
            compute_word_loss(losses, i, y, target_batch, loss_fn)

            for j in range(batch_size):
                translations[j].append(input[0, j].item())

            # don't add padding to attention visualiation
            decoded_word = translations[0][i]
            if not first_sentence_has_reached_end and decoded_word != PAD_trg:
                # add attention weights of first sentence in batch
                attention_weights = torch.cat((attention_weights, attention))
                first_sentence_has_reached_end = decoded_word == EOS
        loss = compute_batch_loss(losses, mask, target_lengths)

        return with_cpu(loss), translations, with_cpu(attention_weights)


def encode(encoder, batch, window_size, PAD):
    source_batch, _ = batch.src
    target_batch, _ = batch.trg
    batch_size = source_batch.shape[1]
    hidden = encoder.init_hidden(batch_size)
    S = source_batch.size(0)
    T = target_batch.size(0)
    input = pad_with_window_size(source_batch, window_size, PAD)
    output, hidden = encoder(input, hidden)
    return output, hidden, S, T


def decode_word(decoder, encoder_output, input, context, hidden, batch_size, lengths):
    y, _, hidden, attention = decoder(encoder_output, input, hidden, lengths)
    _, topi = y.topk(1)
    input = topi.detach().view(1, batch_size)
    return y, input, hidden, attention


def pad_with_window_size(batch, window_size, pad):
    size = batch.size()
    n = len(size)
    if n == 2:
        length, batch_size = size
        padded_length = length + (2 * window_size + 1)
        padded = with_gpu(torch.empty((padded_length, batch_size), dtype=torch.long))
        padded[:window_size, :] = pad
        padded[window_size:window_size+length, :] = batch
        padded[-(window_size+1):, :] = pad
    elif n == 3:
        length, batch_size, hidden = size
        padded_length = length + (2 * window_size + 1)
        padded = with_gpu(torch.empty((padded_length, batch_size, hidden), dtype=torch.long))
        padded[:window_size, :, :] = pad
        padded[window_size:window_size+length, :, :] = batch
        padded[-(window_size+1):, :, :] = pad
    else:
        raise Exception(f'Cannot pad batch with {n} dimensions.')
    return padded


def create_mask(batch_tuple):
    batch, lengths = batch_tuple
    max_length, batch_size = batch.shape
    mask = with_gpu(torch.ones(max_length, batch_size))
    for i, length in enumerate(lengths):
        mask[length:, i] = 0
    return mask


def compute_word_loss(losses, ith_word, y, target_batch, loss_fn):
    target = target_batch[ith_word]
    losses[ith_word] = loss_fn(y, target)


def compute_batch_loss(loss, mask, lengths):
    loss = loss * mask
    loss = torch.sum(loss, 0, dtype=torch.float)
    loss = loss / lengths.float()
    loss = loss.mean()
    return loss


if __name__ == '__main__':
    main()
