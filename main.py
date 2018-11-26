from device import select_device, with_cpu, with_gpu
from parse import get_config
from random import sample
from tensorboardX import SummaryWriter
import torchtext
import torch
from utils import get_or_create_dir, get_text, list2words, torch2words
from visualize import visualize_attention


# TODO: reverse source sentence for better results


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
    EOS = target_language.stoi[EOS_token]
    SOS_token = config.get('SOS_token')
    SOS = target_language.stoi[SOS_token]
    train_iter = config.get('train_iter')
    writer_path = config.get('writer_path')
    writer_train_path = get_or_create_dir(writer_path, 'train')
    writer_val_path = get_or_create_dir(writer_path, 'val')
    writer_train = SummaryWriter(log_dir=writer_train_path)
    writer_val = SummaryWriter(log_dir=writer_val_path)
    batch_size = config.get('batch_size')
    epochs = config.get('epochs')
    loss_fn = config.get('loss_fn')
    decoder = config['decoder']
    encoder = config['encoder']
    decoder_optimizer = config['decoder_optimizer']
    encoder_optimizer = config['encoder_optimizer']
    training = config.get('training')
    eval_every = training.get('eval_every')
    sample_every = training.get('sample_every')
    step = 1
    for epoch in range(epochs):
        for i, training_batch in enumerate(train_iter):
            loss = train_batch(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn, SOS, EOS, training_batch, batch_size)

            writer_train.add_scalar('loss', loss, step)

            if (i + 1) % eval_every == 0:
                val_batch = sample_validation_batches(batch_size)
                val_loss, translation, attention_weights = evaluate_batch(encoder, decoder, loss_fn, SOS, EOS, val_batch, batch_size)
                writer_val.add_scalar('loss', val_loss, step)

                if (i + 1) % sample_every == 0:
                    source_words = torch2words(source_language, val_batch.src[:, 0])
                    target_words = torch2words(target_language, val_batch.trg[:, 0])
                    translation_words = list2words(target_language, translation[0])
                    attention_figure = visualize_attention(source_words, translation_words, attention_weights[0])
                    text = get_text(source_words, target_words, translation_words, SOS_token, EOS_token)
                    writer_val.add_figure('attention', attention_figure, step)
                    writer_val.add_text('translation', text, step)

            step += 1


def train_batch(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn, SOS, EOS, batch, batch_size):
    encoder.train()
    decoder.train()

    source_batch = batch.src
    # ignore first part of target sentence since we are not interested in SOS
    target_batch = batch.trg[1:]
    encoder_hidden = encoder.init_hidden()
    source_sentence_length = source_batch.size(0)
    target_sentence_length = target_batch.size(0)

    encoder_output, encoder_hidden = encoder(source_batch, encoder_hidden)
    decoder_input = with_gpu(torch.LongTensor([[SOS] * batch_size]))
    decoder_hidden = encoder_hidden

    loss = with_gpu(torch.FloatTensor([0]))
    for i in range(target_sentence_length):
        y, _, decoder_hidden, _ = decoder(source_sentence_length, encoder_output, decoder_input, decoder_hidden)
        topv, topi = y.topk(1)
        decoder_input = topi.detach().view(1, batch_size)
        target = target_batch[i]
        y = y.view(batch_size, -1)
        loss += loss_fn(y, target)
        has_reached_eos = all(map(lambda word: word.item() == EOS, decoder_input.squeeze()))
        if has_reached_eos:
            break

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return with_cpu(loss)


def evaluate_batch(encoder, decoder, loss_fn, SOS, EOS, batch, batch_size):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        source_batch = batch.src
        # ignore first part of target sentence since we are not interested in SOS
        target_batch = batch.trg[1:]
        encoder_hidden = encoder.init_hidden()
        source_sentence_length = source_batch.size(0)
        target_sentence_length = target_batch.size(0)

        encoder_output, encoder_hidden = encoder(source_batch, encoder_hidden)
        decoder_input = with_gpu(torch.LongTensor([[SOS] * batch_size]))
        decoder_hidden = encoder_hidden

        decoded_words = [[] * batch_size]
        max_length = target_sentence_length + 5
        attention_weights = with_gpu(torch.zeros(batch_size, max_length, source_sentence_length))
        loss = with_gpu(torch.FloatTensor([0]))
        i = 0
        while True:
            y, _, decoder_hidden, attention = decoder(source_sentence_length, encoder_output, decoder_input, decoder_hidden)
            topv, topi = y.topk(1)
            decoder_input = topi.view(1, batch_size)
            decoded_word = decoder_input
            y = y.view(batch_size, -1)
            if i < target_sentence_length:
                target = target_batch[i]
                loss += loss_fn(y, target).item()
            weights_batch, window_start_batch, window_end_batch = attention
            for j, (weights, window_start, window_end) in enumerate(zip(weights_batch, window_start_batch, window_end_batch)):
                attention_weights[j, i, window_start:window_end+1] = weights
            for j, words in enumerate(decoded_words):
                words.append(decoded_word[0, j].item())
            has_reached_eos = all(map(lambda word: word.item() == EOS, decoded_word.squeeze()))
            if has_reached_eos or (i + 1) == max_length:
                break
            i += 1

        return with_cpu(loss), decoded_words, with_cpu(attention_weights)


if __name__ == '__main__':
    main()
