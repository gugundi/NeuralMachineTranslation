from bleu import compute_bleu
from device import select_device, with_cpu, with_gpu
import json
from parse import get_config
from random import sample
from tensorboardX import SummaryWriter
import torchtext
import torch
from torch.nn.utils import clip_grad_norm_
from utils import filter_words, get_or_create_dir, get_text, list2words, torch2words
from visualize import visualize_attention


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
        return torchtext.data.Batch(sample(val_data, k), val_dataset, device)

    # save source and target language vocabularies
    source_language = config.get('src_language')
    target_language = config.get('trg_language')
    data = {
        "source": {
            "itos": source_language.itos,
            "stoi": source_language.stoi,
        },
        "target": {
            "itos": target_language.itos,
            "stoi": target_language.stoi,
        },
    }
    weights_path = config.get('weights_path')
    with open(f'{weights_path}/language.json', 'w') as f:
        json.dump(data, f)

    train(config, sample_validation_batches)


def train(config, sample_validation_batches):
    source_language = config.get('src_language')
    target_language = config.get('trg_language')
    EOS_token = config.get('EOS_token')
    PAD_token = config.get('PAD_token')
    SOS_token = config.get('SOS_token')
    train_iter = config.get('train_iter')
    val_iter = config.get('val_iter')
    writer_path = config.get('writer_path')
    writer_train_path = get_or_create_dir(writer_path, 'train')
    writer_val_path = get_or_create_dir(writer_path, 'val')
    writer_train = SummaryWriter(log_dir=writer_train_path)
    writer_val = SummaryWriter(log_dir=writer_val_path)
    epochs = config.get('epochs')
    training = config.get('training')
    eval_every = training.get('eval_every')
    sample_every = training.get('sample_every')
    use_attention = config.get('use_attention')
    step = 1
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}/{epochs}')
        save_weights(config)
        for i, training_batch in enumerate(train_iter):
            loss = train_batch(config, training_batch)
            writer_train.add_scalar('loss', loss, step)

            if step % eval_every == 0:
                val_lengths = 0
                val_losses = 0
                reference_corpus = []
                translation_corpus = []
                for val_batch in val_iter:
                    val_loss, translations = evaluate_batch(config, val_batch)
                    val_lengths += 1
                    val_losses += val_loss
                    val_batch_trg, _ = val_batch.trg
                    _, batch_size = val_batch_trg.shape
                    references = map(lambda i: torch2words(target_language, val_batch_trg[:, i]), range(batch_size))
                    references = map(lambda words: [list(filter_words(words, SOS_token, EOS_token, PAD_token))], references)
                    reference_corpus.extend(references)
                    translations = map(lambda translation: list2words(target_language, translation), translations)
                    translations = map(lambda words: list(filter_words(words, SOS_token, EOS_token, PAD_token)), translations)
                    translation_corpus.extend(translations)
                bleu = compute_bleu(reference_corpus, translation_corpus)
                val_loss = val_losses / val_lengths
                writer_val.add_scalar('bleu', bleu, step)
                writer_val.add_scalar('loss', val_loss, step)

            if step % sample_every == 0:
                val_batch = sample_validation_batches(1)
                val_batch_src, val_lengths_src = val_batch.src
                val_batch_trg, _ = val_batch.trg
                s0 = val_lengths_src[0].item()
                _, translations, attention_weights = evaluate_batch(config, val_batch, True)
                source_words = torch2words(source_language, val_batch_src[:, 0])
                target_words = torch2words(target_language, val_batch_trg[:, 0])
                translation_words = list(filter(lambda word: word != PAD_token, list2words(target_language, translations[0])))
                if use_attention and sum(attention_weights.shape) != 0:
                    attention_figure = visualize_attention(source_words[:s0], translation_words, attention_weights)
                    writer_val.add_figure('attention', attention_figure, step)
                text = get_text(source_words, target_words, translation_words, SOS_token, EOS_token, PAD_token)
                writer_val.add_text('translation', text, step)

            step += 1

    save_weights(config)


def train_batch(config, batch):
    model = config.get('model')
    optimizer = config.get('optimizer')
    gradient_clipping = config.get('gradient_clipping')

    model.train()
    ys = model(batch)
    loss = get_loss(config, batch, ys)

    optimizer.zero_grad()
    loss.backward()
    if gradient_clipping:
        clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    return with_cpu(loss)


def evaluate_batch(config, batch, sample=False):
    model = config.get('model')

    with torch.no_grad():
        model.eval()
        if sample:
            ys, translations, attention_weights = model(batch, training=False, sample=True)
            loss = get_loss(config, batch, ys)
            return with_cpu(loss), translations, with_cpu(attention_weights)
        else:
            ys, translations = model(batch, training=False, sample=False)
            loss = get_loss(config, batch, ys)
            return with_cpu(loss), translations


def create_mask(batch_tuple):
    batch, lengths = batch_tuple
    max_length, batch_size = batch.shape
    mask = with_gpu(torch.ones(max_length, batch_size))
    for i, length in enumerate(lengths):
        mask[length:, i] = 0
    return mask


def compute_batch_loss(loss, mask, lengths):
    loss = loss * mask
    loss = torch.sum(loss, 0, dtype=torch.float)
    loss = loss / lengths.float()
    loss = loss.mean()
    return loss


def get_loss(config, batch, ys):
    loss_fn = config.get('loss_fn')
    mask = create_mask(batch.trg)
    target_batch, target_lengths = batch.trg
    T, batch_size = target_batch.shape
    losses = with_gpu(torch.empty((T, batch_size), dtype=torch.float))
    for i in range(T):
        losses[i] = loss_fn(ys[i], target_batch[i])
    loss = compute_batch_loss(losses, mask, target_lengths)
    return loss


def save_weights(config):
    weights_path = config.get("weights_path")
    model_path = f'{weights_path}/model'
    model = config.get('model')
    model_weights = model.state_dict()
    torch.save(model_weights, model_path)


if __name__ == '__main__':
    main()
