import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def visualize_attention(source_words, translation_words, attention_weights):
    attention_weights = attention_weights.numpy()
    n = len(source_words)
    m = len(translation_words)
    fig, ax = plt.subplots(figsize=(n, m))
    ax.imshow(attention_weights, cmap='gray')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    ax.set_xticklabels(source_words)
    ax.set_yticklabels(translation_words)
    ax.set_xlabel('Source')
    ax.set_ylabel('Translation')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    return fig
