import matplotlib.pyplot as plt
import numpy as np


def visualize_attention(source_words, translation_words, attention_weights):
    attention_weights = attention_weights.numpy()
    n = len(source_words)
    m = len(translation_words)
    fig, ax = plt.subplots(figsize=(m, n))
    ax.imshow(attention_weights, cmap='gray')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    ax.set_xticklabels(source_words)
    ax.set_yticklabels(translation_words)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(m):
        for j in range(n):
            weight = f'{attention_weights[i, j]:0.02f}'
            ax.text(j, i, weight, ha='center', va='center', color='r', fontsize=12)
    fig.tight_layout()
    return fig
