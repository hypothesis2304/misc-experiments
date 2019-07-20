import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch


## To use the confusion_matrix function without any errors labels, predictions, num_classes, path should be provided.  


def generate_confusion_matrix(labels, predictions, num_classes, type='diag', print_numbers=False, xticks=None, yticks=None, tick_font=8, path=None):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for l, p in zip(labels.view(-1), predictions.view(-1)):
        confusion_matrix[l.long(), p.long()] += 1
    sns.set(font_scale=1.4)
    confusion_matrix = confusion_matrix.cpu().numpy().astype('float')

    if type == 'diag':
        cm_normalized = confusion_matrix / confusion_matrix.sum(axis=1)[:,np.newaxis]

    elif type == 'no_diag':
        np.fill_diagonal(confusion_matrix, 0.00001)
        mask = np.zeros_like(confusion_matrix, dtype=np.bool)
        mask[np.diag_indices_from(confusion_matrix)] = True
        cm_normalized = confusion_matrix/confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.subplots(figsize=(12,10))
    plot = sns.heatmap(cm_normalized, annot=print_numbers, xticklabels=xticks, yticklabels= yticks, cmap='coolwarm')
    plot.set_title('confusion_matrix')
    plot.xaxis.set_ticklabels(plot.xaxis.get_ticklabels(), fontsize=str(tick_font))
    plot.yaxis.set_ticklabels(plot.yaxis.get_ticklabels(), fontsize=str(tick_font))
    plt.savefig(path, format='png',bbox_inches='tight')
    return
