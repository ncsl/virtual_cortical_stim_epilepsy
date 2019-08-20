from inspect import signature

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from sklearn.metrics import confusion_matrix

from cortstim.base.utils.data_structures_utils import ensure_list


def plot_confusion_matrix(ax, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_aspect('auto')
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #            xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax


def plot_boxplot_withdf(ax, outcome_df, df_xlabel, df_ylabel, ylabel="",
                        titlestr="", ylim=None, yticks=None, color=None):
    ax = sns.violinplot(x=df_xlabel, y=df_ylabel, color=color,
                        data=outcome_df, ax=ax,
                        )

    if color == 'black':
        swarmcolor = 'white'
    else:
        swarmcolor = 'black'
    ax = sns.swarmplot(x=df_xlabel, y=df_ylabel,
                       data=outcome_df, ax=ax, color=swarmcolor)
    ax.set_title(titlestr)
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    return ax


def plot_boxplot(ax, boxdict, titlestr, ylabel, textpoints=[]):
    if isinstance(boxdict, list):
        ax.boxplot(boxdict[0], labels=boxdict[1])
        vals = boxdict[0]
    elif isinstance(boxdict, dict):
        ax.boxplot(boxdict.values(), labels=boxdict.keys())
        vals = list(boxdict.values())
    else:
        raise AttributeError("Boxdict needs to be either dict or list.")

    ax.set_title(titlestr)
    ax.set_ylabel(ylabel)

    ngroup = len(vals)
    clevels = np.linspace(0., 1., ngroup)

    for i, (val, clevel) in enumerate(zip(vals, clevels)):
        val = ensure_list(val)
        x = np.random.normal(i + 1, 0.04, len(val))

        # print(x.shape, len(val), clevel)
        s = np.random.rand(*x.shape) * 200 + 100
        s = 250
        ax.scatter(x, val,
                   s=s,
                   c=cm.jet(np.array([clevel, ])),
                   marker="o", alpha=0.9)

        if textpoints:
            for i, txt in enumerate(textpoints[i]):
                ax.annotate(txt, (x[i], val[i]))


def plot_roc(ax, fpr, tpr, label, titlestr):
    ax.plot(fpr, tpr, lw=1, label=label)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_title(titlestr)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0, 1.01])


def plot_pr(ax, recall, precision, label, titlestr):
    ax.plot(recall, precision, lw=1, label=label)
    # set kwargs to fill in pr-curve
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    ax.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.set_title(titlestr)


def plot_baseline(ax, baselinex, baseliney):
    # np.mean(baselinex).squeeze()
    # np.mean(baseliney).squeeze()
    ax.plot(baselinex, baseliney, '--', label='clinical-baseline')


def plot_roc_inverse(ax, fnr, tnr, label, titlestr):
    ax.plot(fnr, tnr, lw=1, label=label)
    ax.set_ylabel("True Negative Rate")
    ax.set_xlabel("False Negative Rate")
    ax.set_title(titlestr)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0, 1.01])

# def plot_confusion_matrix(ax, y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     #     classes = classes[unique_labels(y_true, y_pred)]
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     #     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            #            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     return ax
