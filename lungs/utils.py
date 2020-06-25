# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import numpy as np
import gdown
from sklearn.metrics import roc_auc_score, roc_curve
import scikitplot as skplt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from os.path import join
import seaborn as sns
sns.set_style("darkgrid")

REMOTE_CKPTS = {
    'cancer_fine_tuned': {'url': '1Zc8KdEz9JUfkT1ZsG9ELYReUPbVapbQC', 'md5': 'cd5271617e090859f73a727da81cc2e3'},
    'i3d_imagenet': {'url': '1FMWHGFYPjuvpgzkGm-_gYKdXpmv5fOq2',  'md5': 'f1408b50e5871153516fe87932121745'}
}

def load_pretrained_ckpt(ckpt, data_dir):
    if ckpt in REMOTE_CKPTS:
        download_ckpt(data_dir, ckpt, REMOTE_CKPTS[ckpt])

    # Load a pre-defined ckpt or a ckpt from path
    predefined = join(data_dir, 'checkpoints', ckpt)
    ckpt_dir = predefined if os.path.exists(predefined) else ckpt

    pre_trained_ckpt = join(ckpt_dir, 'model.ckpt')
    print('\nINFO: Loading pre-trained model:', pre_trained_ckpt)
    return pre_trained_ckpt

def download_ckpt(data_dir, name, download_info):
    if os.path.exists(join(data_dir, 'checkpoints', name)):
        print('\nINFO: {} model already downloaded.'.format(name))
    else:
        print('\nINFO: Downloading {} model...'.format(name))
        url = 'https://drive.google.com/uc?id=' + download_info['url']
        zip_output = join(data_dir, 'checkpoints', name + '.zip')
        md5 = download_info['md5']
        gdown.cached_download(url, zip_output, md5=md5, postprocess=gdown.extractall, quiet=True)
        os.remove(zip_output)

def pretty_print_floats(lst):
    return ',  '.join(['{:.3f}'.format(_) for _ in lst])

def load_npz_as_list(base_dir, npz_file):
    return np.load(join(base_dir, npz_file))['arr_0'].tolist()

def plot_loss(val_loss, tr_loss, plots_dir):
    figure(num=None, figsize=(16, 8), dpi=100)
    title = 'Training and Validation Loss'
    epochs = range(1, len(val_loss) + 1)
    plt.plot(epochs, val_loss, label='Val. Loss')
    plt.plot(epochs, tr_loss, label='Train Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(join(plots_dir, title + '.png'), bbox_inches='tight')

def plot_acc_auc(val_acc, tr_acc, val_auc, tr_auc, plots_dir):
    figure(num=None, figsize=(16, 8), dpi=100)
    title = 'Accuracy and AUC'
    epochs = range(1, len(val_acc) + 1)
    plt.plot(epochs, val_acc, label='Val. Accuracy')
    plt.plot(epochs, tr_acc, label='Train Accuracy')
    plt.plot(epochs, val_auc, label='Val. AUC')
    plt.plot(epochs, tr_auc, label='Train AUC')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(join(plots_dir, title + '.png'), bbox_inches='tight')

def calc_plot_epoch_auc_roc(y, y_probs, title, plots_dir, verbose=False):
    y_prob_2_classes = [(1 - p, p) for p in y_probs]
    fpr, tpr, th = roc_curve(y, y_probs)
    if verbose:
        print('TPR:', pretty_print_floats(tpr))
        print('FPR:', pretty_print_floats(fpr))
        print('TH: ', pretty_print_floats(th), '\n')
    auc = roc_auc_score(y, y_probs)
    title = title + ',  AUC={:.3f}'.format(auc)
    skplt.metrics.plot_roc(y, y_prob_2_classes, classes_to_plot=[], 
                           title= title,
                           figsize=(7, 7), plot_micro=False, plot_macro=True, 
                           title_fontsize=15, text_fontsize=13)
    plt.show()
    plt.savefig(join(plots_dir, title) + '.png', bbox_inches='tight')

def load_and_plot_epoch_auc(metrics_dir, epoch, val_true, tr_true, plots_dir):
    val_preds_epoch = load_npz_as_list(metrics_dir, 'val_preds/epoch_' + str(epoch) + '.npz')
    calc_plot_epoch_auc_roc(val_true, val_preds_epoch, 
                            'Val. ROC for Epoch {}'.format(epoch), plots_dir)

    tr_preds_epoch = load_npz_as_list(metrics_dir, 'tr_preds/epoch_' + str(epoch) + '.npz')
    calc_plot_epoch_auc_roc(tr_true, tr_preds_epoch, 
                            'Train ROC for Epoch {}'.format(epoch), plots_dir)

def plot_metrics(epoch, metrics_dir, plots_dir):
    val_loss = load_npz_as_list(metrics_dir, 'val_loss.npz')
    val_acc = load_npz_as_list(metrics_dir, 'val_acc.npz')
    val_auc = load_npz_as_list(metrics_dir, 'val_auc.npz')
    val_true = load_npz_as_list(metrics_dir, 'val_true.npz')

    tr_loss = load_npz_as_list(metrics_dir, 'tr_loss.npz')
    tr_acc = load_npz_as_list(metrics_dir, 'tr_acc.npz')
    tr_auc = load_npz_as_list(metrics_dir, 'tr_auc.npz')
    tr_true = load_npz_as_list(metrics_dir, 'tr_true.npz')

    plot_loss(val_loss, tr_loss, plots_dir)
    plot_acc_auc(val_acc, tr_acc, val_auc, tr_auc, plots_dir)
    load_and_plot_epoch_auc(metrics_dir, epoch, val_true, tr_true, plots_dir)

def write_metrics(metrics, tr_metrics, val_metrics, metrics_dir, epoch, verbose=False):
    for (loss, acc, auc, preds, _), ds in ((tr_metrics, 'tr'), (val_metrics, 'val')):
        for metric, key in [(loss, 'loss'), (acc, 'acc'), (auc, 'auc'), (preds, 'preds')]:
            name = ds + '_' + key
            metrics[name].append(metric)
            write_number_list(metrics[name], join(metrics_dir, name))
        write_number_list(preds, join(metrics_dir, ds + '_preds', 'epoch_{}'.format(epoch)), verbose=verbose)

def apply_window(volume, axis=4):
    # Windowing
    # Our values currently range from -1024 to around 2000. 
    # Anything above 400 is not interesting to us, as these are simply bones with different radiodensity.  
    # A commonly used set of thresholds in Lungs LDCT to normalize between are -1000 and 400. 
    min_bound = -1000.0
    max_bound = 400.0
    volume = (volume - min_bound) / (max_bound - min_bound)
    volume[volume>1] = 1.
    volume[volume<0] = 0.

    # Normalize rgb values to [-1, 1]
    volume = (volume * 2) - 1
    res =  np.stack((volume, volume, volume), axis=axis)
    return res.astype(np.float32)

def write_number_list(lst, f_name, verbose=False):
    if verbose:
        print('INFO: Saving ' + f_name + '.npz ...')
        print(lst)
    np.savez(f_name + '.npz', np.array(lst))       

def batcher(iterable, batch_size=1):
    iter_len = len(iterable)
    for i in range(0, iter_len, batch_size):
        yield iterable[i: min(i + batch_size, iter_len)]

def load_data_list(path):
    coupled_data = []
    with open(path) as file_list_fp:
        for line in file_list_fp:
            volume_path, label = line.split()
            coupled_data.append((volume_path, int(label)))
    return coupled_data

def get_list_labels(coupled_data):
        return np.array([l for _, l in coupled_data]).astype(np.int64)

def placeholder_inputs(num_slices, crop_size, rgb_channels=3):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    num_slices: The num of slices per volume.
    crop_size: The crop size of per volume.
    channels: The number of RGB input channels per volume.

    Returns:
    volumes_placeholder: volumes placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # volume and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    volumes_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           num_slices,
                                                           crop_size,
                                                           crop_size,
                                                           rgb_channels))
    labels_placeholder = tf.placeholder(tf.int64, shape=(None))
    is_training = tf.placeholder(tf.bool)
    return volumes_placeholder, labels_placeholder, is_training

def focal_loss(logits, labels, alpha=0.75, gamma=2):
    """Compute focal loss for binary classification.

    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `labels`.
    """
    y_pred = tf.nn.sigmoid(logits)
    labels = tf.to_float(labels)
    losses = -(labels * (1 - alpha) * ((1 - y_pred) * gamma) * tf.log(y_pred)) - \
        (1 - labels) * alpha * (y_pred ** gamma) * tf.log(1 - y_pred)
    return tf.reduce_mean(losses)

def cross_entropy_loss(logits, labels):
    # pylint: disable=no-value-for-parameter
    # pylint: disable=unexpected-keyword-arg
    cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                  )
    return cross_entropy_mean

def accuracy(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def get_preds(preds):
    return preds[:, 1]

def get_logits(logits):
    return logits
