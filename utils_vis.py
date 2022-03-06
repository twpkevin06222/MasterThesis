import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn import decomposition
from matplotlib import cm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_loss(loss_list, xlabel, ylabel, title):
    '''
    :param loss_list: List containing total loss values
    :param recon_list: List containing reconstruction loss
    :param xlabel: string for xlabel
    :param ylabel: string for ylabel
    :param title: string for title
    :return: loss value plot
    '''
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(linestyle='dotted')
    plt.plot(loss_list)


def plot_comparison(input_img, caption=None, plot=True, save_path=None, save_name=None, save_as='png',
                    save_dpi=300, captions_font = 20, n_row=1, n_col=2,
                    figsize=(5, 5), cmap='gray'):
    '''
    Plot comparison of multiple image but only in column wise!
    :param input_img: Input image list
    :param caption: Input caption list
    :param save_path: Path to save plot
    :param save_name: Name to be save for plot
    :param: save_as: plot save extension, 'png' by DEFAULT
    :param IMG_SIZE: Image size
    :param n_row: Number of row is 1 by DEFAULT
    :param n_col: Number of columns
    :param figsize: Figure size during plotting (5,5) by DEFAULT
    :return: Plot of (n_row, n_col)
    '''
    print()
    if caption!=None:
        assert len(caption) == len(input_img), "Caption length and input image length does not match"
    assert len(input_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        axes[i].imshow(np.squeeze(input_img[i]), cmap=cmap)
        if caption!=None:
            axes[i].set_xlabel(caption[i], fontsize=captions_font)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    if save_path!=None:
        plt.savefig(save_path+'{}.{}'.format(save_name, save_as), save_dpi=save_dpi)
    if plot:
        plt.show()
    else:
        return fig


def plot_hist(inp_img, titles, n_row=1, n_col=2,
              n_bin=20, ranges=[0, 1], figsize=(5, 5)):
    '''
    Plot histogram side by side
    :param inp_img: Input image stacks as list
    :param titles: Input titles as list
    :param n_row: Number of row by DEFAULT 1
    :param n_col: Number of columns by DEFAULT 2
    :param n_bin: Number of bins by DEFAULT 20
    :param ranges: Range of pixel values by DEFAULT [0,1]
    :param figsize: Figure size while plotting by DEFAULT (5,5)
    :return:
        Plot of histograms
    '''
    assert len(titles) == len(inp_img), "Caption length and input image length does not match"
    assert len(inp_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        inp = np.squeeze(inp_img[i])
        axes[i].hist(inp.ravel(), n_bin, ranges)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# reference https://github.com/naomifridman/Unet_Brain_tumor_segmentation
def show_n_images(imgs, titles=None, enlarge=20, cmap='gray'):
    plt.set_cmap(cmap)
    n = len(imgs)
    gs1 = gridspec.GridSpec(1, n)

    fig1 = plt.figure();  # create a figure with the default size
    fig1.set_size_inches(enlarge, 2 * enlarge);

    for i in range(n):

        ax1 = fig1.add_subplot(gs1[i])

        ax1.imshow(imgs[i], interpolation='none');
        if (titles is not None):
            ax1.set_title(titles[i])
        ax1.set_xticks([])
        ax1.set_yticks([])

    plt.show()


def grid_plot_nn(img_list, captions, nrows, ncols, plot=True, figsize=(10, 10), axes_pad=(0.02, 0.5)
                 , cmap='gray'):
    '''
    This function plots grid images with in take of a list of nearest neighbor

    img_list: A list of images
    nrows: Number of rows
    ncols: Number of columns
    figsize: Figure size of each image in the plot grid
    axes_pad: Padding between the grid
    cmap: Color map
    '''
    assert type(img_list)==list, 'Please input img_list as list'
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=axes_pad)
    nimgs = nrows * ncols
    for steps, (ax, im, cp) in enumerate(zip(grid, img_list, captions)):
        for i in range(0, nimgs, ncols):
            ax.imshow(np.squeeze(im), cmap=cmap)
            ax.set_title(cp)
            ax.set_xticks([])
            ax.set_yticks([])
    if plot:
        plt.show()
    return fig

def plot_hist(inp_img, titles, n_row=1, n_col=2,
              n_bin=20, ranges=[0, 1], figsize=(5, 5)):
    '''
    Plot histogram side by side
    :param inp_img: Input image stacks as list
    :param titles: Input titles as list
    :param n_row: Number of row by DEFAULT 1
    :param n_col: Number of columns by DEFAULT 2
    :param n_bin: Number of bins by DEFAULT 20
    :param ranges: Range of pixel values by DEFAULT [0,1]
    :param figsize: Figure size while plotting by DEFAULT (5,5)
    :return:
        Plot of histograms
    '''
    assert len(titles) == len(inp_img), "Caption length and input image length does not match"
    assert len(inp_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        inp = np.squeeze(inp_img[i])
        axes[i].hist(inp.ravel(), n_bin, ranges)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def pca_plot_2D(embeddings, labels, n_class, title=None, show=True,
                title_fontsize=20, figsize=(10, 5), cmap='tab20'):
    '''
    2D-Principle Component Analysis plot
    @param embeddings: Embeddings input
    @param labels: Labels input
    @param n_class: Number of class
    @param title_fontsize: Fontsize for title
    @param figsize: Plot figure size
    @return: 2D-PCA plot
    '''
    pca = decomposition.PCA(n_components=2)
    pca.fit(embeddings)
    x_projected = pca.transform(embeddings)
    fig = plt.figure(figsize=figsize)
    plt.scatter(x_projected[:, 0], x_projected[:, 1],
                c=labels, cmap=cmap)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(n_class+1) - 0.5).set_ticks(np.arange(n_class))
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    if show:
        plt.show()
    return fig


def pca_plot_2Dv2(embeddings, labels, n_class, title, title_fontsize=15, figsize=(10,10)):
    pca = decomposition.PCA(n_components=2)
    pca.fit(embeddings)
    x_projected = pca.transform(embeddings)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=figsize)
    for lab in range(n_class):
        indices = labels==lab
        ax.scatter(x_projected[indices,0],x_projected[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab)
    ax.legend(loc='best')
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()
    return fig

def easy_plot(image, caption, captions_font = 20, figsize=(10,10), show=False):
    fig = plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.xlabel(caption, fontsize=captions_font)
    plt.xticks([])
    plt.yticks([])
    if show==True:
        plt.show()
    return fig


def cm_plot(y_true, y_pred, show=False, figsize=(5,5), fontsize=15):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
    ax.set_xlabel("Predicted Label", fontsize=fontsize)
    ax.set_ylabel("True Label", fontsize=fontsize)
    if show is True:
       plt.show()
    return fig


def roc_plot(y_true, y_pred_prob, average='macro', show=False):
    # false positive rate, true positive rate
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob, average)
    roc_auc = auc(fpr, tpr)
    plot = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
            label="ROC Curve (area = %0.2f)"% roc_auc)
    plt.plot([0,1],[0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plot')
    plt.legend(loc="lower right")
    plt.grid(linestyle=':')
    if show is True:
        plt.show()
    return plot


def train_val_roc_plot(y_train, y_val, show=False):
    # false positive rate, true positive rate
    tr_fpr, tr_tpr, _ = roc_curve(y_train[0], y_train[1])
    tr_roc_auc = auc(tr_fpr, tr_tpr)
    val_fpr, val_tpr, _ = roc_curve(y_val[0], y_val[1])
    val_roc_auc = auc(val_fpr, val_tpr)
    plot = plt.figure()
    lw = 2
    plt.plot(tr_fpr, tr_tpr, color='darkgreen', lw=lw,
            label="Training ROC Curve (area = %0.2f)"% tr_roc_auc)
    plt.plot(val_fpr, val_tpr, color='darkorange', lw=lw,
            label="Validation ROC Curve (area = %0.2f)"% val_roc_auc)
    plt.plot([0,1],[0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plot')
    plt.legend(loc="lower right")
    plt.grid(linestyle=':')
    if show is True:
        plt.show()
    return plot

def train_val_pr_plot(y_train, y_val, average="macro", show=False):
    tr_pr, tr_re, _ = precision_recall_curve(y_train[0], y_train[1])
    tr_ap = average_precision_score(y_train[0], y_train[1], average)
    val_pr, val_re, _ = precision_recall_curve(y_val[0], y_val[1])
    val_ap = average_precision_score(y_val[0], y_val[1], average)
    lw = 2
    plot = plt.figure()
    plt.plot(tr_re, tr_pr, color='darkgreen', lw=lw,
             label="Training precision-recall (AP = %0.2f)"% tr_ap)
    plt.plot(val_re, val_pr, color='darkorange', lw=lw,
             label="Validation precision-recall (AP = %0.2f)"% val_ap)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.grid(linestyle=':')
    plt.title('Precision Recall Curve')
    if show is True:
        plt.show()
    return plot


def binary_histogram_plot(y_true, y_pred, centre_val=0.5,
                          low_bound=0.0, up_bound=1.2, interval=100,
                          show=True):
    '''
    Plot distribution of y_pred probability w.r.t to negatives and positves
    @param y_true: class labels
    @param y_pred: class probability from the model
    @param show: boolean to display the figure or not
    @return: probability distribution of binary class
    '''
    tp_idx = np.where(y_true==1)[0]
    tn_idx = np.where(y_true==0)[0]
    tp_prob = np.take(y_pred, tp_idx)
    tn_prob = np.take(y_pred, tn_idx)

    bins = np.linspace(low_bound, up_bound, interval)
    plot = plt.figure()
    plt.hist(tp_prob, bins, alpha=0.5, label='Positive')
    plt.hist(tn_prob, bins, alpha=0.5, label='Negative')
    plt.legend(loc='upper right')
    plt.axvline(centre_val, color='k', linestyle='dashed', linewidth=1)
    if not show:
        plt.close(plot)
    else:
        plt.show()
    return plot

