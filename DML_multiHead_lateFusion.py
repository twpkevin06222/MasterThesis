import sys
import argparse
from classification_models_3D.keras import Classifiers
import numpy as np
import tensorflow as tf
import os
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
import wandb
import utils, utils_vis, metric_loss, utils_metric
import pandas as pd
import gc
import tempfile
from sklearn.metrics import precision_recall_fscore_support, classification_report, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from focal_loss import SparseCategoricalFocalLoss
#%%
parser = argparse.ArgumentParser("Arguments for path")
parser.add_argument('--weights_path', type=str, help='weight paths where the T2,DWI,ADC are stored')
parser.add_argument('--save_weights_path',type=str, help='paths where model weights are save')
args = parser.parse_args()
wandb.init()
config = wandb.config
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')
optimizer = eval(config.optimizer)(learning_rate=float(config.learning_rate))
max_epochs = config.epochs
n_class = config.n_class
lr_state = config.lr_state
batch_size = config.batch_size
backbone_model = config.backbone_model
depth = config.depth
patch_size = (144, 144, depth)
init_filters = config.init_filters
weights_decay = float(config.weights_decay)
mm_norm = config.min_max_norm
val_batch_size = config.val_batch_size
margin = config.margin
logits_scale = eval(config.logits_scale)
emb_dim = config.emb_dim
n_channel = config.n_channel
gamma = config.gamma
val_fold = config.val_fold
head = config.head

inference_weights_path = args.weights_path
model_weightList = [inference_weights_path + '/{0}/fold{1}/{0}_t2.h5'.format(head, val_fold),
                    inference_weights_path + '/{0}/fold{1}/{0}_dwi.h5'.format(head, val_fold),
                    inference_weights_path + '/{0}/fold{1}/{0}_adc.h5'.format(head, val_fold)]

if n_class==2:
    label_col = 'labels_2'
    fold_col = 'fold_2'
#%%
from tensorflow.keras.layers import Dense, Dropout, Lambda, GlobalAveragePooling3D
from tensorflow.keras.models import Model

#define model
ResNet, preprocess_input = Classifiers.get(backbone_model)
backbone = ResNet(input_shape=(depth, 144, 144, 1), init_filters=init_filters)


def add_regularization(model, regularizer=tf.keras.regularizers.l2(weights_decay)):
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def pre_model(backbone, weights_decay=float(0)):
    if weights_decay!=float(0):
        backbone = add_regularization(backbone)
    x = backbone.layers[-4].output
    x = Dropout(config.dropout)(x)
    x = GlobalAveragePooling3D()(x)
    if weights_decay!=float(0):
        clf = Dense(1, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(x)
    else:
        clf = Dense(1, use_bias=False)(x)
    model = Model(inputs=backbone.inputs, outputs=clf)
    return model


def base(backbone, weights_decay=weights_decay, load_weights=None):
    pre = pre_model(backbone, weights_decay=weights_decay)
    if load_weights!=None:
        pre.load_weights(load_weights)
    return Model(inputs=backbone.inputs, outputs=pre.layers[-2].output, name="backbone")


class CustomModel(Model):
    def __init__(self, backbone, weights_decay=float(0),
                 margin=0.1, logits_scale=[2.5, 2.5, 3],
                 load_weights=None, freeze=False,
                 head='arc'):
        super(CustomModel, self).__init__()
        self.base = base(backbone, weights_decay, load_weights)
        self.head = head
        if freeze!=False:
            self.base.trainable = False
        self.l2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        if self.head=='arc':
            # archead
            self.arc_head_prostatitis = metric_loss.ArcFaceLoss(2, margin, logits_scale[0])
            # cancerous
            self.arc_head_maglinant = metric_loss.ArcFaceLoss(2, margin, logits_scale[1])
            # gleason grade score
            self.arc_head_ggg = metric_loss.ArcFaceLoss(6, margin, logits_scale[2])
            # risk group
            self.arc_head_risk = metric_loss.ArcFaceLoss(4, margin, logits_scale[3])
            # clinical significant ggg (ggg0, ggg>1, ggg1)
            self.arc_head_gggs = metric_loss.ArcFaceLoss(3, margin, logits_scale[4])
            # tumour or not
            self.arc_head_tumour = metric_loss.ArcFaceLoss(2, margin, logits_scale[5])
        else:
            # dense head
            self.dense_head_prostatitis = Dense(2, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_maglinant = Dense(2, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_ggg = Dense(6, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_risk = Dense(4, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_tumour = Dense(2, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_gggs = Dense(3, use_bias=False, kernel_initializer='he_normal')
    def call(self, inps, training=True, l2_norm=True):
        img, lbl = inps
        x = self.base(img, training)
        x = self.bn1(x, training=training)
        if self.head=='arc':
            # multi-archead
            logits_p, _ = self.arc_head_prostatitis(x, lbl[:,0],
                                          binary=False, easy_margin=False,
                                          training=training)
            logits_m, cos_tm = self.arc_head_maglinant(x, lbl[:,1],
                                          binary=False, easy_margin=False,
                                          training=training)
            logits_g, _ = self.arc_head_ggg(x, lbl[:, 2],
                                            binary=False, easy_margin=False,
                                            training=training)
            logits_r, _ = self.arc_head_risk(x, lbl[:,3],
                                             binary=False, easy_margin=False,
                                             training=training)
            logits_s, _ = self.arc_head_gggs(x, lbl[:,4],
                                             binary=False, easy_margin=False,
                                             training=training)
            logits_t, _ = self.arc_head_tumour(x, lbl[:,5],
                                             binary=False, easy_margin=False,
                                             training=training)
        else:
            logits_p = self.dense_head_prostatitis(x)
            logits_m = self.dense_head_maglinant(x)
            logits_g = self.dense_head_ggg(x)
            logits_r = self.dense_head_risk(x)
            logits_s = self.dense_head_gggs(x)
            logits_t = self.dense_head_tumour(x)
            cos_tm = 0
        if l2_norm:
            emb = self.l2(x)
        else:
            emb = x
        return {"emb":emb, "logits_p":logits_p, "logits_m":logits_m, "logits_g":logits_g,
                "logits_r":logits_r, "logits_s":logits_s, "logits_t":logits_t,
                "cos_tm":cos_tm}


class CustomHead(Model):
    def __init__(self, margin, logits_scale, head):
        super(CustomHead, self).__init__()
        self.dense = Dense(128, use_bias=False, kernel_initializer='he_normal')
        self.drop1 = Dropout(config.dropout)
        self.head = head
        self.l2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        if self.head=='arc':
            # archead
            self.arc_head_prostatitis = metric_loss.ArcFaceLoss(2, margin, logits_scale[0])
            # cancerous
            self.arc_head_maglinant = metric_loss.ArcFaceLoss(2, margin, logits_scale[1])
            # gleason grade score
            self.arc_head_ggg = metric_loss.ArcFaceLoss(6, margin, logits_scale[2])
            # risk group
            self.arc_head_risk = metric_loss.ArcFaceLoss(4, margin, logits_scale[3])
            # clinical significant ggg (ggg0, ggg>1, ggg1)
            self.arc_head_gggs = metric_loss.ArcFaceLoss(3, margin, logits_scale[4])
            # tumour or not
            self.arc_head_tumour = metric_loss.ArcFaceLoss(2, margin, logits_scale[5])
        else:
            # dense head
            self.dense_head_prostatitis = Dense(2, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_maglinant = Dense(2, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_ggg = Dense(6, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_risk = Dense(4, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_tumour = Dense(2, use_bias=False, kernel_initializer='he_normal')
            self.dense_head_gggs = Dense(3, use_bias=False, kernel_initializer='he_normal')
    def call(self, inps, training=None, l2_norm=True):
        concat_emb, lbl = inps
        x = self.dense(concat_emb)
        x = self.drop1(x, training=training)
        if self.head=='arc':
            # multi-archead
            logits_p, _ = self.arc_head_prostatitis(x, lbl[:,0],
                                                    binary=False, easy_margin=False,
                                                    training=training)
            logits_m, cos_tm = self.arc_head_maglinant(x, lbl[:,1],
                                                       binary=False, easy_margin=False,
                                                       training=training)
            logits_g, _ = self.arc_head_ggg(x, lbl[:, 2],
                                            binary=False, easy_margin=False,
                                            training=training)
            logits_r, _ = self.arc_head_risk(x, lbl[:,3],
                                             binary=False, easy_margin=False,
                                             training=training)
            logits_s, _ = self.arc_head_gggs(x, lbl[:,4],
                                             binary=False, easy_margin=False,
                                             training=training)
            logits_t, _ = self.arc_head_tumour(x, lbl[:,5],
                                               binary=False, easy_margin=False,
                                               training=training)
        else:
            logits_p = self.dense_head_prostatitis(x)
            logits_m = self.dense_head_maglinant(x)
            logits_g = self.dense_head_ggg(x)
            logits_r = self.dense_head_risk(x)
            logits_s = self.dense_head_gggs(x)
            logits_t = self.dense_head_tumour(x)
            cos_tm = 0
        if l2_norm:
            emb = self.l2(x)
        else:
            emb = x
        return {"emb":emb, "logits_p":logits_p, "logits_m":logits_m, "logits_g":logits_g,
                "logits_r":logits_r, "logits_s":logits_s, "logits_t":logits_t,
                "cos_tm":cos_tm}
#%%
lr_decay = utils.LRDecay(initial_learning_rate=config.learning_rate, max_epochs=config.epochs, decay_rate=0.9)

loss_fn = config.head
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
bxent = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=config.label_smoothing)

# class weights
w_p = [0.31, 0.69]
w_m = [0.42, 0.58]
w_g = [0.04, 0.11, 0.09, 0.18, 0.38,  0.2]
w_r = [0.13, 0.31, 0.18, 0.38]
w_s = [0.23, 0.55, 0.22]
w_t = [0.59, 0.41]

# loss
sfl_p = SparseCategoricalFocalLoss(from_logits=True, gamma=gamma, class_weight=w_p)
sfl_m = SparseCategoricalFocalLoss(from_logits=True, gamma=gamma, class_weight=w_m)
sfl_g = SparseCategoricalFocalLoss(from_logits=True, gamma=gamma, class_weight=w_g)
sfl_r = SparseCategoricalFocalLoss(from_logits=True, gamma=gamma, class_weight=w_r)
sfl_s = SparseCategoricalFocalLoss(from_logits=True, gamma=gamma, class_weight=w_s)
sfl_t = SparseCategoricalFocalLoss(from_logits=True, gamma=gamma, class_weight=w_t)


@tf.function
def train_fn(model, image, label, epochs, lr):
    learning_rates = [1e-4, 5e-5, 2.5e-5, 1e-5]
    epochs_list = [20, 40, 60, 80]
    with tf.GradientTape() as tape:
        model_output = model([image, label], training=True)
        # focal loss
        loss_p = sfl_p(y_true=label[:, 0], y_pred=model_output["logits_p"])
        loss_m = sfl_m(y_true=label[:, 1], y_pred=model_output["logits_m"])
        loss_g = sfl_g(y_true=label[:, 2], y_pred=model_output["logits_g"])
        loss_r = sfl_r(y_true=label[:, 3], y_pred=model_output["logits_r"])
        loss_s = sfl_s(y_true=label[:, 4], y_pred=model_output["logits_s"])
        loss_t = sfl_t(y_true=label[:, 5], y_pred=model_output["logits_t"])
        if config.loss_aggregration=="mean":
            # take the mean of losses
            loss = tf.reduce_mean([loss_p, loss_m, loss_g, loss_r, loss_s, loss_t])
        else:
            # take the sum of the losses
            loss = tf.reduce_sum([loss_p, loss_m, loss_g, loss_r, loss_s, loss_t])
        if weights_decay!=float(0):
            reg_loss = tf.reduce_sum(model.losses)
        else:
            reg_loss = 0
        total_loss = reg_loss + loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    if lr_state=='scheduler':
        # optimizer.lr = lr_decay(epochs, mode="step_wise", epochs_list = epochs_list, learning_rates=learning_rates)
        # optimizer.lr = lr_decay(epochs)
        optimizer.lr = lr
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # probability output
    # protatitis
    y_p = tf.squeeze(tf.cast(label[:,0], dtype=tf.int64))
    softmax_logits_p = tf.squeeze(tf.math.softmax(model_output["logits_p"]))
    y_pred_p = tf.cast(tf.argmax(softmax_logits_p, axis=-1), dtype=tf.int64)
    train_acc_p = tf.reduce_mean(tf.cast(tf.equal(y_pred_p, y_p), dtype=tf.float32))
    # maglinant
    y_m = tf.squeeze(tf.cast(label[:, 1], dtype=tf.int64))
    softmax_logits_m = tf.squeeze(tf.math.softmax(model_output["logits_m"]))
    y_pred_m = tf.cast(tf.argmax(softmax_logits_m, axis=-1), dtype=tf.int64)
    train_acc_m = tf.reduce_mean(tf.cast(tf.equal(y_pred_m, y_m), dtype=tf.float32))
    # ggg
    y_g = tf.squeeze(tf.cast(label[:,2], dtype=tf.int64))
    softmax_logits_g = tf.squeeze(tf.math.softmax(model_output["logits_g"]))
    y_pred_g = tf.cast(tf.argmax(softmax_logits_g, axis=-1), dtype=tf.int64)
    train_acc_g = tf.reduce_mean(tf.cast(tf.equal(y_pred_g, y_g), dtype=tf.float32))
    # risk
    y_r = tf.squeeze(tf.cast(label[:,3], dtype=tf.int64))
    softmax_logits_r = tf.squeeze(tf.math.softmax(model_output["logits_r"]))
    y_pred_r = tf.cast(tf.argmax(softmax_logits_r, axis=-1), dtype=tf.int64)
    train_acc_r = tf.reduce_mean(tf.cast(tf.equal(y_pred_r, y_r), dtype=tf.float32))
    # ggg_s
    y_s = tf.squeeze(tf.cast(label[:,4], dtype=tf.int64))
    softmax_logits_s = tf.squeeze(tf.math.softmax(model_output["logits_s"]))
    y_pred_s = tf.cast(tf.argmax(softmax_logits_s, axis=-1), dtype=tf.int64)
    train_acc_s = tf.reduce_mean(tf.cast(tf.equal(y_pred_s, y_s), dtype=tf.float32))
    # tumour
    y_t = tf.squeeze(tf.cast(label[:, 5], dtype=tf.int64))
    softmax_logits_t = tf.squeeze(tf.math.softmax(model_output["logits_t"]))
    y_pred_t = tf.cast(tf.argmax(softmax_logits_t, axis=-1), dtype=tf.int64)
    train_acc_t = tf.reduce_mean(tf.cast(tf.equal(y_pred_t, y_t), dtype=tf.float32))

    # store in list
    train_acc = [train_acc_p, train_acc_m, train_acc_g, train_acc_r, train_acc_s, train_acc_t] # shape(6)
    y_pred = [y_pred_p, y_pred_m, y_pred_g, y_pred_r, y_pred_s, y_pred_t] # shape (B, 6)
    y_pred_prob = [softmax_logits_p[:,1], softmax_logits_m[:,1], softmax_logits_t[:,1]] # shape (B, 6)
    loss_sub = [loss_p, loss_m, loss_g, loss_r, loss_s, loss_t] # shape (6)
    return {"train_{}".format(loss_fn): loss, "train_acc": train_acc,
            "train_reg_loss": reg_loss,"train_total_loss": total_loss, "train_pred": y_pred,
            "lr": optimizer.lr, "train_pred_prob": y_pred_prob, "train_loss_sub":loss_sub,
            "gradients":[tf.reduce_min(gradients[0]), tf.reduce_max(gradients[0])]}


def test_fn(model, image, label, l2_norm):
    model_output = model([image, label], training=False, l2_norm=l2_norm)
    # focal loss
    loss_p = sfl_p(y_true=label[:, 0], y_pred=model_output["logits_p"])
    loss_m = sfl_m(y_true=label[:, 1], y_pred=model_output["logits_m"])
    loss_g = sfl_g(y_true=label[:, 2], y_pred=model_output["logits_g"])
    loss_r = sfl_r(y_true=label[:, 3], y_pred=model_output["logits_r"])
    loss_s = sfl_s(y_true=label[:, 4], y_pred=model_output["logits_s"])
    loss_t = sfl_t(y_true=label[:, 5], y_pred=model_output["logits_t"])
    if config.loss_aggregration == "mean":
        # take the mean of losses
        loss = tf.reduce_mean([loss_p, loss_m, loss_g, loss_r, loss_s, loss_t])
    else:
        # take the sum of the losses
        loss = tf.reduce_sum([loss_p, loss_m, loss_g, loss_r, loss_s, loss_t])
    if weights_decay != float(0):
        reg_loss = tf.reduce_sum(model.losses)
    else:
        reg_loss = 0
    total_loss = reg_loss + loss
    # probability output
    # protatitis
    y_p = tf.squeeze(tf.cast(label[:,0], dtype=tf.int64))
    softmax_logits_p = tf.squeeze(tf.math.softmax(model_output["logits_p"]))
    y_pred_p = tf.cast(tf.argmax(softmax_logits_p, axis=-1), dtype=tf.int64)
    val_acc_p = tf.reduce_mean(tf.cast(tf.equal(y_pred_p, y_p), dtype=tf.float32))
    # maglinant
    y_m = tf.squeeze(tf.cast(label[:,1], dtype=tf.int64))
    softmax_logits_m = tf.squeeze(tf.math.softmax(model_output["logits_m"]))
    y_pred_m = tf.cast(tf.argmax(softmax_logits_m, axis=-1), dtype=tf.int64)
    val_acc_m = tf.reduce_mean(tf.cast(tf.equal(y_pred_m, y_m), dtype=tf.float32))
    # ggg
    y_g = tf.squeeze(tf.cast(label[:,2], dtype=tf.int64))
    softmax_logits_g = tf.squeeze(tf.math.softmax(model_output["logits_g"]))
    y_pred_g = tf.cast(tf.argmax(softmax_logits_g, axis=-1), dtype=tf.int64)
    val_acc_g = tf.reduce_mean(tf.cast(tf.equal(y_pred_g, y_g), dtype=tf.float32))
    # risk
    y_r = tf.squeeze(tf.cast(label[:,3], dtype=tf.int64))
    softmax_logits_r = tf.squeeze(tf.math.softmax(model_output["logits_r"]))
    y_pred_r = tf.cast(tf.argmax(softmax_logits_r, axis=-1), dtype=tf.int64)
    val_acc_r = tf.reduce_mean(tf.cast(tf.equal(y_pred_r, y_r), dtype=tf.float32))
    # ggg_s
    y_s = tf.squeeze(tf.cast(label[:,4], dtype=tf.int64))
    softmax_logits_s = tf.squeeze(tf.math.softmax(model_output["logits_s"]))
    y_pred_s = tf.cast(tf.argmax(softmax_logits_s, axis=-1), dtype=tf.int64)
    val_acc_s = tf.reduce_mean(tf.cast(tf.equal(y_pred_s, y_s), dtype=tf.float32))
    # tumour
    y_t = tf.squeeze(tf.cast(label[:, 5], dtype=tf.int64))
    softmax_logits_t = tf.squeeze(tf.math.softmax(model_output["logits_t"]))
    y_pred_t = tf.cast(tf.argmax(softmax_logits_t, axis=-1), dtype=tf.int64)
    val_acc_t = tf.reduce_mean(tf.cast(tf.equal(y_pred_t, y_t), dtype=tf.float32))
    # store in list
    val_acc = [val_acc_p, val_acc_m, val_acc_g, val_acc_r, val_acc_s, val_acc_t]
    y_pred = [y_pred_p, y_pred_m, y_pred_g, y_pred_r, y_pred_s, y_pred_t]
    y_pred_prob = [softmax_logits_p[:,1], softmax_logits_m[:,1], softmax_logits_t[:,1]]
    loss_sub = [loss_p, loss_m, loss_g, loss_r, loss_s, loss_t]
    # embedding
    emb = model_output["emb"]
    return {"val_{}".format(loss_fn): loss, "val_acc": val_acc,
            "val_reg_loss": reg_loss, "val_total_loss": total_loss, "val_pred": y_pred,
            "val_pred_prob": y_pred_prob, "val_emb": emb, "val_loss_sub":loss_sub}


#%%
def main(model, model_head, val_fold):
    print("Fold: {}".format(val_fold))
    # initialize saving criterion--------------------------------
    current_acc = 0
    current_auc = 0
    current_p = 0
    current_map = 0
    k = config.k
    R = config.R
    # ----------------------------------------------------------
    tabular_data = pd.read_csv(config.csv)
    ds_dict = utils.multi_get_split_fold(tabular_data, label_col=label_col, fold_col=fold_col, val_fold=val_fold)
    tr_transforms = utils.get_train_transform(patch_size=patch_size, prob=config.aug_prob)
    lr_decay = utils.ReduceLRonPlateau(initial_lr=config.learning_rate, max_lr=float(1e-5), factor=0.5, mode='min')

    epochs = 1
    val_total_loss = 0
    while epochs <= max_epochs:
        if epochs==1:
            lr = float(config.learning_rate)
        else:
            lr = lr_decay(monitor=val_total_loss, patience=5)
        train_dl = utils.DataLoader3D_multi(data=ds_dict['train_ds'], batch_size=batch_size, patch_size=patch_size,
                                      num_threads_in_multithreaded=4, crop_type="random", seed_for_shuffle=5243,
                                      margins=(45, 45, 0), return_incomplete=False, shuffle=True, infinite=True)
        if config.aug_prob!=0.0:
            train_gen = MultiThreadedAugmenter(train_dl, tr_transforms, num_processes=4,
                                               num_cached_per_queue=2,
                                               seeds=None, pin_memory=False)
        else:
            print("No augmentation!")
            train_gen = MultiThreadedAugmenter(train_dl, None, num_processes=4,
                                               num_cached_per_queue=2,
                                               seeds=None, pin_memory=False)
        val_dl = utils.DataLoader3D_multi(data=ds_dict['val_ds'], batch_size=val_batch_size, patch_size=patch_size,
                                    num_threads_in_multithreaded=1, crop_type="center", seed_for_shuffle=5243,
                                    margins=(0, 0, 0), return_incomplete=True, shuffle=False, infinite=False)
        print("Epochs: {} -----------------------".format(epochs))
        # training list...
        train_batch_loss = 0
        train_batch_reg_loss = 0
        train_batch_total_loss = 0
        train_batch_loss_sub = np.zeros(6)
        train_acc_batch = np.zeros(6)
        # validation list...
        val_batch_loss = 0
        val_batch_reg_loss = 0
        val_batch_total_loss = 0
        val_batch_loss_sub = np.zeros(6)
        val_acc_batch = np.zeros(6)
        if len(ds_dict['train_ds']['labels'])%batch_size == 0:
            train_total_batch = (int(round(len(ds_dict['train_ds']['labels'])/batch_size)))
        else:
            train_total_batch = (int(len(ds_dict['train_ds']['labels'])/batch_size)+1)
        # print("Training...")
        for i in tqdm(range(train_total_batch)):
            train_batch = next(train_gen)
            imgs = np.swapaxes(train_batch["data"], 1, -1)
            # standardization
            imgs = utils.standardization(imgs, (1, 2, 3))
            # min max normalization
            if mm_norm==True:
                imgs = utils.min_max_norm(imgs, (1, 2, 3))
            labels = np.stack([train_batch["prostatitis"], train_batch["maglinant"], train_batch["ggg"],
                               train_batch["risk"], train_batch["ggg_s"], train_batch["tumour"]], axis=-1)
            # concat embeddings
            for m in range(3):
                model.load_weights(model_weightList[m])
                backbone_output = model([imgs[...,m], labels], training=False, l2_norm=False)
                # embedding
                train_emb = backbone_output["emb"]
                if m == 0:
                    train_concat_emb = np.zeros_like(train_emb)
                # -------------
                train_concat_emb = np.concatenate((train_concat_emb, train_emb), axis=-1)
            train_concat_emb = train_concat_emb[:, config.emb_dim:]
            train_output = train_fn(model_head, train_concat_emb, labels, epochs, lr)
            #
            train_pred = np.stack(train_output["train_pred"], axis=-1)
            train_pred_prob = np.stack(train_output["train_pred_prob"], axis=-1)
            # train collections....
            train_batch_loss += train_output["train_{}".format(loss_fn)]
            train_batch_reg_loss += train_output["train_reg_loss"]
            train_batch_total_loss += train_output["train_total_loss"]
            train_batch_loss_sub += np.array(train_output['train_loss_sub'])
            train_acc_batch += np.array(train_output["train_acc"])

            if i == 0:
                # stack total label
                train_multi_lbl_metric = np.zeros_like(labels)
                train_pred_metric = np.zeros_like(train_pred)
                train_pred_prob_metric = np.zeros_like(train_pred_prob)
            train_multi_lbl_metric = np.vstack((train_multi_lbl_metric, labels))
            train_pred_metric = np.vstack((train_pred_metric, train_pred))
            train_pred_prob_metric = np.vstack((train_pred_prob_metric, train_pred_prob))
        # train loss and accuracy...
        train_loss = train_batch_loss/train_total_batch
        train_reg_loss = train_batch_reg_loss/train_total_batch
        train_total_loss = train_batch_total_loss/train_total_batch
        train_loss_sub = train_batch_loss_sub/train_total_batch
        train_acc_sub = train_acc_batch/train_total_batch

        # remove the intantiated zero like for train embedding
        train_multi_lbl_metric = np.squeeze(train_multi_lbl_metric[batch_size:])
        train_pred_metric = np.squeeze(train_pred_metric[batch_size:])
        train_pred_prob_metric = np.squeeze(train_pred_prob_metric[batch_size:])

        # Validation-----------------------------------------------
        if len(ds_dict['val_ds']['labels'])%val_batch_size == 0:
            val_total_batch = int(round(len(ds_dict['val_ds']['labels'])/val_batch_size))
        else:
            val_total_batch = int(len(ds_dict['val_ds']['labels'])/val_batch_size)+1
        # print("Validating...")
        for j in tqdm(range(val_total_batch)):
            val_batch = next(val_dl)
            imgs = np.swapaxes(val_batch["data"], 1, -1)
            # standardization
            imgs = utils.standardization(imgs, (1, 2, 3))
            # min max normalization
            if mm_norm==True:
                imgs = utils.min_max_norm(imgs, (1, 2, 3))
            labels = np.stack([val_batch["prostatitis"], val_batch["maglinant"], val_batch["ggg"],
                               val_batch["risk"], val_batch["ggg_s"], val_batch["tumour"]], axis=-1)
            for m in range(3):
                model.load_weights(model_weightList[m])
                backbone_output = model([imgs[...,m], labels], training=False, l2_norm=False)
                # embedding
                val_emb = backbone_output["emb"]
                if m == 0:
                    val_concat_emb = np.zeros_like(val_emb)
                # -------------
                val_concat_emb = np.concatenate((val_concat_emb, val_emb), axis=-1)

            val_concat_emb = val_concat_emb[:, config.emb_dim:]
            val_output = test_fn(model_head, val_concat_emb, labels, l2_norm=True)
            #
            val_pred = np.stack(val_output["val_pred"], axis=-1)
            val_pred_prob = np.stack(val_output["val_pred_prob"], axis=-1)
            # embedding
            val_emb = val_output["val_emb"]
            # stack embedding for each batch because the validation ds cannot load one shot
            # to the GPU yielding OOM Error
            if j == 0:
                #----------
                val_multi_lbl_metric = np.zeros_like(labels)
                val_pred_metric = np.zeros_like(val_pred)
                val_pred_prob_metric = np.zeros_like(val_pred_prob)
                val_emb_metric = np.zeros_like(val_emb)
            #-------------
            val_multi_lbl_metric = np.vstack((val_multi_lbl_metric, labels))
            val_pred_metric = np.vstack((val_pred_metric, val_pred))
            val_pred_prob_metric = np.vstack((val_pred_prob_metric, val_pred_prob))
            val_emb_metric = np.vstack((val_emb_metric, val_emb))
            #loss------------
            val_batch_loss += val_output["val_{}".format(loss_fn)]
            val_batch_reg_loss += val_output["val_reg_loss"]
            val_batch_total_loss += val_output["val_total_loss"]
            val_batch_loss_sub += np.array(val_output['val_loss_sub'])
            val_acc_batch += np.array(val_output["val_acc"])
        #------
        val_multi_lbl_metric = np.squeeze(val_multi_lbl_metric[val_batch_size:])
        val_pred_metric = np.squeeze(val_pred_metric[val_batch_size:])
        val_pred_prob_metric = np.squeeze(val_pred_prob_metric[val_batch_size:])
        val_emb_metric = np.squeeze(val_emb_metric[val_batch_size:])
        # Validation loss and accuracy--------------------------------------------------------------
        val_loss = val_batch_loss/val_total_batch
        val_reg_loss = val_batch_reg_loss/val_total_batch
        val_total_loss = val_batch_total_loss/val_total_batch
        val_loss_sub = val_batch_loss_sub/val_total_batch
        val_acc_sub = val_acc_batch/val_total_batch
        # partial
        val_metric_ = utils_metric.Metric(val_emb_metric, k, None, val_multi_lbl_metric[:,1], max_match=R)
        val_metric_dict = val_metric_.get_metric()
        # Output-------------------------------------------------------------
        print("Training_loss: {}, Training_acc_m: {}".format(train_loss, train_acc_sub[1]))
        print("Training_acc_p: {}, Training_acc_g: {}, Training_acc_r: {}".format(train_acc_sub[0], train_acc_sub[2], train_acc_sub[3]))
        print("Training_acc_s: {}, Training_acc_t: {}".format(train_acc_sub[4], train_acc_sub[5]))
        print("Training_reg_loss: {}, Training_total_loss: {}".format(train_reg_loss, train_total_loss))
        print("Gradients: ", train_output["gradients"])
        print("Validation_loss: {}, Validation_acc_m: {}".format(val_loss, val_acc_sub[1]))
        print("Validatin_acc_p: {}, Validation_acc_g: {}, Validation_acc_r: {}".format(val_acc_sub[0], val_acc_sub[2], val_acc_sub[3]))
        print("Validation_acc_s: {}, Validation_acc_t: {}".format(val_acc_sub[4], val_acc_sub[5]))
        print("Validation_reg_loss: {}, Validation_total_loss: {}".format(val_reg_loss, val_total_loss))
        print("Metrics:")
        # maglinant
        print("Malignant")
        target_names = ['non-cancer', 'cancer']
        print("Training_m:")
        print(classification_report(y_true=train_multi_lbl_metric[:,1], y_pred=train_pred_metric[:,1], target_names=target_names, zero_division=1))
        print("Validation_m:")
        print(classification_report(y_true=val_multi_lbl_metric[:,1], y_pred=val_pred_metric[:,1], target_names=target_names, zero_division=1))
        # embedding
        print("Validation Metrics")
        print("precision@{0} recall@{0} r_precision MAP@R".format(k))
        f = '{:10.3f}{:9.3f}{:10.3f}{:8.3f}'
        print(f.format(val_metric_dict['precision@{}'.format(k)], val_metric_dict['recall@{}'.format(k)],
                       val_metric_dict['r_precision'], val_metric_dict['map@r']))

        # Logging----------------------------------------------------------------
        head = ['prostatitis', 'malignant', 'ggg', 'risk', 'ggg_s', 'tumour']
        # prostatitis
        train_auc_p = utils.auc_score(train_multi_lbl_metric[:,0], train_pred_prob_metric[:,0])
        val_auc_p = utils.auc_score(val_multi_lbl_metric[:,0], val_pred_prob_metric[:,0])
        # maglinant
        train_auc_m = utils.auc_score(train_multi_lbl_metric[:,1], train_pred_prob_metric[:,1])
        val_auc_m = utils.auc_score(val_multi_lbl_metric[:,1], val_pred_prob_metric[:,1])
        # tumour
        train_auc_t = utils.auc_score(train_multi_lbl_metric[:, 5], train_pred_prob_metric[:,2])
        val_auc_t = utils.auc_score(val_multi_lbl_metric[:, 5], val_pred_prob_metric[:,2])
        print("Val AUC: ", val_auc_m)
        # losses-----------------------------------------------------------------
        wandb.log({"Train_loss": train_loss, "Val_loss": val_loss}, step=epochs)
        wandb.log({"Train_reg_loss": train_reg_loss, "Train_total_loss": train_total_loss}, step=epochs)
        wandb.log({"Val_reg_loss": val_reg_loss, "Val_total_loss": val_total_loss}, step=epochs)
        # accuracy-------------------------------------------------------------
        for i in range(len(head)):
            # accuracy
            wandb.log({"Train_accuracy_{}".format(head[i]): train_acc_sub[i], "Val_accuracy_{}".format(head[i]): val_acc_sub[i]}, step=epochs)
            # loss for each granularity
            wandb.log({"Train_loss_{}".format(head[i]): train_loss_sub[i], "Val_loss_{}".format(head[i]): val_loss_sub[i]}, step=epochs)
        wandb.log({'learning rate': train_output["lr"]}, step=epochs)
        # # metrics
        # AUC
        wandb.log({"Train AUC Prostatitis": train_auc_p, "Validation AUC Prostatitis": val_auc_p}, step=epochs)
        wandb.log({"Train AUC malignant": train_auc_m, "Validation AUC malignant": val_auc_m}, step=epochs)
        wandb.log({"Train AUC Tumour": train_auc_t, "Validation AUC Tumour": val_auc_t}, step=epochs)
        # validation metrics
        wandb.log({'val_P@{}'.format(k): val_metric_dict['precision@{}'.format(k)],
                   'val_R@{}'.format(k): val_metric_dict['recall@{}'.format(k)],
                   'val_R_Precision': val_metric_dict['r_precision'],
                   'val_MAP@R':val_metric_dict['map@r']
                   }, step=epochs)
        # Plots
        # Confusion Matrix ------------------------------------
        for i in range(len(head)):
            train_cm = utils_vis.cm_plot(train_multi_lbl_metric[:,i], train_pred_metric[:,i], show=False)
            wandb.log({"Training CM {}".format(head[i]): wandb.Image(train_cm)}, step=epochs)
            val_cm = utils_vis.cm_plot(val_multi_lbl_metric[:,i], val_pred_metric[:,i], show=False)
            wandb.log({"Validation CM {}".format(head[i]): wandb.Image(val_cm)}, step=epochs)

        # AUC -------------------------------------------------
        # prostatitis
        y_train_p = [train_multi_lbl_metric[:,0], train_pred_prob_metric[:,0]]
        y_val_p = [val_multi_lbl_metric[:,0], val_pred_prob_metric[:,0]]
        # malignant
        y_train_m = [train_multi_lbl_metric[:,1], train_pred_prob_metric[:,1]]
        y_val_m = [val_multi_lbl_metric[:,1], val_pred_prob_metric[:,1]]
        # tumour
        y_train_t = [train_multi_lbl_metric[:, 5], train_pred_prob_metric[:,2]]
        y_val_t = [val_multi_lbl_metric[:, 5], val_pred_prob_metric[:,2]]
        # Training Validaiton AUC plot
        roc_plot_p = utils_vis.train_val_roc_plot(y_train_p, y_val_p, show=False)
        wandb.log({"Prostatitis ROC": wandb.Image(roc_plot_p)}, step=epochs)
        roc_plot_m = utils_vis.train_val_roc_plot(y_train_m, y_val_m, show=False)
        wandb.log({"maglinant ROC": wandb.Image(roc_plot_m)}, step=epochs)
        roc_plot_t = utils_vis.train_val_roc_plot(y_train_t, y_val_t, show=False)
        wandb.log({"Tumour ROC": wandb.Image(roc_plot_t)}, step=epochs)
        # Embedding plot-------------------------------------
        val_title = "Validation embedding PCA at epochs: {}".format(epochs)
        val_emb = utils_vis.pca_plot_2D(val_emb_metric, val_multi_lbl_metric[:,1], n_class=2,
                                        title=val_title, show=False)
        wandb.log({"Validation Embedding": wandb.Image(val_emb)}, step=epochs)

        plt.clf()
        plt.close('all')

        del train_cm, val_cm
        del roc_plot_p, roc_plot_m, val_emb
        # Save weights----------------------------------------------------
        weights_dir = args.save_weights_path
        base_path = os.path.dirname(weights_dir)
        p_at_1 = val_metric_dict['precision@{}'.format(k)]
        map = val_metric_dict['map@r']
        if val_auc_m>current_auc:
            current_auc=val_auc_m
            best_auc_file = glob.glob(weights_dir+'best_auc*')
            for file in best_auc_file:
                best_auc_path = weights_dir + file
                if os.path.exists(best_auc_path):
                    os.remove(best_auc_path)
            model_head.save_weights(os.path.join(weights_dir, 'best_auc{}.h5'.format(epochs)))
            wandb.save(os.path.join(weights_dir, 'best_auc{}.h5'.format(epochs)), base_path=base_path)
        if val_acc_sub[1] > current_acc:
            current_acc = val_acc_sub[1]
            best_acc_file = glob.glob(weights_dir + 'best_acc*')
            for file in best_acc_file:
                best_acc_path = weights_dir + file
                if os.path.exists(best_acc_path):
                    os.remove(best_acc_path)
            model_head.save_weights(os.path.join(weights_dir, 'best_acc{}.h5'.format(epochs)))
            wandb.save(os.path.join(weights_dir, 'best_acc{}.h5'.format(epochs)), base_path=base_path)
        if p_at_1>current_p:
            current_p=p_at_1
            best_acc_file = glob.glob(weights_dir+'best_p@{}*'.format(k))
            for file in best_acc_file:
                best_acc_path = weights_dir + file
                if os.path.exists(best_acc_path):
                    os.remove(best_acc_path)
            model_head.save_weights(os.path.join(weights_dir, 'best_p@{}_{}.h5'.format(k, epochs)))
            wandb.save(os.path.join(weights_dir, 'best_p@{}_{}.h5'.format(k, epochs)), base_path=base_path)
        if map>current_map:
            current_map=map
            best_map_file = glob.glob(weights_dir+'best_map*')
            for file in best_map_file:
                best_map_path = weights_dir + file
                if os.path.exists(best_map_path):
                    os.remove(best_map_path)
            model_head.save_weights(os.path.join(weights_dir, 'best_map{}.h5'.format(epochs)))
            wandb.save(os.path.join(weights_dir, 'best_map{}.h5'.format(epochs)), base_path=base_path)
        if epochs==max_epochs:
            model_head.save_weights(os.path.join(weights_dir, 'checkpoint_{}.h5'.format(epochs)))
            wandb.save(os.path.join(weights_dir, 'checkpoint_{}.h5'.format(epochs)), base_path=base_path)
        # memory management--------------------------------------------
        del val_batch_loss, val_batch_total_loss, val_batch_reg_loss
        del train_batch_loss, train_batch_total_loss, train_batch_reg_loss
        del train_output, val_output
        del train_gen, val_dl
        del train_multi_lbl_metric, train_pred_metric, train_pred_prob_metric
        del val_multi_lbl_metric, val_pred_metric, val_pred_prob_metric
        del y_train_p, y_train_m, y_val_m, y_val_p
        del train_auc_p, train_auc_m, val_auc_p, val_auc_m
        gc.collect()
        print()
        epochs += 1
    del train_dl, tabular_data, ds_dict, tr_transforms
    gc.collect()
#%%
if __name__ == '__main__':
    model = CustomModel(backbone, weights_decay=weights_decay,
                        margin=margin, logits_scale=logits_scale,
                        load_weights=None, freeze=False, head=config.head)
    model.build([(None, depth, 144, 144, 1), (None, 6)])
    model_head = CustomHead(margin=margin, logits_scale=logits_scale, head=config.head)
    model_head.build([(None, config.emb_dim*3), (None, 6)])
    model_head.summary()
    # model.summary()
    main(model, model_head, val_fold=config.val_fold)
    sys.exit(0)

