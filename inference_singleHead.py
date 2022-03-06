import sys
from classification_models_3D.keras import Classifiers
import numpy as np
import tensorflow as tf
import os
import wandb
import utils, utils_vis, metric_loss, utils_metric
import pandas as pd
import tempfile
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from focal_loss import SparseCategoricalFocalLoss
#%%
wandb.init()
config = wandb.config
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')
optimizer = eval(config.optimizer)(learning_rate=float(config.learning_rate))
max_epochs = config.epochs
n_class = config.n_class
batch_size = config.batch_size
backbone_model = config.backbone_model
depth = config.depth
patch_size = (144, 144, depth)
init_filters = config.init_filters
weights_decay = float(config.weights_decay)
mm_norm = config.min_max_norm
val_batch_size = config.val_batch_size
margin = config.margin
logits_scale = config.logits_scale
emb_dim = config.emb_dim
n_channel = config.n_channel
gamma = config.gamma
val_fold = config.val_fold
head = config.head

if n_class==2:
    label_col = 'labels_2'
    fold_col = 'fold_2'
#%%
from tensorflow.keras.layers import Dense, Dropout, Lambda, GlobalAveragePooling3D
from tensorflow.keras.models import Model

#define model
ResNet, preprocess_input = Classifiers.get(backbone_model)
if n_channel!=3:
    backbone = ResNet(input_shape=(depth, 144, 144, 1), init_filters=init_filters)
else:
    backbone = ResNet(input_shape=(depth, 144, 144, 3), init_filters=init_filters)

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
    x = backbone.layers[-3].output
    x = Dropout(config.dropout)(x)
    if weights_decay!=float(0):
        clf = Dense(1, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(x)
    else:
        clf = Dense(1, use_bias=False)(x)
    model = Model(inputs=backbone.inputs, outputs=clf)
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
                 margin=0.1, logits_scale=2.5,
                 load_weights=None, freeze=False,
                 head='arc', include_emb=False):
        super(CustomModel, self).__init__()
        self.base = base(backbone, weights_decay, load_weights)
        self.head = head
        self.include_emb = include_emb
        self.l2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        if freeze!=False:
            self.base.trainable = False
        if self.include_emb is True:
            self.emb = Dense(config.emb_dim, use_bias=False, kernel_initializer='he_normal')
            self.dropout_1 = Dropout(config.dropout)
        if self.head=='arc':
            # archead
            self.arc_head_maglinant = metric_loss.ArcFaceLoss(2, margin, logits_scale)
        else:
            # dense head
            self.dense_head_maglinant = Dense(2, use_bias=False, kernel_initializer='he_normal')
    def call(self, inps, training=True, l2_norm=True):
        img, lbl = inps
        x = self.base(img, training=training)
        x = self.bn1(x, training=training)
        if self.include_emb is True:
            x = self.emb(x)
            x = self.dropout_1(x, training=training)
        if self.head=='arc':
            # multi-archead
            logits_m, cos_tm = self.arc_head_maglinant(x, lbl,
                                          binary=False, easy_margin=False,
                                          training=training)
        else:
            logits_m = self.dense_head_maglinant(x)
            cos_tm = 0
        if l2_norm:
            emb = self.l2(x)
        else:
            emb = x
        return {"emb":emb, "logits_m":logits_m, "cos_tm":cos_tm}


class CustomHead(Model):
    def __init__(self, margin, logits_scale, head='arc'):
        super(CustomHead, self).__init__()
        self.dense = Dense(128, use_bias=False, kernel_initializer='he_normal')
        self.drop1 = Dropout(config.dropout)
        self.l2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        self.head = head
        if self.head=='arc':
            # archead
            self.arc_head_maglinant = metric_loss.ArcFaceLoss(2, margin, logits_scale)
        else:
            # dense head
            self.dense_head_maglinant = Dense(2, use_bias=False, kernel_initializer='he_normal')
    def call(self, inps, training=None, l2_norm=True):
        concat_emb, lbl = inps
        x = self.dense(concat_emb)
        x = self.drop1(x, training=training)
        if self.head=='arc':
            logits_m, cos_tm = self.arc_head_maglinant(x, lbl,
                                          binary=False, easy_margin=False,
                                          training=training)
        else:
            logits_m = self.dense_head_maglinant(x)
            cos_tm = 0
        if l2_norm:
            emb = self.l2(x)
        else:
            emb = x
        return {"emb":emb, "logits_m":logits_m, "cos_tm":cos_tm}
#%%
loss_fn = config.head

# class weights
w_m = [0.42, 0.58]

# loss
sfl_m = SparseCategoricalFocalLoss(from_logits=True, gamma=gamma, class_weight=w_m)


def test_fn(model, image, label, l2_norm=True):
    model_output = model([image, label], training=False, l2_norm=l2_norm)
    # focal loss
    loss = sfl_m(y_true=label, y_pred=model_output["logits_m"])
    if weights_decay != float(0):
        reg_loss = tf.reduce_sum(model.losses)
    else:
        reg_loss = 0
    total_loss = reg_loss + loss
    # probability output
    # maglinant
    y_m = tf.squeeze(tf.cast(label, dtype=tf.int64))
    y_pred_prob = tf.squeeze(tf.math.softmax(model_output["logits_m"]))
    y_pred = tf.cast(tf.argmax(y_pred_prob, axis=-1), dtype=tf.int64)
    val_acc = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_m), dtype=tf.float32))

    # embedding
    emb = model_output["emb"]
    return {"val_{}".format(loss_fn): loss, "val_acc": val_acc,
            "val_reg_loss": reg_loss, "val_total_loss": total_loss, "val_pred": y_pred,
            "val_pred_prob": y_pred_prob[:,1], "val_emb": emb}


#%%
def main(model, model_head, val_fold):
    print()
    print("Fold: {}".format(val_fold))
    # initialize saving criterion--------------------------------
    k = config.k
    R = config.R
    ds_list = ['val_ds','test_ds']
    modalities = ["t2", "dwi", "adc"]
    inference_weights_path = "/inference_weights/singlehead"
    if n_channel!=3:
        model_weightList = [inference_weights_path + '/{0}/fold{1}/{0}_t2.h5'.format(head, val_fold),
                            inference_weights_path + '/{0}/fold{1}/{0}_dwi.h5'.format(head, val_fold),
                            inference_weights_path + '/{0}/fold{1}/{0}_adc.h5'.format(head, val_fold),
                            inference_weights_path + '/{0}/fold{1}/{0}_concatv2.h5'.format(head, val_fold)]
    else:
        model_weightList = [inference_weights_path + '/{0}/fold{1}/{0}_3channel.h5'.format(head, val_fold)]
    npy_path = "/results_npy/singlehead/{}/fold{}/".format(head, val_fold)
    # ----------------------------------------------------------
    tabular_data = pd.read_csv(config.csv)
    ds_dict = utils.get_split_fold(tabular_data, label_col=label_col, fold_col=fold_col, val_fold=val_fold)
    for ds in ds_list:
        print()
        print("-------------------------------------------------------------------------------------------------------")
        print("Current dataset: ",ds)
        # loop through all modalities
        if n_channel!=3:
            emb_m = []
            for m in range(len(modalities)):
                print("Current modality: ", modalities[m])
                epochs = 1
                while epochs <= max_epochs:
                    val_dl = utils.DataLoader3D(data=ds_dict[ds], batch_size=val_batch_size, patch_size=patch_size,
                                                num_threads_in_multithreaded=1, crop_type="center", seed_for_shuffle=5243,
                                                margins=(0, 0, 0), return_incomplete=True, shuffle=False, infinite=False)
                    print("Epochs: {} -----------------------".format(epochs))

                    # Validation-----------------------------------------------
                    if len(ds_dict[ds]['labels'])%val_batch_size == 0:
                        val_total_batch = int(round(len(ds_dict[ds]['labels'])/val_batch_size))
                    else:
                        val_total_batch = int(len(ds_dict[ds]['labels'])/val_batch_size)+1
                        for j in tqdm(range(val_total_batch)):
                            val_batch = next(val_dl)
                            imgs = np.swapaxes(val_batch["data"], 1, -1)
                            # standardization
                            imgs = utils.standardization(imgs, (1, 2, 3))
                            # min max normalization
                            if mm_norm==True:
                                imgs = utils.min_max_norm(imgs, (1, 2, 3))
                            labels = val_batch["lbl"]
                            model.load_weights(model_weightList[m])
                            # embedding
                            val_output = test_fn(model, imgs[...,m], labels, l2_norm=False)
                            val_pred = val_output["val_pred"]
                            val_pred_prob = val_output["val_pred_prob"]
                            val_emb = val_output["val_emb"]
                            if j == 0:
                                #----------
                                val_pred_metric = np.zeros_like(np.expand_dims(val_pred, axis=-1))
                                val_pred_prob_metric = np.zeros_like(np.expand_dims(val_pred_prob, axis=-1))
                                val_emb_metric = np.zeros_like(val_emb)
                            #-------------
                            val_pred_metric = np.vstack((val_pred_metric, np.expand_dims(val_pred, axis=-1)))
                            val_pred_prob_metric = np.vstack((val_pred_prob_metric, np.expand_dims(val_pred_prob, axis=-1)))
                            val_emb_metric = np.vstack((val_emb_metric, val_emb))
                            #------
                        val_multi_lbl_metric = np.squeeze(np.array(ds_dict[ds]['labels']))
                        val_pred_metric = np.squeeze(val_pred_metric[val_batch_size:])
                        val_pred_prob_metric = np.squeeze(val_pred_prob_metric[val_batch_size:])
                        val_emb_metric = np.squeeze(val_emb_metric[val_batch_size:])
                        # l2_norm
                        l2_val_emb_metric = tf.math.l2_normalize(val_emb_metric, axis=1).numpy()
                        # partial
                        val_metric_ = utils_metric.Metric(l2_val_emb_metric, k, None, val_multi_lbl_metric, max_match=R)
                        val_metric_dict = val_metric_.get_metric()
                        # save npy
                        embedding_npy = npy_path+"{}_embedding.npy".format(modalities[m])
                        prob_npy = npy_path+"{}_prob.npy".format(modalities[m])
                        np.save(embedding_npy, l2_val_emb_metric)
                        np.save(prob_npy, val_pred_prob_metric)
                        if epochs==1:
                            print("Validation_m:")
                            target_names = ['non-cancer', 'cancer']
                            print(classification_report(y_true=val_multi_lbl_metric, y_pred=val_pred_metric, target_names=target_names, zero_division=1))
                            # embedding
                            print("Validation Metrics")
                            print("precision@{0} recall@{0} r_precision MAP@R".format(k))
                            f = '{:10.3f}{:9.3f}{:10.3f}{:8.3f}'
                            print(f.format(val_metric_dict['precision@{}'.format(k)], val_metric_dict['recall@{}'.format(k)],
                                           val_metric_dict['r_precision'], val_metric_dict['map@r']))
                            val_auc_m = utils.auc_score(val_multi_lbl_metric, val_pred_prob_metric)
                            print("AUC Scores: {}".format(val_auc_m))
                        # Logging----------------------------------------------------------------
                        # Plots
                        # Embedding plot-------------------------------------
                        val_title = "{} embedding PCA".format(ds)
                        val_emb = utils_vis.pca_plot_2D(l2_val_emb_metric, val_multi_lbl_metric, n_class=2,
                                                        title=val_title, show=False)
                        wandb.log({"{} {} Embedding".format(ds, modalities[m]): wandb.Image(val_emb)})

                        plt.clf()
                        plt.close('all')
                        print()
                        epochs += 1
                emb_m.append(val_emb_metric)
            if len(model_weightList) == 4:
                print()
                print("Modality Concatenate Output")
                concat_emb = np.concatenate(emb_m, axis=-1)
                print(concat_emb.shape)
                model_head.load_weights(model_weightList[-1])
                concat_model = test_fn(model_head, concat_emb, val_multi_lbl_metric, l2_norm=True)
                concat_pred = np.stack(concat_model["val_pred"], axis=-1)
                concat_pred_prob = np.stack(concat_model["val_pred_prob"], axis=-1)
                concat_model_emb = concat_model["val_emb"]
                val_metric_ = utils_metric.Metric(concat_model_emb, k, None, val_multi_lbl_metric, max_match=R)
                val_metric_dict = val_metric_.get_metric()
                # save npy
                concat_embedding_npy = npy_path + "concat_embedding.npy"
                concat_prob_npy = npy_path + "concat_prob.npy"
                lbl_npy = npy_path+"lbl.npy"
                np.save(concat_embedding_npy, concat_model_emb)
                np.save(concat_prob_npy, concat_pred_prob)
                np.save(lbl_npy, val_multi_lbl_metric)
                print(classification_report(y_true=val_multi_lbl_metric, y_pred=concat_pred, target_names=target_names, zero_division=1))
                print("Validation Metrics")
                print("precision@{0} recall@{0} r_precision MAP@R".format(k))
                f = '{:10.3f}{:9.3f}{:10.3f}{:8.3f}'
                print(f.format(val_metric_dict['precision@{}'.format(k)], val_metric_dict['recall@{}'.format(k)],
                               val_metric_dict['r_precision'], val_metric_dict['map@r']))
                concat_auc_m = utils.auc_score(val_multi_lbl_metric, concat_pred_prob)
                print("AUC Scores: {}".format(concat_auc_m))
                # Plot
                val_title = "{} embedding PCA".format(ds)
                val_emb = utils_vis.pca_plot_2D(concat_model_emb, val_multi_lbl_metric, n_class=2,
                                                title=val_title, show=False)
                wandb.log({"{} Concat Embedding".format(ds): wandb.Image(val_emb)})
                plt.clf()
                plt.close('all')
        else:
            print("3channel Fold: ", config.val_fold)
            epochs = 1
            while epochs <= max_epochs:
                val_dl = utils.DataLoader3D(data=ds_dict[ds], batch_size=val_batch_size, patch_size=patch_size,
                                            num_threads_in_multithreaded=1, crop_type="center", seed_for_shuffle=5243,
                                            margins=(0, 0, 0), return_incomplete=True, shuffle=False, infinite=False)
                print("Epochs: {} -----------------------".format(epochs))

                # Validation-----------------------------------------------
                if len(ds_dict[ds]['labels']) % val_batch_size == 0:
                    val_total_batch = int(round(len(ds_dict[ds]['labels']) / val_batch_size))
                else:
                    val_total_batch = int(len(ds_dict[ds]['labels']) / val_batch_size) + 1
                    for j in tqdm(range(val_total_batch)):
                        val_batch = next(val_dl)
                        imgs = np.swapaxes(val_batch["data"], 1, -1)
                        # standardization
                        imgs = utils.standardization(imgs, (1, 2, 3))
                        # min max normalization
                        if mm_norm == True:
                            imgs = utils.min_max_norm(imgs, (1, 2, 3))

                        labels = val_batch["lbl"]
                        model.load_weights(model_weightList[0])
                        # embedding
                        val_output = test_fn(model, imgs, labels, l2_norm=True)
                        val_pred = np.stack(val_output["val_pred"], axis=-1)
                        val_pred_prob = np.stack(val_output["val_pred_prob"], axis=-1)
                        val_emb = val_output["val_emb"]
                        if j == 0:
                            # ----------
                            val_pred_metric = np.zeros_like(np.expand_dims(val_pred, axis=-1))
                            val_pred_prob_metric = np.zeros_like(np.expand_dims(val_pred_prob, axis=-1))
                            val_emb_metric = np.zeros_like(val_emb)
                        # -------------
                        val_pred_metric = np.vstack((val_pred_metric, np.expand_dims(val_pred, axis=-1)))
                        val_pred_prob_metric = np.vstack((val_pred_prob_metric, np.expand_dims(val_pred_prob, axis=-1)))
                        val_emb_metric = np.vstack((val_emb_metric, val_emb))
                        # ------
                    val_multi_lbl_metric = np.squeeze(np.array(ds_dict[ds]['labels']))
                    val_pred_metric = np.squeeze(val_pred_metric[val_batch_size:])
                    val_pred_prob_metric = np.squeeze(val_pred_prob_metric[val_batch_size:])
                    val_emb_metric = np.squeeze(val_emb_metric[val_batch_size:])
                    # partial
                    val_metric_ = utils_metric.Metric(val_emb_metric, k, None, val_multi_lbl_metric, max_match=R)
                    val_metric_dict = val_metric_.get_metric()
                    # save npy
                    channel_embedding_npy = npy_path + "3channel_embedding.npy"
                    channel_prob_npy = npy_path + "3channel_prob.npy"
                    np.save(channel_embedding_npy, val_emb_metric)
                    np.save(channel_prob_npy, val_pred_prob_metric)
                    if epochs == 1:
                        print("Validation_m:")
                        target_names = ['non-cancer', 'cancer']
                        print(classification_report(y_true=val_multi_lbl_metric, y_pred=val_pred_metric, target_names=target_names, zero_division=1))
                        # embedding
                        print("Validation Metrics")
                        print("precision@{0} recall@{0} r_precision MAP@R".format(k))
                        f = '{:10.3f}{:9.3f}{:10.3f}{:8.3f}'
                        print(f.format(val_metric_dict['precision@{}'.format(k)], val_metric_dict['recall@{}'.format(k)],
                                       val_metric_dict['r_precision'], val_metric_dict['map@r']))
                        val_auc_m = utils.auc_score(val_multi_lbl_metric, val_pred_prob_metric)
                        print("AUC Scores: {}".format(val_auc_m))
                    # Logging----------------------------------------------------------------
                    # Plots
                    # Embedding plot-------------------------------------
                    val_title = "{} embedding PCA".format(ds)
                    val_emb = utils_vis.pca_plot_2D(val_emb_metric, val_multi_lbl_metric, n_class=2,
                                                    title=val_title, show=False)
                    wandb.log({"{} Embedding 3Channel".format(ds): wandb.Image(val_emb)})

                    plt.clf()
                    plt.close('all')
                    print()
                    epochs += 1
#%%


if __name__ == '__main__':
    model = CustomModel(backbone, weights_decay=weights_decay,
                        margin=margin, logits_scale=logits_scale,
                        load_weights=None, freeze=False, head=config.head)
    if n_channel!=3:
        model.build([(None, depth, 144, 144, 1), (None, 1)])
        model_head = CustomHead(margin=margin, logits_scale=logits_scale, head=config.head)
        model_head.build([(None, config.emb_dim * 3), (None, 1)])
        # model_head.summary()
    else:
        model.build([(None, depth, 144, 144, 3), (None, 1)])
        model_head = None
    # model.summary()
    main(model, model_head, val_fold=config.val_fold)
    sys.exit(0)

