import sys
sys.path.append('/home/kevinteng/Desktop/Masterarbeit')
from classification_models_3D.keras import Classifiers
import numpy as np
import tensorflow as tf
import os
import wandb
import utils, utils_vis, metric_loss, utils_metric
import pandas as pd
import gc
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
logits_scale = eval(config.logits_scale)
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
loss_fn = config.head

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
    print()
    print("Fold: {}".format(val_fold))
    # initialize saving criterion--------------------------------
    k = config.k
    R = config.R
    # ds_list = ['val_ds','test_ds']
    ds_list = ['test_ds']
    modalities = ["t2", "dwi", "adc"]
    inference_weights_path = "/inference_weights/multihead"
    if n_channel!=3:
        model_weightList = [inference_weights_path + '/{0}/fold{1}/{0}_t2.h5'.format(head, val_fold),
                            inference_weights_path + '/{0}/fold{1}/{0}_dwi.h5'.format(head, val_fold),
                            inference_weights_path + '/{0}/fold{1}/{0}_adc.h5'.format(head, val_fold),
                            inference_weights_path + '/{0}/fold{1}/{0}_concat.h5'.format(head, val_fold)]
    else:
        model_weightList = [inference_weights_path + '/{0}/fold{1}/{0}_3channel.h5'.format(head, val_fold)]

    npy_path = "/results_npy/multihead/{}/fold{}/".format(head, val_fold)
    # ----------------------------------------------------------
    tabular_data = pd.read_csv(config.csv)
    ds_dict = utils.multi_get_split_fold(tabular_data, label_col=label_col, fold_col=fold_col, val_fold=val_fold)
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
                    val_dl = utils.DataLoader3D_multi(data=ds_dict[ds], batch_size=val_batch_size, patch_size=patch_size,
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
                            labels = np.stack([val_batch["prostatitis"], val_batch["maglinant"], val_batch["ggg"],
                                               val_batch["risk"], val_batch["ggg_s"], val_batch["tumour"]], axis=-1)
                            model.load_weights(model_weightList[m])
                            # embedding
                            val_output = test_fn(model, imgs[...,m], labels, l2_norm=False)
                            val_pred = np.stack(val_output["val_pred"], axis=-1)
                            val_pred_prob = np.stack(val_output["val_pred_prob"], axis=-1)
                            val_emb = val_output["val_emb"]
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
                            #------
                        val_multi_lbl_metric = np.squeeze(val_multi_lbl_metric[val_batch_size:])
                        val_pred_metric = np.squeeze(val_pred_metric[val_batch_size:])
                        val_pred_prob_metric = np.squeeze(val_pred_prob_metric[val_batch_size:])
                        val_emb_metric = np.squeeze(val_emb_metric[val_batch_size:])
                        # l2_norm
                        l2_val_emb_metric = tf.math.l2_normalize(val_emb_metric, axis=1).numpy()
                        # partial
                        val_metric_ = utils_metric.Metric(l2_val_emb_metric, k, None, val_multi_lbl_metric[:,1], max_match=R)
                        val_metric_dict = val_metric_.get_metric()
                        # save npy
                        embedding_npy = npy_path+"{}_embedding.npy".format(modalities[m])
                        prob_npy = npy_path+"{}_prob.npy".format(modalities[m])
                        np.save(embedding_npy, l2_val_emb_metric)
                        np.save(prob_npy, val_pred_prob_metric[:,1])
                        if epochs==1:
                            print("Validation_m:")
                            target_names = ['non-cancer', 'cancer']
                            print(classification_report(y_true=val_multi_lbl_metric[:,1], y_pred=val_pred_metric[:,1], target_names=target_names, zero_division=1))
                            # embedding
                            print("Validation Metrics")
                            print("precision@{0} recall@{0} r_precision MAP@R".format(k))
                            f = '{:10.3f}{:9.3f}{:10.3f}{:8.3f}'
                            print(f.format(val_metric_dict['precision@{}'.format(k)], val_metric_dict['recall@{}'.format(k)],
                                           val_metric_dict['r_precision'], val_metric_dict['map@r']))
                            val_auc_m = utils.auc_score(val_multi_lbl_metric[:,1], val_pred_prob_metric[:,1])
                            print("AUC Scores: {}".format(val_auc_m))
                        # Logging----------------------------------------------------------------
                        # Plots
                        # Embedding plot-------------------------------------
                        val_title = "{} embedding PCA".format(ds)
                        val_emb = utils_vis.pca_plot_2D(l2_val_emb_metric, val_multi_lbl_metric[:,1], n_class=2,
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
                val_metric_ = utils_metric.Metric(concat_model_emb, k, None, val_multi_lbl_metric[:,1], max_match=R)
                val_metric_dict = val_metric_.get_metric()
                # save npy
                concat_embedding_npy = npy_path + "concat_embedding.npy"
                concat_prob_npy = npy_path + "concat_prob.npy"
                lbl_npy = npy_path+"lbl.npy"
                np.save(concat_embedding_npy, concat_model_emb)
                np.save(concat_prob_npy, concat_pred_prob[:,1])
                np.save(lbl_npy, val_multi_lbl_metric[:,1])
                print(classification_report(y_true=val_multi_lbl_metric[:,1], y_pred=concat_pred[:,1], target_names=target_names, zero_division=1))
                print("Validation Metrics")
                print("precision@{0} recall@{0} r_precision MAP@R".format(k))
                f = '{:10.3f}{:9.3f}{:10.3f}{:8.3f}'
                print(f.format(val_metric_dict['precision@{}'.format(k)], val_metric_dict['recall@{}'.format(k)],
                               val_metric_dict['r_precision'], val_metric_dict['map@r']))
                concat_auc_m = utils.auc_score(val_multi_lbl_metric[:,1], concat_pred_prob[:,1])
                print("AUC Scores: {}".format(concat_auc_m))
                # Plot
                val_title = "{} embedding PCA".format(ds)
                val_emb = utils_vis.pca_plot_2D(concat_model_emb, val_multi_lbl_metric[:,1], n_class=2,
                                                title=val_title, show=False)
                wandb.log({"{} Concat Embedding".format(ds): wandb.Image(val_emb)})
                plt.clf()
                plt.close('all')
        else:
            print("3channel Fold: ", config.val_fold)
            epochs = 1
            while epochs <= max_epochs:
                val_dl = utils.DataLoader3D_multi(data=ds_dict[ds], batch_size=val_batch_size, patch_size=patch_size,
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

                        labels = np.stack([val_batch["prostatitis"], val_batch["maglinant"], val_batch["ggg"],
                                           val_batch["risk"], val_batch["ggg_s"], val_batch["tumour"]], axis=-1)
                        model.load_weights(model_weightList[0])
                        # embedding
                        val_output = test_fn(model, imgs, labels, l2_norm=True)
                        val_pred = np.stack(val_output["val_pred"], axis=-1)
                        val_pred_prob = np.stack(val_output["val_pred_prob"], axis=-1)
                        val_emb = val_output["val_emb"]
                        if j == 0:
                            # ----------
                            val_multi_lbl_metric = np.zeros_like(labels)
                            val_pred_metric = np.zeros_like(val_pred)
                            val_pred_prob_metric = np.zeros_like(val_pred_prob)
                            val_emb_metric = np.zeros_like(val_emb)
                        # -------------
                        val_multi_lbl_metric = np.vstack((val_multi_lbl_metric, labels))
                        val_pred_metric = np.vstack((val_pred_metric, val_pred))
                        val_pred_prob_metric = np.vstack((val_pred_prob_metric, val_pred_prob))
                        val_emb_metric = np.vstack((val_emb_metric, val_emb))
                        # ------
                    val_multi_lbl_metric = np.squeeze(val_multi_lbl_metric[val_batch_size:])
                    val_pred_metric = np.squeeze(val_pred_metric[val_batch_size:])
                    val_pred_prob_metric = np.squeeze(val_pred_prob_metric[val_batch_size:])
                    val_emb_metric = np.squeeze(val_emb_metric[val_batch_size:])
                    # partial
                    # malignancy
                    val_metric_m = utils_metric.Metric(val_emb_metric, k, None, val_multi_lbl_metric[:,5], max_match=R)
                    val_metric_dict_m = val_metric_m.get_metric()
                    # tumour
                    val_metric_t = utils_metric.Metric(val_emb_metric, k, None, val_multi_lbl_metric[:,5], max_match=R)
                    val_metric_dict_t = val_metric_t.get_metric()
                    # prostatitis
                    val_metric_p = utils_metric.Metric(val_emb_metric, k, None, val_multi_lbl_metric[:,0], max_match=R)
                    val_metric_dict_p = val_metric_p.get_metric()
                    # save npy
                    channel_embedding_npy = npy_path + "3channel_embedding.npy"
                    channel_prob_npy = npy_path + "3channel_prob_all.npy"
                    lbl_npy = npy_path + "lbl_all.npy"
                    np.save(channel_embedding_npy, val_emb_metric)
                    np.save(channel_prob_npy, val_pred_prob_metric)
                    np.save(lbl_npy, val_multi_lbl_metric)
                    f = '{:10.3f}{:9.3f}{:10.3f}{:8.3f}'
                    if epochs == 1:
                        # # malignancy
                        # print("Validation_m:")
                        # target_names = ['non-cancer', 'cancer']
                        # print(classification_report(y_true=val_multi_lbl_metric[:,1], y_pred=val_pred_metric[:,1], target_names=target_names, zero_division=1))
                        # # embedding
                        # print("Validation Metrics")
                        # print("precision@{0} recall@{0} r_precision MAP@R".format(k))
                        # print(f.format(val_metric_dict_m['precision@{}'.format(k)], val_metric_dict_m['recall@{}'.format(k)],
                        #                val_metric_dict_m['r_precision'], val_metric_dict_m['map@r']))
                        # val_auc_m = utils.auc_score(val_multi_lbl_metric[:,1], val_pred_prob_metric[:,1])
                        # print("AUC Scores: {}".format(val_auc_m))
                        # print()
                        # tumour
                        print("Validation_t:")
                        target_names = ['non-tumour', 'tumour']
                        print(classification_report(y_true=val_multi_lbl_metric[:,5], y_pred=val_pred_metric[:,5], target_names=target_names, zero_division=1))
                        # embedding
                        print("Validation Metrics")
                        print("precision@{0} recall@{0} r_precision MAP@R".format(k))
                        print(f.format(val_metric_dict_t['precision@{}'.format(k)], val_metric_dict_t['recall@{}'.format(k)],
                                       val_metric_dict_t['r_precision'], val_metric_dict_t['map@r']))
                        val_auc_t = utils.auc_score(val_multi_lbl_metric[:,5], val_pred_prob_metric[:,2])
                        print("AUC Scores: {}".format(val_auc_t))
                        print()
                        # prostatitis
                        print("Validation_p:")
                        target_names = ['non-prostatitis', 'prostatitis']
                        print(classification_report(y_true=val_multi_lbl_metric[:,0], y_pred=val_pred_metric[:,0], target_names=target_names, zero_division=1))
                        # embedding
                        print("Validation Metrics")
                        print("precision@{0} recall@{0} r_precision MAP@R".format(k))
                        print(f.format(val_metric_dict_p['precision@{}'.format(k)], val_metric_dict_p['recall@{}'.format(k)],
                                       val_metric_dict_p['r_precision'], val_metric_dict_p['map@r']))
                        val_auc_p = utils.auc_score(val_multi_lbl_metric[:,0], val_pred_prob_metric[:,0])
                        print("AUC Scores: {}".format(val_auc_p))
                    # Logging----------------------------------------------------------------
                    # Plots
                    # Embedding plot-------------------------------------
                    # val_title = "{} embedding PCA".format(ds)
                    # val_emb = utils_vis.pca_plot_2D(val_emb_metric, val_multi_lbl_metric[:,1], n_class=2,
                    #                                 title=val_title, show=False)
                    # wandb.log({"{} Embedding 3Channel".format(ds): wandb.Image(val_emb)})
                    #
                    # plt.clf()
                    # plt.close('all')
                    print()
                    epochs += 1
#%%


if __name__ == '__main__':
    model = CustomModel(backbone, weights_decay=weights_decay,
                        margin=margin, logits_scale=logits_scale,
                        load_weights=None, freeze=False, head=config.head)
    if n_channel!=3:
        model.build([(None, depth, 144, 144, 1), (None, 6)])
        model_head = CustomHead(margin=margin, logits_scale=logits_scale, head=config.head)
        model_head.build([(None, config.emb_dim * 3), (None, 6)])
        # model_head.summary()
    else:
        model.build([(None, depth, 144, 144, 3), (None, 6)])
        model_head = None
    # model.summary()
    main(model, model_head, val_fold=config.val_fold)
    sys.exit(0)

