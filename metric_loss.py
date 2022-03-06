import tensorflow as tf
import math
K = tf.keras.backend

hn = 'he_normal'

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask, tf.float32)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.cast(mask_anchor_positive, tf.float32)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.cast(mask_anchor_negative, tf.float32)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


class ArcFaceLoss(tf.keras.layers.Layer):
    """
    ArcMarginPenaltyLogists
    Adapted from: https://github.com/peteryuX/arcface-tf2/blob/master/modules/layers.py
    """
    def __init__(self, num_classes, margin=0.5, logits_scale=64, **kwargs):
        super(ArcFaceLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logits_scale = logits_scale
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.sin_ms = tf.identity(math.sin(math.pi - self.margin), name='sin_ms')
        self.mm = tf.multiply(self.sin_ms, self.margin, name='mm')

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights", shape=[int(input_shape[-1]), self.num_classes],
            initializer='he_normal', trainable=True) # [B, n_dim, n_class]

    def call(self, embds, labels, binary=False,
             easy_margin=False, training=True):
        labels = tf.squeeze(labels) # no extra dimension for labels
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd') # [B, n_dim]
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights') # [n_dim, n_class]
        # w.x
        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t') # [B, n_class]
        cos_t = tf.clip_by_value(cos_t, -1.0 + 1e-7, 1.0 - 1e-7)
        if training==True:
            # trigonomery, cos(t)^2 + sin(t)^2 = 1
            sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')
            # trigonometry, cos(t+m) = cos(t)cos(m)-sin(t)sin(m)
            cos_mt = tf.subtract(cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')
            if easy_margin==True:
                cos_mt = tf.where(cos_t > 0, cos_mt, cos_t)
            else:
                cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)
            if binary==False:
                mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                                  name='one_hot_mask')
                logits = tf.where(mask == 1., cos_mt, cos_t)
            else:
                logits = cos_mt
            logits = tf.multiply(logits, self.logits_scale, 'arcface_logits')
        else:
            logits = tf.multiply(cos_t, self.logits_scale, 'arcface_logits')
        return logits, cos_t


def dist_weighted_sampling(labels, embeddings, high_var_threshold=0.5, nonzero_loss_threshold=1.4, neg_multiplier=1):
    """
    Distance weighted sampling.
    # References
        - [sampling matters in deep embedding learning]
          (https://arxiv.org/abs/1706.07567)
    # Arguments:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multi-class integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        high_var_threshold: float. cutoff for high gradient variance.
        nonzero_loss_threshold: float. cutoff for non-zero loss zone.
        neg_multiplier: int, default=1. the multiplier to enlarger the negative and positive samples.
    Returns:
        a_indices: indices of anchors.
        anchors: sampled anchor embeddings.
        positives: sampled positive embeddings.
        negatives: sampled negative embeddings.
    """
    if not isinstance(neg_multiplier, int):
        raise ValueError("`neg_multiplier` must be an integer.")
    n = tf.size(labels)
    if not isinstance(embeddings, tf.Tensor):
        embeddings = tf.convert_to_tensor(embeddings)
    d = embeddings.shape[1]

    distances = _pairwise_distances(embeddings, squared=False)
    # cut off to void high variance.
    distances = tf.maximum(distances, high_var_threshold)

    # subtract max(log(distance)) for stability
    log_weights = (2 - d) * tf.math.log(distances + 1e-16) - 0.5 * (d - 3) * tf.math.log(1 + 1e-16 - 0.25 * (distances**2))
    weights = tf.exp(log_weights - tf.reduce_max(log_weights))

    # sample only negative examples by setting weights of the same class examples to 0.
    lshape = tf.shape(labels)
    assert lshape.shape == 1
    labels = tf.reshape(labels, [lshape[0], 1])
    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)
    mask = tf.cast(adjacency_not, tf.float32)

    # number of negative/positive samples to sampling per sample.
    # For imbalanced data, this sampling method can be a sample weighted method.
    adjacency_ex = tf.cast(adjacency, tf.int32) - tf.linalg.diag(tf.ones(n, dtype=tf.int32))
    m = tf.reduce_sum(adjacency_ex, axis=1)
    if tf.reduce_min(m) == 0:
        m = tf.linalg.diag(tf.cast(tf.equal(m,0), tf.int32))
        adjacency_ex += m
    k = tf.maximum(tf.reduce_max(m),1) * neg_multiplier

    pos_weights = tf.cast(adjacency_ex, tf.float32)

    weights = weights * mask * tf.cast(distances < nonzero_loss_threshold, tf.float32)
    weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-16)

    #  anchors indices
    a_indices = tf.reshape(tf.range(n), (-1,1))
    a_indices = tf.tile(a_indices, [1, k])
    a_indices = tf.reshape(a_indices, (-1,))

    # negative sampling
    def neg_sampling(i):
        s = tf.squeeze(tf.random.categorical(tf.math.log(tf.expand_dims(weights[i] + 1e-16, axis=0)), k, dtype=tf.int32), axis=0)
        return s

    n_indices = tf.map_fn(neg_sampling, tf.range(n), dtype=tf.int32)
    n_indices = tf.reshape(n_indices, (-1,))

    # postive samping
    def pos_sampling(i):
        s = tf.squeeze(tf.random.categorical(tf.math.log(tf.expand_dims(pos_weights[i] + 1e-16, axis=0)), k, dtype=tf.int32), axis=0)
        return s

    p_indices = tf.map_fn(pos_sampling, tf.range(n), fn_output_signature=tf.int32)
    p_indices = tf.reshape(p_indices, (-1,))

    anchors = tf.gather(embeddings, a_indices, name='gather_anchors')
    positives = tf.gather(embeddings, p_indices, name='gather_pos')
    negatives = tf.gather(embeddings, n_indices, name='gather_neg')

    return a_indices, anchors, positives, negatives


def margin_based_loss(labels, embeddings, beta_in=1.0, margin=0.2, nu=0.0, high_var_threshold=0.5,
                      nonzero_loss_threshold=1.4, neg_multiplier=1):
    """
    Computes the margin base loss.
    # References
        - [sampling matters in deep embedding learning]
          (https://arxiv.org/abs/1706.07567)
    Args:
        labels: 1-D. tf.int32 `Tensor` with shape [batch_size] of multi-class integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        beta_in: float,int or 1-D, float `Tensor` with shape [labels_size] of multi-class boundary parameters.
        margin: Float, margin term in the loss function.
        nu: float. Regularization parameter for beta.
        high_var_threshold: float. cutoff for high gradient variance.
        nonzero_loss_threshold: float. cutoff for non-zero loss zone.
        neg_multiplier: int, default=1. the multiplier to enlarger the negative and positive samples.
    Returns:
        margin_based_Loss: tf.float32 scalar
    """

    a_indices, anchors, positives, negatives = dist_weighted_sampling(labels,
                                                                      embeddings,
                                                                      high_var_threshold=high_var_threshold,
                                                                      nonzero_loss_threshold=nonzero_loss_threshold,
                                                                      neg_multiplier=neg_multiplier)
    if isinstance(beta_in, (float,int)):
        beta = beta_in
        beta_reg_loss = 0.0
    else:
        if isinstance(beta_in, tf.Tensor):
            assert tf.shape(beta_in).shape == 1
            k = tf.size(a_indices) / tf.size(labels)
            k = tf.cast(k, tf.int32)
            beta = tf.reshape(beta_in, (-1, 1))
            beta = tf.tile(beta, [1, k])
            beta = tf.reshape(beta, (-1,))
            beta_reg_loss = tf.reduce_sum(beta) * nu
        else:
            raise ValueError("`beta_in` must be one of [float, int, tf.Tensor].")

    d_ap = tf.sqrt(tf.reduce_sum(tf.square(positives - anchors), axis=1) + 1e-16)
    d_an = tf.sqrt(tf.reduce_sum(tf.square(negatives - anchors), axis=1) + 1e-16)

    pos_loss = tf.maximum(margin + d_ap - beta, 0)
    neg_loss = tf.maximum(margin + beta - d_an, 0)

    pair_cnt = tf.cast(tf.size(a_indices), tf.float32)

    # normalize based on the number of pairs
    loss = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss) + beta_reg_loss) / pair_cnt
    return loss


def distance_weighted_triplet_loss(labels, embeddings, margin=1.0, squared=False, high_var_threshold=0.5,
                                   nonzero_loss_threshold=1.4, neg_multiplier=1):
    """distance weighted sampling + triplet loss
    Args:
        labels: 1-D. tf.int32 `Tensor` with shape [batch_size] of multi-class integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        margin: Float, margin term in the loss function.
        squared: Boolean, whether or not to square the triplet distances.
        nu: float. Regularization parameter for beta.
        high_var_threshold: float. cutoff for high gradient variance.
        nonzero_loss_threshold: float. cutoff for non-zero loss zone.
        neg_multiplier: int, default=1. the multiplier to enlarger the negative and positive samples.
    Returns:
        triplet_loss: tf.float32 scalar
    """
    a_indices, anchors, positives, negatives = dist_weighted_sampling(labels,
                                                                      embeddings,
                                                                      high_var_threshold=high_var_threshold,
                                                                      nonzero_loss_threshold=nonzero_loss_threshold,
                                                                      neg_multiplier=neg_multiplier)

    d_ap = tf.reduce_sum(tf.square(positives - anchors), axis=1)
    d_an = tf.reduce_sum(tf.square(negatives - anchors), axis=1)
    if not squared:
        d_ap = K.sqrt(d_ap + 1e-16)
        d_an = K.sqrt(d_an + 1e-16)

    loss = tf.maximum(d_ap - d_an + margin, 0)
    loss = tf.reduce_mean(loss)
    return loss


class AngularSoftMax(tf.keras.layers.Layer):
    def __init__(self, num_classes, batch_size, logits_scale=1.0, use_bias=True, **kwargs):
        super(AngularSoftMax, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.logits_scale = logits_scale
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights", shape=[int(input_shape[-1]), self.num_classes],
            initializer='he_normal', trainable=True) # [n_dim, n_class]
        if self.use_bias:
            self.b = self.add_weight(
                name="bias", shape=[self.batch_size, self.num_classes],
                initializer='he_normal', trainable=True) # [batch, n_class]

    def call(self, embds, labels, norm=True):
        # if the input for embedding is already normalised
        if norm:
            normed_embds = embds
        else:
            normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd') # [B, n_dim]
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights') # [n_dim, n_class]
        # w.x
        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t') # [B, n_class]
        cos_t = tf.clip_by_value(cos_t, -1.0, 1.0)
        if self.use_bias:
            cos_t += self.b
        logits = tf.multiply(cos_t, self.logits_scale)
        return logits, cos_t


class CosFaceLoss(tf.keras.layers.Layer):
    def __init__(self, num_classes, batch_size, margin=0.2, logits_scale=1.0, use_bias=True, **kwargs):
        super(CosFaceLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.margin = margin
        self.logits_scale = logits_scale
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights", shape=[int(input_shape[-1]), self.num_classes],
            initializer='he_normal', trainable=True) # [n_dim, n_class]
        if self.use_bias:
            self.b = self.add_weight(
                name="bias", shape=[self.batch_size, self.num_classes],
                initializer='he_normal', trainable=True) # [batch, n_class]

    def call(self, embds, labels, norm=True, binary=False):
        # if the input for embedding is already normalised
        if norm:
            normed_embds = embds
        else:
            normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')# [B, n_dim]
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')# [n_dim, n_class]
        # w.x
        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')# [B, n_class]
        cos_t = tf.clip_by_value(cos_t, -1.0+1e-7, 1.0-1e-7)
        if binary==False:
            mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                              name='one_hot_mask')
            cos_mt = tf.where(mask==1, cos_t-self.margin, cos_t)
        else:
            cos_mt = tf.subtract(cos_t, self.margin)
        if self.use_bias:
            cos_mt += self.b
        logits = tf.multiply(cos_mt, self.logits_scale)
        return logits, cos_t


class SubCenterArcFaceLoss(tf.keras.layers.Layer):
    """
    ArcMarginPenaltyLogists
    Adapted from: https://github.com/peteryuX/arcface-tf2/blob/master/modules/layers.py
    """
    def __init__(self, num_classes, margin=0.5, logits_scale=64, k=3, **kwargs):
        super(SubCenterArcFaceLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logits_scale = logits_scale
        self.k = k

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weights", shape=[int(input_shape[-1]), self.num_classes, self.k],
            initializer='he_normal', trainable=True) # [B, n_dim, n_class, k]
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels, norm=True, easy_margin=False, binary=False):
        labels = tf.squeeze(labels) # no extra dimension for labels
        # if the input for embedding is already normalised
        if norm:
            normed_embds = embds
        else:
            normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd') # [B, n_dim]
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights') # [n_dim, n_class, k]
        # w.x, subclass-wise consine similarity
        cos_t = tf.einsum('bd,dck->bck', normed_embds, normed_w, name='cos_t') # [B, n_class, k]
        # cos_t = tf.clip_by_value(cos_t, -1.0+1e-7, 1.0-1e-7) # cosine in the range of [-1, 1]
        # class-wise cosine similarity
        cos_t = tf.reduce_max(cos_t, axis=-1)
        # trigonomery, cos(t)^2 + sin(t)^2 = 1
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')
        # trigonometry, cos(t+m) = cos(t)cos(m)-sin(t)sin(m)
        cos_mt = tf.subtract(cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')
        if easy_margin==True:
            cos_mt = tf.where(cos_t > 0, cos_mt, cos_t)
        else:
            cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)
        if binary==False:
            mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                              name='one_hot_mask')
            logits = tf.where(mask == 1., cos_mt, cos_t)
        else:
            logits = cos_mt
        logits = tf.multiply(logits, self.logits_scale, 'arcface_logits')
        return logits, cos_t


