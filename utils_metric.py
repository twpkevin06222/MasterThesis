import numpy as np
import io
import tensorflow as tf
import faiss
from collections import defaultdict
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class Metric:
    def __init__(self, q_embedding, k, x, y, max_match=None):
        self.q_embedding = q_embedding  # model embedding output/query embedding
        self.k = k  # k-nearest neighbor
        self.x = x  # corresponding image
        self.y = y  # corresponding labels
        self.n_class = np.unique(y).size  # number of class w.r.t label
        self.n, self.n_dim = q_embedding.shape  # n => number of entries, n_dim => embedding vector
        self.max_match = max_match  # maximum comparison neighbor

        if max_match != None:
            assert self.max_match <= self.y.shape[0], "Max match should be lesser or equal to query size!"

        if self.max_match is None:
            # if the number of input labels is less than embedding dimension
            if self.y.shape[0] < self.n_dim:
                self.max_match = self.y.shape[0] - 1
            else:
                self.max_match = self.n_dim

        # needs q_embedding to be numpy array
        if tf.is_tensor(self.q_embedding):
            self.q_embedding = self.q_embedding.numpy()
        if self.y.ndim == 2:
            # print("[Note:] The dimension for 'y' should be 1!")
            self.y = np.squeeze(self.y)

    def get_class_idx_to_idxs(self):
        # map class idx to test idx
        class_idx_to_idxs = defaultdict(list)
        for y_idx, y in enumerate(self.y):
            class_idx_to_idxs[y].append(y_idx)
        return class_idx_to_idxs

    def nn_search(self, mode='faiss'):
        '''
        Nearest neighbour search for query embedding

        q_embedding: Query embedding
        k: Number of nearest neighbours
        mode: 'faiss' for facebook algorithm(essentially faster gram-matrix computation)
              'normal' for dot product to get gram-matrix

        return: Nearest neighbor indices with shape (number of entries, n_dim)
        '''
        assert mode == 'faiss' or mode == 'normal', "Please insert 'faiss' or 'normal for mode selection'"
        # references
        if mode == 'faiss':
            index = faiss.IndexFlatL2(self.n_dim)
            index.add(self.q_embedding)
            nn_dist, nn_indx = index.search(self.q_embedding, self.n_dim + 1)
        if mode == 'normal':
            # cosine similarity
            gram_matrix = tf.einsum("ae,be->ab", self.q_embedding, self.q_embedding)  # [B, B]
            # gram_matrix gives a similarity score matrix calculation hence
            # argsort ranks similarity to high similairy (diagonal)
            nn_indx = tf.argsort(gram_matrix, direction='DESCENDING')[:, :(self.max_match + 1)]
        return nn_indx

    def nn_array2list(self, nrows):
        '''
        Convert nearest neighbour array to image list for grid plot

        nn_array: Nearest neightbour array in 2D
        idx2img_array: An array that maps idices to corresponding image
        k: k-th number of nearest neighbor

        return: list of images corresponding to the indices of nearest neighbours
        '''
        nn_array = self.nn_search()[:, :self.k + 1]

        assert nn_array.ndim <= 2, "Only 2D array are accepted!"
        img_list = []
        # loop through each row of the nn_array
        for i in range(nrows):
            img_idx_row = nn_array[i]
            for idx in img_idx_row:
                img_list.append(self.x[idx])
        return img_list

    def grid_plot_nn(self, nrows, figsize=(10, 10), axes_pad=0.05
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
        ncols = self.k + 1
        img_list = self.nn_array2list(nrows)

        assert type(img_list) == list, 'Please input img_list as list'

        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=axes_pad)
        nimgs = nrows * ncols
        for steps, (ax, im) in enumerate(zip(grid, img_list)):
            for i in range(0, nimgs, ncols):
                ax.imshow(np.squeeze(im), cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
        plt.show()

    def get_confusion_matrix(self):
        # compute confusion matrix for the nearest neighbor
        confusion_matrix = np.zeros((self.n_class, self.n_class))
        # For each class.
        for class_idx in range(self.n_class):
            # example_idxs = self.get_class_idx_to_idxs()[class_idx]
            example_idxs = np.where(self.y==class_idx)[0]
            for y_idx in example_idxs:
                # And count the classes of its near neighbours.
                # does not include the diagonal itself
                for nn_idx in self.nn_search()[y_idx][1:self.max_match+1]:
                    nn_class_idx = self.y[nn_idx]
                    confusion_matrix[class_idx, nn_class_idx] += 1
        return confusion_matrix

    def plot_confusion_matrix(self, plot=True):
        # plot confusion matrix
        cm = self.get_confusion_matrix()
        labels = ['{}'.format(i) for i in range(self.n_class)]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
        if plot:
            plt.show()
        return fig

    def get_metric(self):
        '''
        Calculate the metrics for a given nearest neighbor array
        Instead of batching and looping, load all input data
        # cleaner implementation!

        nn_array: nearest neighbor array
        k: number of nearest neighbor

        return: dictionary of metrics
        '''
        max_match = self.max_match
        # search nearest neighbour
        if max_match == self.n_dim:
            mode = 'faiss'
        else:
            # search neighbour dimension is equal to embedding dimension
            mode = 'normal'
        nn_array = self.nn_search(mode)
        # map indices to labels
        query = tf.gather(self.y, tf.range(0, self.y.shape[0]))
        query_broadcast = tf.broadcast_to(tf.expand_dims(query, 1), nn_array[:, 1:].shape)
        total_relevance_mask = tf.cast(tf.equal(query_broadcast, tf.gather(self.y, nn_array[:, 1:])),
                                       dtype=tf.float32)  # [B, query]

        # true pos by masking up till k
        hit = tf.squeeze(tf.reduce_sum(total_relevance_mask[:, :self.k], axis=1))
        # ------------------------------------------------
        # total TP by masking
        total_hit = tf.squeeze(tf.reduce_sum(total_relevance_mask, axis=1))

        # ------------------------------------------
        # Calculate metrics
        # precision@k=relevant document@k/retrived document
        precision = hit / self.k  # [B, ]
        # recall@k=>percentage of queries having at least one neighbor retrieved
        # in the first k results by taking min(hit, 1.0)
        recall = tf.where(hit >= 1.0, 1.0, 0.0)
        # r-precision
        r_precision = total_hit / max_match  # [B, ]
        # MAP@R = (1/R)*sum(p(i)*rel(i))
        # accumulate the relevant at k along axis 1 in the batch
        relevant_at_k = tf.math.cumsum(total_relevance_mask, axis=1)  # [B,]

        # k position for element wise division
        k_pos = tf.cast(tf.range(1, max_match + 1), dtype=tf.float32)  # (1,2,..., R)
        MAP_at_R = tf.reduce_sum((relevant_at_k * total_relevance_mask) / k_pos, axis=1) / max_match

        mean_precision = tf.reduce_mean(precision)
        mean_recall = tf.reduce_mean(recall)
        mean_r_precision = tf.reduce_mean(r_precision)
        mean_MAP_at_R = tf.reduce_mean(MAP_at_R)

        return {"precision@{}".format(self.k): mean_precision,
                "recall@{}".format(self.k): mean_recall,
                "r_precision": mean_r_precision,
                "map@r": mean_MAP_at_R,
                "raw_precision": precision,
                "raw_recall": recall,
                "raw_r_precision": r_precision,
                "raw_map": MAP_at_R
                }

    def print_metrics(self):
        """
        Retrieve metric dictionary and print
        """
        metric_dict = self.get_metric()
        print("precision@{0} recall@{0} r_precision MAP@R".format(self.k))
        f = '{:10.3f}{:9.3f}{:10.3f}{:8.3f}'
        print(f.format(metric_dict['precision@{}'.format(self.k)], metric_dict['recall@{}'.format(self.k)],
                       metric_dict['r_precision'], metric_dict['map@r']))

    def save_embedding(self, save_path, ver, margin):
        # exclude first column since its the query
        embedding = self.q_embedding
        np.savetxt(save_path + "{}_vecs_margin_{}.tsv".format(ver, margin), embedding, delimiter='\t')

        out_m = io.open(save_path + '{}_meta_margin_{}.tsv'.format(ver, margin), 'w', encoding='utf-8')
        # batch dataset for faster serialization
        for labels in self.y:
            out_m.write(str(labels) + "\n")
        out_m.close()

    def get_bad_rows(self, return_mode="row", summary="short", topk=5):
        """
        Retrieve the indices in the query where the nearest neighbor (k=1) does not match
        the query label
        @param return_mode: Mode of return:
                            "row" for bad row,
                            "array" for query bad array
        @param summary: {short, all}
                        "short" return 2 sample per classes where labels doesn't match for R@1
                        "all" return all examples where labels doesn't match for R@1
        @param topk: Top-k retrieval
        @return: 1-D array containing indices of labels that does not match the query at k=1
        """
        max_match = self.max_match
        if max_match == self.n_dim:
            mode = 'faiss'
        else:
            # search neighbour dimension is equal to embedding dimension
            mode = 'normal'
        nn_array = self.nn_search(mode)
        # map index to labels for query
        # query_labels = tf.gather(self.y, nn_array[:, 0]) #[B,B]
        # map index of nearest neigbor at k=1
        nn_labels = tf.gather(self.y, nn_array[:, 1])
        # compare nearest neighbor labels with query labels and cast bool=>binary
        compare = tf.cast(tf.equal(self.y, nn_labels), dtype=tf.float32)
        # retrieve indices where the labels doesnt match
        bad_match_idx = np.where(compare == 0.0)[0]
        bad_match_idx = np.squeeze(bad_match_idx)
        if return_mode=="row":
            return bad_match_idx
        if return_mode=="array":
            if summary=='short':
                # gather the label where the nearest neighbour had a bad match
                bad_label_list = tf.gather(self.y, bad_match_idx)
                label_idx = []
                for c in range(self.n_class):
                    # retrieve the indices of respective class
                    bad_label_idx = np.where(bad_label_list==c)[0]
                    # randomly pick 2 indices from the respective class
                    bad_class = np.random.choice(bad_label_idx, 2, replace=False).tolist()
                    label_idx.append([bad_match_idx[k] for k in bad_class])
                label_idx_flatten = sum(label_idx, [])
                bad_query_nn = []
                for idx in label_idx_flatten:
                    # retrieve query and top 5 nearest neighbour from the nearest neighbour array
                    bad_query_nn.append(nn_array[idx, :topk+1])
            else:
                bad_query_nn = []
                for idx in bad_match_idx:
                    bad_query_nn.append(nn_array[idx, :topk+1])
            return bad_query_nn

    def get_good_rows(self, return_mode="row", summary="short", topk=5):
        """
        Retrieve the indices in the query where the nearest neighbor (k=1) match
        the query label
        @param return_mode: Mode of return:
                            "row" for bad row,
                            "array" for query bad array
        @param summary: {short, all}
                        "short" return 2 sample per classes where labels doesn't match for R@1
                        "all" return all examples where labels doesn't match for R@1
        @param topk: Top-k retrieval
        @return: 1-D array containing indices of labels that does not match the query at k=1
        """
        max_match = self.max_match
        if max_match == self.n_dim:
            mode = 'faiss'
        else:
            # search neighbour dimension is equal to embedding dimension
            mode = 'normal'
        nn_array = self.nn_search(mode)
        # map index to labels for query
        # query_labels = tf.gather(self.y, nn_array[:, 0]) #[B,B]
        # map index of nearest neigbor at k=1
        nn_labels = tf.gather(self.y, nn_array[:, 1])
        # compare nearest neighbor labels with query labels and cast bool=>binary
        compare = tf.cast(tf.equal(self.y, nn_labels), dtype=tf.float32)
        # retrieve indices where the labels match
        good_match_idx = np.where(compare == 1.0)[0]
        good_match_idx = np.squeeze(good_match_idx)
        if return_mode=="row":
            return good_match_idx
        if return_mode=="array":
            if summary=='short':
                # gather the label where the nearest neighbour had a bad match
                good_label_list = tf.gather(self.y, good_match_idx)
                label_idx = []
                for c in range(self.n_class):
                    # retrieve the indices of respective class
                    good_label_idx = np.where(good_label_list==c)[0]
                    # randomly pick 2 indices from the respective class
                    good_class = np.random.choice(good_label_idx, 2, replace=False).tolist()
                    label_idx.append([good_match_idx[k] for k in good_class])
                label_idx_flatten = sum(label_idx, [])
                good_query_nn = []
                for idx in label_idx_flatten:
                    # retrieve query and top 5 nearest neighbour from the nearest neighbour array
                    good_query_nn.append(nn_array[idx, :topk+1])
            else:
                good_query_nn = []
                for idx in good_match_idx:
                    good_query_nn.append(nn_array[idx, :topk+1])
            return good_query_nn

class DistanceMetrics:
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.class_ = np.unique(self.labels)

    def _pairwise_distances(self, squared=False):
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
        dot_product = tf.matmul(self.embeddings, tf.transpose(self.embeddings))

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

    def _get_anchor_positive_triplet_mask(self):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(self.labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(self.labels, 0), tf.expand_dims(self.labels, 1))

        # Combine the two masks
        mask = tf.logical_and(indices_not_equal, labels_equal)

        return tf.cast(mask, tf.float32)

    def _get_anchor_negative_triplet_mask(self):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = tf.equal(tf.expand_dims(self.labels, 0), tf.expand_dims(self.labels, 1))

        mask = tf.logical_not(labels_equal)

        return tf.cast(mask, tf.float32)

    def intra_dist(self):
        """
        Calculate the intra class distance
        return: Dictionary{class: intra class distance}
        """
        pos_mask = self._get_anchor_positive_triplet_mask()
        # each row now only consisted of distance of its own class
        pos_dis = pos_mask * self._pairwise_distances()
        # align distance to the left
        pos_dis_squeeze = tf.reduce_mean(pos_dis, axis=1)
        intra_dict = {}
        for c in self.class_:
            # collect the coordinates for each labels==class
            label_idx = tf.where(self.labels == c)
            label_dis = tf.gather(pos_dis_squeeze, label_idx)
            intra_dict["{}".format(c)] = tf.reduce_mean(label_dis).numpy()
        return intra_dict

    def inter_dist(self):
        """
        Calculate the inter class distance
        return: Dictionary{class: inter class distance}
        """
        neg_mask = self._get_anchor_negative_triplet_mask()
        # each row now only consisted of distance of other than its own class
        neg_dis = neg_mask * self._pairwise_distances()
        # align distance to the left
        neg_dis_squeeze = tf.reduce_mean(neg_dis, axis=1)
        inter_dict = {}
        for c in self.class_:
            label_idx = tf.where(self.labels == c)
            not_label_dis = tf.gather(neg_dis_squeeze, label_idx)
            inter_dict["{}".format(c)] = tf.reduce_mean(not_label_dis).numpy()
        return inter_dict
