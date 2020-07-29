from nltk.probability import FreqDist
import logging
import sys

LOG_FORMAT = '%(asctime)s : %(name)s : %(message)s'
formatter = logging.Formatter(fmt=LOG_FORMAT)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
# configuring root logger
logger = logging.getLogger('LabelNameGenerator')
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


class LabelNameGenerator:
    def __init__(self, cluster_processed_df):
        self.cluster_processed_df = cluster_processed_df

    def _apply_positions(self, multi_word_key, n_gram):
        positions = []
        for word in multi_word_key:
           # print(n_gram)
            positions.append(n_gram.index(word))

        label = []
        for pos in sorted(positions):
            label.append(n_gram[pos])

        return tuple(label)

    # cluster key is list of sorted (topic) words
    def get_label_preferences(self, cluster_key):
        def exists(label_tree, key):
            return key in label_tree.nodes

        n_grams = self.cluster_processed_df[self.cluster_processed_df.labels_tree.apply(lambda tree: exists(tree, cluster_key))].n_grams_emb

        if len(n_grams) == 0:
            logger.info('Error: ' + cluster_key + ' does not exist in label graph.')
            return

        label_name_distr = FreqDist()
        for n_gram in n_grams:
            label = self._apply_positions(cluster_key, n_gram)
            label_name_distr[label] += 1

        return label_name_distr.most_common()

    def get_label(self, cluster_key):
        label_preferences = self.get_label_preferences(cluster_key)

        if len(label_preferences) == 0:
            logger.info('Label does not exist for ' + cluster_key)
            return

        return label_preferences[0][0]

