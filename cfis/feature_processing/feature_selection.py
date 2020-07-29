from feature_embedding_clustering import FeatureEmbeddingClustering
from _utils import get_tfidf_matrix, get_tfidf_words
import tempfile
import pickle
import os

class FeatureSelection:

    def __init__(self, max_features_frac=.20, embedding=None, num_embedding_passes=None):
        self.max_features_frac = max_features_frac
        self.labels_cluster_obj = None

        if embedding:
            leader_threshes = [0.8, 0.7, 0.6, 0.5]
            member_threshes = [0.7, 0.6, 0.5, 0.4]
            thresh_pairs = list(zip(leader_threshes, member_threshes))

            if num_embedding_passes and 0 < num_embedding_passes < 4:
                thresh_pairs = thresh_pairs[:num_embedding_passes]

            self.labels_cluster_obj = FeatureEmbeddingClustering(thresh_pairs, embedding)


    def select_features(self, feature_extracted_df):

        _, initial_features = get_tfidf_matrix(feature_extracted_df.n_grams, min_df=1, max_df=1.0)

        max_features = int(self.max_features_frac * len(feature_extracted_df)) if self.max_features_frac else None
        print('Max_features: ', max_features)

        if self.labels_cluster_obj:

            self.labels_cluster_obj.fit(feature_extracted_df.n_grams.values)

            feature_extracted_df['n_grams_emb'] = self.labels_cluster_obj.transform(feature_extracted_df.n_grams.values)
            tfidf_matrix, tfidf_terms = get_tfidf_matrix(feature_extracted_df.n_grams_emb, max_features=max_features)
        else:
            tfidf_matrix, tfidf_terms = get_tfidf_matrix(feature_extracted_df.n_grams, max_features=max_features)

        # select features from extracted candidates
        feature_extracted_df['terms_doc'] = _load_selected_features(tfidf_matrix, tfidf_terms)

        print('Initial features: ', len(initial_features))
        print('Selected features: ', len(tfidf_terms))
        return feature_extracted_df, len(initial_features), len(tfidf_terms)


def _load_selected_features(tfidf_matrix, terms):
    print('tfidf_matrix shape: {}\n'.format(tfidf_matrix.shape))
    X = tfidf_matrix.toarray()
    return [get_tfidf_words(X[i], terms) for i in range(len(X))]


def fs_mlflowrun(feature_extracted_df, max_features_frac=.20, embedding=None, num_embedding_passes=None):

    fs = FeatureSelection(max_features_frac, embedding, num_embedding_passes)
    fs_df, initial_feats_count, selected_feats_count = fs.select_features(feature_extracted_df)

    local_dir = tempfile.mkdtemp()
    fs_file = os.path.join(local_dir, 'feature_selected_df.pkl')
    pickle.dump(fs_df, open(fs_file, 'wb'))

    # prepare metrics and artifacts maps
    params_map = {'max_features_frac':max_features_frac, 'embedding':embedding}
    metrics_map = {'initial_features': initial_feats_count, 'selected_features_count':selected_feats_count}
    artifacts_map = {"feature-selected-dir": fs_file}

    if fs.labels_cluster_obj is not None:
        params_map['num_embedding_passes'] = len(fs.labels_cluster_obj.thresh_pairs)
        metrics_map['embedding_singular_features'] = len(fs.labels_cluster_obj.singular_terms)
        metrics_map['embedding_clusters_count'] = len(fs.labels_cluster_obj.leaders_map)

    return params_map, metrics_map, artifacts_map, fs_df

