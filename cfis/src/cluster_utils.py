from nltk.probability import FreqDist

def print_graph(G):
    cur_queue = list([('root',)])
    visited = set()

    while True:
        while cur_queue:
            next_queue = list()
            cur = cur_queue.pop(0)
            print('\ncur node:', cur)
            for v in G.successors(cur):
                print(v)
                next_queue.append(v)
                visited.add(v)
            print('-' * 20)
        print('==' * 40)
        if not next_queue:
            break
        cur_queue = next_queue

# %%
def check_cluster_distr(G):
    closed_fi_distr = {1:FreqDist(),2:FreqDist(),3:FreqDist(),4:FreqDist(),5:FreqDist()}

    for node in G.nodes:
        closed_fi_distr[len(node)][node] = G.nodes[node]['freq']

    return closed_fi_distr


def print_clusters(clusters):
    for label in clusters:
        print()
        print(label, ':', len(clusters[label]), '\n')
        for doc, score in clusters[label]:
            print(doc)
        print('--' * 20)


"""
    # %%
    # Filter threshold criteria
    def filter_thresh(self, scores, choice):
        if choice == 1:
            return (max(scores)-min(scores))/2
        elif choice == 2:
            return np.mean(scores[scores>0])
        elif choice == 3:
            return np.mean([max(0, score) for score in scores])
        elif choice == 4:
            return np.mean(scores)
        elif choice == 5:
            return np.percentile(scores, 95)
        elif choice == 6:
            return np.percentile(scores, 75) + 1.5*scipy.stats.iqr(scores)
        else:
            print('Wrong choice.')

    # %%
    # Get top matched labels in ranked order for a doc
    def get_top_ranked_labels(self, doc, fi_graph, debug=False):
        label_scores = self.get_label_scores(doc, fi_graph)
        ordered_label_scores = label_scores.most_common()

        if debug:
            print('available labels: ', ordered_label_scores[:10],'\n')

        scores = np.asarray(list(label_scores.values()))
        if debug:
            boxplot(scores)

        max_percentile = self.filter_thresh(scores, 5)
        outlier_thresh = self.filter_thresh(scores, 6)
        strict_filter_thresh = max(max_percentile, outlier_thresh)
        loose_filter_thresh = min(max_percentile, outlier_thresh)

        if debug:
            print('95th percentile:', self.filter_thresh(scores, 5))
            print('+1.5*iqr:', self.filter_thresh(scores, 6))

        best_labels, good_labels = [], []
        for label,score in ordered_label_scores:
            if score >= strict_filter_thresh:
                best_labels.append((label, score))
            elif score >= loose_filter_thresh:
                good_labels.append((label, score))
            else:
                break

        return (np.asarray(best_labels), np.asarray(good_labels)),(strict_filter_thresh,loose_filter_thresh)

    # %%
    # identify weak singleton labels
    def is_weak_singleton(self, label, doc, match_score, doc_labels_map, thresh):
        top_n = 1

        # high score
        if match_score >= thresh: #label==doc:
            return False

        available_labels = doc_labels_map[doc] #list(itertools.chain(*list(doc_labels_map[doc].values())))
        available_labels.sort(key=lambda x:x[1], reverse=True)
        top_labels = np.asarray(available_labels[:top_n])[:,0] # extract label names from top labels

        if label in top_labels: # highest score
            return False

        return True


    # %%
    def get_weak_singletons(self, singletons, doc_lbls_map, thresh=1):
        weak_singlenton = {}
        for label,[(doc,score)] in singletons.items():
            if self.is_weak_singleton(label, doc, score, doc_lbls_map, thresh):
                weak_singlenton[label] = (doc,score)

        return weak_singlenton


    # %%
    def remove_weak_singletons(self, weak_singleton_label_docs, label_docs_map, doc_labels_map, cluster_graph, debug=False):
        print('original labels:', len(label_docs_map))
        filtered_lbl_docs_map = deepcopy(label_docs_map)
        filtered_doc_lbls_map = deepcopy(doc_labels_map)

        for label,(doc,score) in weak_singleton_label_docs.items():
            # remove weak singleton label occurrences
            del filtered_lbl_docs_map[label]
            filtered_doc_lbls_map[doc].remove((label,score))
            cluster_graph.remove_node(label)

        if debug:
            print('Removed %d singletons, remaining labels: %d'% (len(weak_singleton_label_docs), len(filtered_lbl_docs_map)))

        return frozendict(filtered_lbl_docs_map), frozendict(filtered_doc_lbls_map), cluster_graph

    # %%
    def is_perfect_match(self, doc, label):
        if (len(label) <= len(doc) and set(label).issubset(doc)):# or set(doc).issubset(label):
            return True
        return False

    """