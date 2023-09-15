# general import
import time
import copy
import itertools
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# import from cobras_ts
from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier.commandlinequerier import CommandLineQuerier

# XCobrasExplainer
from model_explainer import ClusteringExplainer


class COBRA_KMeans(COBRAS_kmeans):
    def __init__(self, data, querier, max_questions, train_indices=None, store_intermediate_results=True, init_k=10, verbose=False, y=None):
        super().__init__(data, querier, max_questions, train_indices, store_intermediate_results)
        self.init_k = init_k
        self.verbose = verbose
        self.ml, self.cl = [],[]
        self.y = y
        self.aris = []
        self.y_tmp = []
        self.start_time = []

    def fit(self):
        self.start_time = time.time()

        # STEP1: initial clustering of all the dataset
        ##### At first we only have one super instance/Cluster that we split into K clusters (using K-Means)
        initial_superinstance = self.create_superinstance(list(range(self.data.shape[0])))
        self.clustering = Clustering([Cluster([initial_superinstance])])
        superinstances = self.split_superinstance(initial_superinstance, self.init_k)
        self.clustering.clusters = []
        [self.clustering.clusters.append(Cluster([si])) for si in superinstances]

        # STEP2: merge
        ##### 
        self.merge_containing_clusters(copy.deepcopy(self.clustering.construct_cluster_labeling()))
        last_valid_clustering = copy.deepcopy(self.clustering)

        return self.clustering
    
    # Pas modifiée pour le moment, juste là :)
    # à part les "print" et "if verbose"
    def merge_containing_clusters(self, clustering_to_store):
        """
            Execute the merging phase on the current clustering


        :param clustering_to_store: the last valid clustering, this clustering is stored as an intermediate result for each query that is posed during the merging phase
        :return: a boolean indicating whether the merging phase was able to complete before the query limit is reached
        """
        
        additional_information = [None, None] # Yacine: Additional  information can be either datapoints or explanations

        query_limit_reached = False
        merged = True
        while merged and len(self.ml) + len(self.cl) < self.max_questions:

            clusters_to_consider = [cluster for cluster in self.clustering.clusters if not cluster.is_finished]

            cluster_pairs = itertools.combinations(clusters_to_consider, 2)
            cluster_pairs = [x for x in cluster_pairs if
                            not x[0].cannot_link_to_other_cluster(x[1], self.cl)]
            cluster_pairs = sorted(cluster_pairs, key=lambda x: x[0].distance_to(x[1]))

            merged = False
            for x, y in cluster_pairs:

                if x.cannot_link_to_other_cluster(y, self.cl):
                    # print("hi")
                    continue

                bc1, bc2 = x.get_comparison_points(y)
                pt1 = min([bc1.representative_idx, bc2.representative_idx])
                pt2 = max([bc1.representative_idx, bc2.representative_idx])

                if (pt1, pt2) in self.ml:
                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    merged = True
                    break

                if len(self.ml) + len(self.cl) == self.max_questions:
                    query_limit_reached = True
                    break
                
                # Yacine --------
                self.y_tmp.append(list(self.clustering.construct_cluster_labeling()))

                ari = adjusted_rand_score(self.clustering.construct_cluster_labeling(), self.y)
                self.aris.append(ari)
                if self.verbose:
                    ...# print(f"Query n{len(self.ml+self.cl)}: ARI{ari}")
                # ---------------

                if self.querier.query_points(pt1, pt2):
                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    self.ml.append((pt1, pt2))
                    merged = True

                    if self.store_intermediate_results:
                        self.intermediate_results.append(
                            (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
                    break
                else:
                    self.cl.append((pt1, pt2))

                    if self.store_intermediate_results:
                        self.intermediate_results.append(
                            (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
                

        fully_merged = not query_limit_reached and not merged

        # if self.store_intermediate_results and not starting_level:
        if fully_merged and self.store_intermediate_results:
            self.intermediate_results[-1] = (self.clustering.construct_cluster_labeling(), time.time() - self.start_time,
                                            len(self.ml) + len(self.cl))
        
        # Yacine --------
        ari = adjusted_rand_score(self.clustering.construct_cluster_labeling(), self.y)
        self.aris.append(ari)
        if self.verbose:
            ...# print(f"<FIN> Query n{len(self.ml+self.cl)}: ARI{adjusted_rand_score(self.clustering.construct_cluster_labeling(), self.y)}")
        # ---------------
        
        return fully_merged
    

class XCOBRA_KMeans(COBRA_KMeans):
    def __init__(self, data, querier, max_questions, train_indices=None, store_intermediate_results=True, init_k=10, verbose=False, y=None, feature_names=None, explain_it = None, one_vs_all=True, use_explanation=True):
        """_summary_

        Args:
            data (numpy array): raw data points.
            querier (querier): oracle/querier 
            max_questions (int): budget
            train_indices (_type_, optional): voir docu COBRAS. Defaults to None. 
            store_intermediate_results (bool, optional): True: store intermediate clustering. Defaults to True.
            init_k (int, optional): Number of initial KMeans clustering. Defaults to 10.
            verbose (bool, optional): Verbose (Have more prints). Defaults to False.
            y (numpy array, optional): Ground truth labels. Defaults to None.
            feature_names (numpy array, optional): feature names. Defaults to None.
            explain_it (model_explainer instance, optional): EXPLAIN-IT instance via model_explainer.py class . Defaults to None.
            one_vs_all (bool, optional): 1vsAll if true, else 2vsAll. Defaults to True.
            use_explanation (bool, optional): If False, DQuerier (Raw data) will be used to provide answers to the queries. Defaults to True.
        """
        super().__init__(data, querier, max_questions, train_indices, store_intermediate_results, init_k, verbose, y)
        self.explain_it = explain_it
        self.one_vs_all = one_vs_all
        self.feature_names = feature_names
        self.use_explanation = use_explanation

    # Modifiée: ajout  des explications!!!
    def merge_containing_clusters(self, clustering_to_store):
        """
            Execute the merging phase on the current clustering


        :param clustering_to_store: the last valid clustering, this clustering is stored as an intermediate result for each query that is posed during the merging phase
        :return: a boolean indicating whether the merging phase was able to complete before the query limit is reached
        """

        additional_information = [None, None] # Yacine: Additional  information can be either datapoints or explanations

        query_limit_reached = False
        merged = True
        i = 0 # yacine : compteur
        while merged and len(self.ml) + len(self.cl) < self.max_questions and i < self.max_questions:
            i+=1
            clusters_to_consider = [cluster for cluster in self.clustering.clusters if not cluster.is_finished]

            cluster_pairs = itertools.combinations(clusters_to_consider, 2)
            cluster_pairs = [x for x in cluster_pairs if
                            not x[0].cannot_link_to_other_cluster(x[1], self.cl)]
            cluster_pairs = sorted(cluster_pairs, key=lambda x: x[0].distance_to(x[1]))
            # print(f"--> On devrait tester {len(list(cluster_pairs))} couples de clusters")

            merged = False
            for x, y in cluster_pairs:
                
                if x.cannot_link_to_other_cluster(y, self.cl):
                    continue

                bc1, bc2 = x.get_comparison_points(y)
                pt1 = min([bc1.representative_idx, bc2.representative_idx])
                pt2 = max([bc1.representative_idx, bc2.representative_idx])

                if (pt1, pt2) in self.ml:
                    # print("Salut")
                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    merged = True
                    break

                if len(self.ml) + len(self.cl) == self.max_questions:
                    query_limit_reached = True
                    break
                
                # Yacine --------
                self.y_tmp.append(list(self.clustering.construct_cluster_labeling()))
                
                ari = adjusted_rand_score(self.clustering.construct_cluster_labeling(), self.y)
                self.aris.append(ari)
                if self.verbose:

                    ... # print(f"Query n{len(self.ml+self.cl)}: ARI{ari}")
                # ---------------
                
                # Yacine --------
                if self.use_explanation:

                    y_hat = np.array([0]*len(self.data)) # current clustering / will be used to train the svm model
                    labels_1 = x.get_all_points()
                    labels_2 = y.get_all_points()
                
                    assert pt1 in labels_1+labels_2
                    assert pt2 in labels_1+labels_2
                    
                    if self.one_vs_all:
                        mask = labels_1+labels_2
                        y_hat[mask] = 1 # 1vsAll : pt1,pt2 \in Class1 & all others in \class 0
                        if sum(y_hat==1)==len(self.data):
                            ...# print("Nous n'avons qu'une seule classe ici!!!!!!")                            
                    else:
                        y_hat[labels_1] = 1 # pt1 \in Class1
                        y_hat[labels_2] = 2 # pt1 \in Class2 & all others in \class 0

                    try:
                        # print(f"\t> |C1|={sum(y_hat==1)} - |C0|={sum(y_hat==0)}")
                        shap_values = self.explain_it.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names)                
                    except Exception:
                        print(f"\t> |C1|={sum(y_hat==1)} - |C0|={sum(y_hat==0)} -- PASSED")
                        continue

                    additional_information[0],additional_information[1] = shap_values[0], shap_values[1]
                else:
                    # use raw data
                    additional_information[0],additional_information[1] = self.data[pt1], self.data[pt2]


                if self.querier.query_points(pt1, pt2, additional_information[0], additional_information[1]):
                # ---------------
                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    self.ml.append((pt1, pt2))
                    merged = True

                    if self.store_intermediate_results:
                        self.intermediate_results.append(
                            (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
                    break
                else:
                    self.cl.append((pt1, pt2))

                    if self.store_intermediate_results:
                        self.intermediate_results.append(
                            (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
                

        fully_merged = not query_limit_reached and not merged

        # if self.store_intermediate_results and not starting_level:
        if fully_merged and self.store_intermediate_results:
            self.intermediate_results[-1] = (self.clustering.construct_cluster_labeling(), time.time() - self.start_time,
                                            len(self.ml) + len(self.cl))
        
        # Yacine --------
        ari = adjusted_rand_score(self.clustering.construct_cluster_labeling(), self.y)
        self.aris.append(ari)
        if self.verbose:
            ... # print(f"<FIN> Query n{len(self.ml+self.cl)}: ARI{adjusted_rand_score(self.clustering.construct_cluster_labeling(), self.y)}")
        # ---------------
        
        return fully_merged