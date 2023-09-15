# general import
import time
import copy
import itertools
import numpy as np
from sklearn.cluster import KMeans
from XQuerier import XQuerier
from DQuerier import DQuerier
# import from cobras_ts
from cobras_ts.cluster import Cluster
from cobras_ts.clustering import Clustering
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.querier.commandlinequerier import CommandLineQuerier

# XCobrasExplainer
from model_explainer import ClusteringExplainer


class XCOBRAS_kmeans(COBRAS_kmeans):
    def __init__(self, 
                 xquerier = True,
                 X=None, 
                 labels = None,
                 budget=10, 
                 model_explainer=ClusteringExplainer() , 
                 querier=None, 
                 split_one_versus_all = True,
                 merge_one_versus_all = True,
                 use_explanation=True):
        """ Constructor of the "XCOBRAS_kmeans", extends  class "COBRAS_kmeans"

        Args:
            budget (int, optional): _description_. Defaults to 10.
        """
        self.xquerier = xquerier
        self.budget = budget
        self.fitted = False
        self.explain_it = model_explainer
        self.number_query_GT = 0
        self.arret_gt = False
        self.split_one_vs_all = split_one_versus_all
        self.merge_one_vs_all = merge_one_versus_all
        self.nbr_expplications_used = 0
        self.use_explanation = use_explanation
        self.nbr_data_used = 0

        # compare the different supervised model 
        # TODO Run for 1vsAll and for 1v1 
        self.y_tmp = []
        self.data_answer = [[], []]  # sequence of queried couple of data
        self.data_pts = []  # their corresponding vector points
        self.gt_answer = [[], []] # the answer to those queries using the GT labels
                            # this choice was made to keep every model_explainer, no matter the generated explanation, running on the same...
        self.xquerr = XQuerier(labels, xai_method="shap", strat="cosine_similarity", threshold=0.95)
        self.dquerr = DQuerier(strat='cosine_similarity', threshold=0.95)

        ################ 2VSALL 
        # Values of the supervised models to store
        self.model_1_2vsAll = []
        self.model_2_2vsAll = []
        self.model_3_2vsAll = []
        self.model_4_2vsAll = []

        # shap EXPLAIN-IT framework
        self.model_1_shap_2vsAll = ClusteringExplainer(model="rbf_svm", xai_model="shap", verbose=False, one_versus_all=False)
        self.model_2_shap_2vsAll = ClusteringExplainer(model="decision_tree", xai_model="shap", verbose=False, one_versus_all=False)
        self.model_3_shap_2vsAll = ClusteringExplainer(model="knn-uniform", xai_model="shap", verbose=False, one_versus_all=False)
        self.model_4_shap_2vsAll = ClusteringExplainer(model="knn-dist", xai_model="shap", verbose=False, one_versus_all=False)

        ################ 1VSALL 
        # Values of the supervised models to store
        self.model_1 = []
        self.model_2 = []
        self.model_3 = []
        self.model_4 = []

        # shap EXPLAIN-IT framework
        self.model_1_shap = ClusteringExplainer(model="rbf_svm", xai_model="shap", verbose=False, one_versus_all=True)
        self.model_2_shap = ClusteringExplainer(model="decision_tree", xai_model="shap", verbose=False, one_versus_all=True)
        self.model_3_shap = ClusteringExplainer(model="knn-uniform", xai_model="shap", verbose=False, one_versus_all=True)
        self.model_4_shap = ClusteringExplainer(model="knn-dist", xai_model="shap", verbose=False, one_versus_all=True)
        

    def fit(self, X, feature_names=None, y=CommandLineQuerier(), store_intermediate_results=True):
        """Function that mimics the sklearn "fit" function.
        TODO... compléter après l'avoir terminée

        Args:
            X (np.array): dataset of size (nb_samples, nb_features)
            y (Querier object (from cobras_ts), optional): object that answers the must-link or cannot_link questions. Defaults to CommandLineQuerier.

        Returns:
            - a :class:`~clustering.Clustering` object representing the resulting clustering
            - a list of intermediate clustering labellings for each query (each item is a list of clustering labels)
            - a list of timestamps for each query
            - the list of must-link constraints that was queried
            - the list of cannot-link constraints that was queried
        """
        # calling the super class' constructor
        super().__init__(
            data = X, 
            querier = y, 
            max_questions  = self.budget,
            store_intermediate_results = store_intermediate_results
        )
        self.feature_names = feature_names
        # performs clustering
        self.fitted = True
        return self.cluster()

    def cluster(self):
        """Perform clustering

        :return: if cobras.store_intermediate_results is set to False, this method returns a single Clustering object
                 if cobras.store_intermediate_results is set to True, this method returns a tuple containing the following items:

                     - a :class:`~clustering.Clustering` object representing the resulting clustering
                     - a list of intermediate clustering labellings for each query (each item is a list of clustering labels)
                     - a list of timestamps for each query
                     - the list of must-link constraints that was queried
                     - the list of cannot-link constraints that was queried
        """
        self.start_time = time.time()

        # initially, there is only one super-instance that contains all data indices
        # (i.e. list(range(self.data.shape[0])))
        initial_superinstance = self.create_superinstance(list(range(self.data.shape[0])))

        self.ml = []
        self.cl = []

        self.clustering = Clustering([Cluster([initial_superinstance])])

        # the split level for this initial super-instance is determined,
        # the super-instance is split, and a new cluster is created for each of the newly created super-instances
        initial_k = self.determine_split_level(initial_superinstance,
                                               copy.deepcopy(self.clustering.construct_cluster_labeling()))

        # split the super-instance and place each new super-instance in its own cluster
        superinstances = self.split_superinstance(initial_superinstance, initial_k)
        self.clustering.clusters = []
        for si in superinstances:
            self.clustering.clusters.append(Cluster([si]))

        # the first bottom up merging step
        # the resulting cluster is the best clustering we have so use this as first valid clustering
        self.merge_containing_clusters(copy.deepcopy(self.clustering.construct_cluster_labeling()))
        last_valid_clustering = copy.deepcopy(self.clustering)

        # while we have not reached the max number of questions
        while len(self.ml) + len(self.cl) < self.max_questions:
            # notify the querier that there is a new clustering
            # such that this new clustering can be displayed to the user
            self.querier.update_clustering(self.clustering)

            # after inspecting the clustering the user might be satisfied
            # let the querier check whether or not the clustering procedure should continue
            # note: at this time only used in the notebook queriers
            if not self.querier.continue_cluster_process():
                break

            # choose the next super-instance to split
            to_split, originating_cluster = self.identify_superinstance_to_split()
            if to_split is None:
                break

            # clustering to store keeps the last valid clustering
            clustering_to_store = None
            if self.intermediate_results:
                clustering_to_store = self.clustering.construct_cluster_labeling()

            #### YACINE #####
            # 1. Get all labels of that cluster
            indices_cluster_to_split = originating_cluster.get_all_points()

            ####################
            # remove the super-instance to split from the cluster that contains it
            originating_cluster.super_instances.remove(to_split)
            if len(originating_cluster.super_instances) == 0:
                self.clustering.clusters.remove(originating_cluster)

            # - splitting phase -
            # determine the splitlevel
            split_level = self.determine_split_level(to_split, clustering_to_store, indices_cluster_to_split = indices_cluster_to_split)

            # split the chosen super-instance
            new_super_instances = self.split_superinstance(to_split, split_level)

            # add the new super-instances to the clustering (each in their own cluster)
            new_clusters = self.add_new_clusters_from_split(new_super_instances)
            if not new_clusters:
                # it is possible that splitting a super-instance does not lead to a new cluster:
                # e.g. a super-instance constains 2 points, of which one is in the test set
                # in this case, the super-instance can be split into two new ones, but these will be joined
                # again immediately, as we cannot have super-instances containing only test points (these cannot be
                # queried)
                # this case handles this, we simply add the super-instance back to its originating cluster,
                # and set the already_tried flag to make sure we do not keep trying to split this superinstance
                originating_cluster.super_instances.append(to_split)
                to_split.tried_splitting = True
                to_split.children = None

                if originating_cluster not in self.clustering.clusters:
                    self.clustering.clusters.append(originating_cluster)

                continue
            else:
                self.clustering.clusters.extend(new_clusters)

            # perform the merging phase
            fully_merged = self.merge_containing_clusters(clustering_to_store)
            # if the merging phase was able to complete before the query limit was reached
            # the current clustering is a valid clustering
            if fully_merged:
                last_valid_clustering = copy.deepcopy(self.clustering)

        # clustering procedure is finished
        # change the clustering result to the last valid clustering
        self.clustering = last_valid_clustering

        # return the correct result based on what self.store_intermediate_results contains
        if self.store_intermediate_results:
            return self.clustering, [clust for clust, _, _ in self.intermediate_results], [runtime for _, runtime, _ in
                                                                                           self.intermediate_results], self.ml, self.cl
        else:
            return self.clustering

    # Override this method to include explanations
    def determine_split_level(self, superinstance, clustering_to_store, indices_cluster_to_split = None):
        """ Determine the splitting level for the given super-instance using a small amount of queries

        For each query that is posed during the execution of this method the given clustering_to_store is stored as an intermediate result.
        The provided clustering_to_store should be the last valid clustering that is available

        :return: the splitting level k
        :rtype: int
        """
        
        
        # need to make a 'deep copy' here, we will split this one a few times just to determine an appropriate splitting
        # level
        si = self.create_superinstance(superinstance.indices)

        must_link_found = False
        # the maximum splitting level is the number of instances in the superinstance
        max_split = len(si.indices)
        split_level = 0
        while not must_link_found and len(self.ml) + len(self.cl) < self.max_questions:
            if len(si.indices) == 2:
                # if the superinstance that is being splitted just contains 2 elements split it in 2 superinstances with just 1 instance
                new_si = [self.create_superinstance([si.indices[0]]), self.create_superinstance([si.indices[1]])]
            else:
                # otherwise use k-means to split it
                new_si = self.split_superinstance(si, 2)

            if len(new_si) == 1:
                # we cannot split any further along this branch, we reached the splitting level
                split_level = max([split_level, 1])
                split_n = 2 ** int(split_level)
                return min(max_split, split_n)

            s1 = new_si[0]
            s2 = new_si[1]
            pt1 = min([s1.representative_idx, s2.representative_idx])
            pt2 = max([s1.representative_idx, s2.representative_idx])

            ############################################################################################################
            ##### YACINE #####            
            y_hat = np.array([0]*len(self.data)) # current clustering / will be used to train the svm model
            explanations = [None, None]
            labels_1 = s1.indices
            labels_2 = s2.indices
            assert pt1 in labels_1+labels_2
            assert pt2 in labels_1+labels_2
            
    
            y_hat[labels_1] = 1 # pt1 \in Class1
            y_hat[labels_2] = 2 # pt1 \in Class2 & all others in \class 0

            try:

                # print(f"\t> |C2|={sum(y_hat==1)} - |C1|={sum(y_hat==1)} - |C0|={sum(y_hat==0)}")
                explanations = self.model_1_shap_2vsAll.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names)

                self.model_1_2vsAll.append(int(self.xquerr.query_points(pt1, pt2, explanations[0], explanations[1]  )))
                explanations = self.model_2_shap_2vsAll.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names)

                self.model_2_2vsAll.append(int(self.xquerr.query_points(pt1, pt2, explanations[0], explanations[1]  )))
                explanations = self.model_3_shap_2vsAll.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names)

                self.model_3_2vsAll.append(int(self.xquerr.query_points(pt1, pt2, explanations[0], explanations[1]  )))
                explanations = self.model_4_shap_2vsAll.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names)

                self.model_4_2vsAll.append(int(self.xquerr.query_points(pt1, pt2, explanations[0], explanations[1]  )))
                self.gt_answer[0].append(int(self.querier.query_points(pt1, pt2)))
                self.data_answer[0].append(int(self.dquerr.query_points(pt1, pt2, self.data[pt1], self.data[pt2])))
            except Exception as e:
                print(e)
                print(f"\t> |C1|={sum(y_hat==1)} - |C0|={sum(y_hat==0)} -- PASSED")
                continue                
            ############################################################################################################

            # print(type(self.querier))
            if self.querier.query_points(pt1, pt2):
                self.ml.append((pt1, pt2))
                must_link_found = True
                if self.store_intermediate_results:
                    self.intermediate_results.append(
                        (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
                continue
            else:
                self.cl.append((pt1, pt2))
                split_level += 1
                if self.store_intermediate_results:
                    self.intermediate_results.append(
                        (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))

            si_to_choose = []
            if len(s1.train_indices) >= 2:
                si_to_choose.append(s1)
            if len(s2.train_indices) >= 2:
                si_to_choose.append(s2)

            if len(si_to_choose) == 0:
                split_level = max([split_level, 1])
                split_n = 2 ** int(split_level)
                return min(max_split, split_n)

            si = min(si_to_choose, key=lambda x: len(x.indices))

        split_level = max([split_level, 1])
        split_n = 2 ** int(split_level)
        return min(max_split, split_n)        
    
    def merge_containing_clusters(self, clustering_to_store):
        """
            Execute the merging phase on the current clustering


        :param clustering_to_store: the last valid clustering, this clustering is stored as an intermediate result for each query that is posed during the merging phase
        :return: a boolean indicating whether the merging phase was able to complete before the query limit is reached
        """
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
                
                
                #### YACINE ####
                
                y_hat = np.array([0]*len(self.data)) # current clustering / will be used to train the svm model
                labels_1 = bc1.indices
                labels_2 = bc2.indices
            
                assert pt1 in labels_1+labels_2
                assert pt2 in labels_1+labels_2
                
        
                mask = labels_1+labels_2
                y_hat[mask] = 1 # 1vsAll : pt1,pt2 \in Class1 & all others in \class 0
                                    
                try:
                    # print("merge")
                    # print(f"\t> |C1|={sum(y_hat==1)} - |C0|={sum(y_hat==0)}")
                    # print("model1")
                    self.model_1.append(int(self.xquerr.query_points(pt1, pt2, self.model_1_shap.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names))  ))
                    # print("model2")
                    self.model_2.append(int(self.xquerr.query_points(pt1, pt2, self.model_2_shap.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names))  ))
                    # print("model3")
                    self.model_3.append(int(self.xquerr.query_points(pt1, pt2, self.model_3_shap.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names))  ))
                    # print("model4")
                    self.model_4.append(int(self.xquerr.query_points(pt1, pt2, self.model_4_shap.fit_explain(self.data, y_hat, [pt1, pt2], feature_names=self.feature_names))  ))
                    self.gt_answer[1].append(int(self.querier.query_points(pt1, pt2)))
                    # print("narmol")
                    self.data_answer[1].append(int(self.dquerr.query_points(pt1, pt2, self.data[pt1], self.data[pt2])))
                    # print("narmol2")
                except Exception as e:

                    print(e)
                    print(f"\t> HEYYY???? |C1|={sum(y_hat==1)} - |C0|={sum(y_hat==0)} -- PASSED")
                    continue

               
                # print(type(self.querier))
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
        return fully_merged

    def get_all_SICM(self):
        """getter that gets all the :
        SI (super instances) - C(centroids) - M(mapping function: SI -> Cluster)
        TODO peut-être essayer d'optimiser ça (passer moins de temps à tout reconstruire après chaque étape)
        Returns:
            np.array: array of all the super instances of size (nb_si)
            np.array: array of all their corresponding "centroid" of size (nb_si, nb_features)
            list:     mapping function: list : argument(centroid) -> associated cluster
        """
        if self.fit == None:
            ...
        
        all_clusters = self.clustering.clusters

        # lists to store all the current* SI and 
        #  TODO peut-être essayer d'optimiser ça (passer moins de temps à tout reconstruire après chaque étape)
        # 
        all_super_instances = []
        map_si_to_cluster = []

        for ci, cluster in enumerate(all_clusters):
            temp_si = cluster.super_instances
            all_super_instances+=temp_si
            for i in range(len(temp_si)):
                map_si_to_cluster.append(ci)
 
        all_super_instances = np.array(all_super_instances)
        all_centroids = np.array([si.centroid for si in all_super_instances])
        return all_super_instances, all_centroids, map_si_to_cluster
        
    def predict(self, X):
        """
        Function that mimics the "predict" function of any other sklearn model.
        Returns the "label" (here, cluster) of each data.

        Args:
            X (np.array): dataset of size (nb_samples, nb_features)

        Returns:
            np.array: array of the associated labels of size (nb_samples,)
        """
        
        # TODO changer ça si on prend encompte la "sauvegarde" ou "construction ittérative"
        _, all_centroids, map_si_to_cluster  = self.get_all_SICM()
        
        # ---- GET THE CLOSEST SUPER INSTANCE
        # Use sklearn.KMeans to get the closest super instance (faster)
        k = KMeans(n_clusters=all_centroids.shape[0], max_iter=1,n_init=1)
        k.cluster_centers_ = all_centroids
        # cannot call the predict function of kmeans without at least one fitting iteration
        k.fit(all_centroids)
        # to make sure the indices were not swapped
        k.cluster_centers_ = all_centroids

        # ---- PREDICT THE LABELS
        #    -  kmeans has "nb_super_instances" centroids
        #    -  in reality, several super instances maps to the same cluster
        KMeans_labels = k.predict(X)
        COBRAS_labels = np.array([map_si_to_cluster[i] for i in  KMeans_labels])

        # Returns the clustering labels 
        return COBRAS_labels
    
    def get_cluster_and_all_super_instances(self, super_instance):
        """Function that looks for the super instances leading to the same cluster.
        Objective: Look for all the partitions that are refering to the same cluster. 

        Args:
            super_instance (cobras_ts.superinstance_kmeans.SuperInstance_kmeans): a super instance

        Returns:
            dict: The key is of type:   cobras_ts.cluster.Cluster
                The value is of type: list(cobras_ts.superinstance_kmeans.SuperInstance_kmeans) representing the same cluster
        """
        my_dict = self.clustering.get_cluster_to_generalized_super_instance_map()
        return [{k:list(itertools.chain.from_iterable(v))} for k, v in my_dict.items() if [super_instance] in v][0]

    def score(self, X, y):
        # TODO :)
        pass