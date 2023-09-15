from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import shap
from lime import lime_tabular

# import 2e modèle
# .....
    

class ClusteringExplainer():
    """
    décrire son intêret... 

    Attributes:
        - self.model (str): which classifier are we going to use
            by default"rbf_svm" 
        - self.clf (sklearn Pipeline): the actual pipeline of this classifier.
            Using "Pipeline()" makes it clearer and easier to fit, predict and manipulate.
        - self.grid_search_cv (dict): The parameters that are going to be fine-tuned
                                      wrt the chosen classifier
        - self.test_size (je l'ai mis ici, comme ça ne touche pas directement 'COBRAS')

    ...    
    """
    def __init__(self, 
                 model="rbf_svm", 
                 xai_model="shap", 
                 test_size=0.4,               # when training the "model" model 
                 one_versus_all = True,       
                 discretize_continuous=False, # for lime only
                 verbose=True) -> None:
        """Init function

        Args:
            model (str, optional): Which classifier are we going to use. Defaults to "rbf_svm".
            test_size (float, optional): Proportion of the test dataset. Defaults to (0.4).
        """
        self.model = model
        self.test_size = test_size
        self.param_grid = None
        self.clf = None
        self.grid_search_cv = None
        self.verbose = verbose
        self.xai_model = xai_model
        self.one_vs_all = one_versus_all
        self.discretize_continuous = discretize_continuous
        self.explainer = None
        self.explanations = None


        # ----- Model selection
        if self.model == "rbf_svm":
            # RBF Model
            self.clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001, probability=True))
            ])
            # Parameters  of the grid search
            gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
            Cs = [1, 10, 100, 1e3, 1e4, 1e5]
            self.param_grid = {
                "svm_clf__gamma": gammas, 
                "svm_clf__C": Cs
                }
        elif self.model=="decision_tree":
            # Decision Tree
            self.clf = DecisionTreeClassifier()
            # Parameters of the grid search
            criterions = ["gini", "entropy"]
            self.param_grid = {
                "criterion": criterions
            }
        elif self.model == "knn-dist":
            # k-NeighboursClassifier
            self.clf = KNeighborsClassifier(weights='distance')
            # Parameters of the grid search
            n_neighbors = [3, 5, 7, 10]
            self.param_grid = {
                "n_neighbors": n_neighbors
            }
        elif self.model == "knn-uniform":
            # k-NeighboursClassifier
            self.clf = KNeighborsClassifier(weights='uniform') # default
            # Parameters of the grid search
            n_neighbors = [3, 5, 7, 10]
            self.param_grid = {
                "n_neighbors": n_neighbors
            }
        else:
            ... # un autre modèle avec d'autres hyperparam à fine-tune 

        self.grid_search_cv = GridSearchCV(
            estimator=self.clf, 
            param_grid=self.param_grid, 
            # factor=2, # only half of the candidates are selected
            cv=StratifiedKFold(n_splits=2) # default value
            )

    def fit(self, X, y):
        """Function that fits the classification model in  `self.clf`.
                  i. splits the data into train-test set
                 ii. gridsearchCV on the train set
                iii. (optional) test on the test set to prevent overfitting
        Args:
            X (np.array/pd.DataFrame): Dataset
            y (np.array/pd.DataFrame): Labels
        """
        # ----- dataset split (X and y)
        # `y_hat` because it is the current "partitionning" of COBRAS.
        # These are not ground truth label of the dataset, but cluster assigniation of COBRAS algorithm

        self.X_train, self.X_test, self.y_hat_train, self.y_hat_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # ----- Cross-Validation on the TRAIN set
        # Fitting this model
        # GridSearchCV
        # TODO GERER LES NUMPY ARRAY ET LES DATAFRAME 
        # TODO POUR LE MOMENT QUE DES NUMPY ARRAY
        self.grid_search_cv.fit(self.X_train,self.y_hat_train)
        self.best_model = self.grid_search_cv.best_estimator_
        # ----- Showing some results on the test set
        if self.verbose:
            # print("heyyyyyy "+ self.best_model.score)
            y_test_pred = self.predict(self.X_test)
            print("---------Some scores:---------")
            print("------------------------------")
            print(f"f1-score (macro): {f1_score(self.y_hat_test, y_test_pred, average='macro'):.10f}")
            print(f"         (micro): {f1_score(self.y_hat_test, y_test_pred, average='micro'):.10f}")
            # print(f"  accuracy_score: {accuracy_score(self.y_hat_test, y_test_pred):.10f}")
            print("------------------------------")
            print("")
                
    def predict(self, X):
        """Prediction function

        Args:
            X (np.array/pd.DataFrame): Dataset we want to predict

        Returns:
            np.array: that represents the list of predictions (labels)
        """
        return self.best_model.predict(X)

    def explain(self, X, feature_names=None, class_names=None):
        # TODO warning / try / catch: self.best_model
        
        # if feature_names == None:
        #     feature_names = ["A: "+str(i)for i in range(X.shape[1])]
        class_names = list(set(self.y_hat_train).union(set(self.y_hat_test)))
        if self.xai_model == "shap":

            self.explainer = shap.Explainer(
                self.best_model.predict,
                self.X_train,
                feature_names=feature_names
            )
            self.explanations = self.explainer(X)
            # iek 
            return self.explanations

        elif self.xai_model == "KernelShap"    :
            ## PAS UTILISE
            print(" -- >Explaining with Kernel Shap")
            self.explainer = shap.KernelExplainer(
                self.best_model.predict,
                self.X_train,
                feature_names=feature_names
            )
            self.explanations = self.explainer(X)
            return self.explanations
        
        else: # lime
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data=self.X_train, 
                feature_names=feature_names,
                # class_names=class_names,
                discretize_continuous=self.discretize_continuous,
                mode="classification"
            )

            labels = [0, 1, 2]
            if self.one_vs_all == True:
                labels = [0, 1]
            
            exp1 = self.explainer.explain_instance(
                data_row=X[0], 
                predict_fn=self.best_model.predict_proba,
                num_features = X.shape[1],
                labels=labels
            )

            exp2 = self.explainer.explain_instance(
                data_row=X[1], 
                predict_fn=self.best_model.predict_proba,
                num_features = X.shape[1],
                labels=labels
            )
            if self.one_vs_all:
                self.explanations = [exp1.as_list(label=1), exp2.as_list(label=1)]
            else:
                self.explanations = [exp1.as_list(label=1), exp2.as_list(label=2)]
            return self.explanations
            

        
    

    def fit_explain(self, X, y, ids, feature_names=None):
        self.fit(X,y)
        return self.explain(X[ids], feature_names = feature_names, class_names=list(set(y)))
    
    def get_scores(self):
        y_test_pred = self.predict(self.X_test)
        return f1_score(self.y_hat_test, y_test_pred, average='macro'), f1_score(self.y_hat_test, y_test_pred, average='micro')