# Copyright Toyota Research Institute 2021
"""
This module implements agents and helper functions designed
support multi-fidelity discovery campaigns.

See the following reference:

Palizhati A, Aykol M, Suram S, Hummelshøj JS, Montoya JH. Multi-fidelity Sequential Learning for
Accelerated Materials Discovery. ChemRxiv. Cambridge: Cambridge Open Engage; 2021; This content
is a preprint and has not been peer-reviewed.
"""

import GPy
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from camd.agent.base import HypothesisAgent

from copy import copy

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error

from scipy.stats import norm
from scipy.special import ndtr

def get_features_from_df(df, features):
    """
    Helper function to get feature columns of a dataframe.

    Args:
        df (pd.DataFrame)       A pd.DataFrame where the features are extracted.
        features (list of str)  Name of the features columns in df.

    Returns:
            feature_df      A pd.DataFrame that only contains the features used in ML.
    """
    feature_df = df[features]
    return feature_df


def process_data(
    candidate_data, seed_data, features, label, y_reshape=False, preprocessor=StandardScaler()):
    """
    Helper function that process data for ML model and returns
    np.ndarray of training features, training labels,
    test features and test labels.
    """
    X_train = get_features_from_df(seed_data, features).values
    y_train = np.array(seed_data[label])
    X_test = get_features_from_df(candidate_data, features).values
    y_test = np.array(candidate_data[label])

    X_train_comps = seed_data['Composition']
    X_test_comps = candidate_data['Composition']

    if y_reshape:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    if preprocessor:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
    return X_train, y_train, X_test, y_test, X_train_comps, X_test_comps

def EI(mean, std, max_val, tradeoff):
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff) * ndtr(z) + std * norm.pdf(z)


class EpsilonGreedyMultiAgent(HypothesisAgent):
    """
    A multi-fidelity agent that allocates the desired budget for high fidelity versus
    low fidelity candidate data, and acquires candidates in each fidelity via exploitation
    """

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        features=None,
        fidelities=("theory_data", "expt_data"),
        target_prop=None,
        target_prop_val=None,
        preprocessor=StandardScaler(),
        model=None,
        ranking_method="minimize",
        total_budget=None,
        highFi_query_frac=None,
        similarity_thres=300.0,
        lowFi_per_highFi=1,
    ):
        """
        Args:
            candidate_data (df)      Candidate search space for the campaign.
            seed_data (df)           Seed (training) data for the campaign.
            features (tuple of str)  Name of the feature columns used in machine learning model training.
            fidelities (tuple)       Fidelity levels of the datasets. The strings in the tuple should be arranged
                                     from low to high fidelity.
            target_prop (str)        The property machine learning model is learning, given feature space.
            target_prop_val (float)  The ideal value of the target property.
            preprocessor             A preprocessor that preprocess the features. It can be None, a single
                                     processor, or a pipeline. The default is sklearn StandardScaler.
            model                    A sklearn supervised machine learning regressor.
            ranking_method (str)     Ranking method of candidates based on ML predictions. Select one of the
                                     following: "minimize", "ascending", or "descending". "minimize" will rank
                                     candidates with ML candidates closest to the target property value. "ascending"
                                     or "descening" will rank candidates with ascending/descening ML predictions, best
                                     to use when there is no target propety value (i.e. smaller/larger the better).
            total_budget (int)       The number of the hypotheses at a given iteration of the campaign.
            highFi_query_frac        The fraction of the total budget used for high fidelity hypotheses queries.
                                     The value should be >0 and <=1.
            similarity_thres(float)  The threshold value for l2 norm similarity between a candidate composition and
                                     compositions in the seed data. User will need to run some calculations
                                     to determine the best threshold value.
            lowFi_per_highFi (int)   The number of low fidelity candidate selected to support each
                                     high fidelity candidates that predicted to be good, but the agent
                                     does not want to generate that experimental hypotheses yet.
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.features = features
        self.fidelities = fidelities
        self.target_prop = target_prop
        self.target_prop_val = target_prop_val
        self.preprocessor = preprocessor
        self.model = model
        self.ranking_method = ranking_method
        self.total_budget = total_budget
        self.highFi_query_frac = highFi_query_frac
        self.similarity_thres = similarity_thres
        self.lowFi_per_highFi = lowFi_per_highFi
        self.num_LF_added = list()
        self.num_HF_added = list()
        self.rmse_train = list()
        self.rmse_test = list()
        self.rmse_all = list()
        self.Xtrain_size = list()
        self.iter = 0
        super(EpsilonGreedyMultiAgent).__init__()

    def _calculate_similarity(self, comp, seed_comps):
        """
        Helper function that calculates similarity between
        a composition and the seed data compositions. The similarity
        is reprsented by l2_norm.

        Args:
            comp(pd.core.series):    A specific composition represented by Magpie.
            seed_comps (df):         Compostions in seed represented by Magpie.
        """
        l2_norm = np.linalg.norm(comp.values - seed_comps.values, axis=1)
        return l2_norm

    def _query_hypotheses(self, candidate_data, seed_data):
        """
        Query hypotheses given candidate and seed data via exploitation.
        """
        # separate the candidate space into high and low fidelity candidates
        high_fidelity_candidates = candidate_data.loc[
            candidate_data[self.fidelities[1]] == 1
        ]
        low_fidelity_candidates = candidate_data.loc[
            candidate_data[self.fidelities[0]] == 1
        ]

        # edge cases: end campaign if there are no high fidelity candidate
        # use the entire query budget if there are only high fidelity candidate
        if len(high_fidelity_candidates) == 0:
            return None

        elif (len(high_fidelity_candidates) != 0) & (len(low_fidelity_candidates) == 0):
            selected_hypotheses = high_fidelity_candidates.head(self.total_budget)
            self.num_HF_added.append(self.total_budget)
            self.num_LF_added.append(0)

        else:
            selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)

            # Query high fidelity candidates first
            highFi_budget = int(self.total_budget * self.highFi_query_frac)
            seed_data_fea = get_features_from_df(seed_data, self.features)
            highFi_cands_fea = get_features_from_df(
                high_fidelity_candidates, self.features
            )

            for idx, cand_fea in highFi_cands_fea.iterrows():
                if len(selected_hypotheses) < highFi_budget:
                    normdiff = self._calculate_similarity(cand_fea, seed_data_fea)
                    if len(normdiff[(normdiff <= self.similarity_thres)]) >= 1:
                        selected_hypotheses = selected_hypotheses.append(
                            high_fidelity_candidates.loc[idx]
                        )

            # query low fidelity candidate for remaining budget
            remained_highFi_cands_fea = highFi_cands_fea.drop(selected_hypotheses.index)
            lowFi_candidates_copy = low_fidelity_candidates.copy()
            for idx, cand_fea in remained_highFi_cands_fea.iterrows():
                if (len(selected_hypotheses) < self.total_budget) & (
                    len(lowFi_candidates_copy) != 0
                ):
                    lowFi_cands_fea = get_features_from_df(
                        lowFi_candidates_copy, self.features
                    )
                    lowFi_candidates_copy["normdiff"] = self._calculate_similarity(
                        cand_fea, lowFi_cands_fea
                    )
                    lowFi_candidates_copy = lowFi_candidates_copy.sort_values(
                        "normdiff"
                    )
                    selected_hypotheses = selected_hypotheses.append(
                        lowFi_candidates_copy.head(self.lowFi_per_highFi)
                    )
                    lowFi_candidates_copy = lowFi_candidates_copy.drop(
                        lowFi_candidates_copy.head(self.lowFi_per_highFi).index
                    )

            self.num_HF_added.append(highFi_budget)
            self.num_LF_added.append((self.total_budget-highFi_budget)*self.lowFi_per_highFi)

        return selected_hypotheses

    def get_hypotheses(self, candidate_data, seed_data):
        """
        Gets hypotheses using agent.

        Args:
            candidate_data (pd.DataFrame): dataframe of candidates
            seed_data (pd.DataFrame): dataframe of known data

        Returns:
            (pd.DataFrame): dataframe of selected candidates

        """

        features_columns = copy(self.features)
        if self.fidelities not in features_columns:
            features_columns += list(self.fidelities)

        X_train, y_train, X_test, y_test, X_train_comps, X_test_comps = process_data(
            candidate_data,
            seed_data,
            features_columns,
            self.target_prop,
            preprocessor=self.preprocessor,
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_pred_train = self.model.predict(X_train)
        y_pred_all = self.model.predict(np.concatenate([X_train, X_test]))

        # Make the LF model parity plots
        #if self.iter % 50 == 0:
        #    Scatter().plot_predicted_vs_true(y_true=np.concatenate([y_train, y_test]), y_pred=y_pred_all, savepath='',
        #                                     data_type='HF_' + str(self.iter), x_label='HF Bandgap (eV)',
        #                                     metrics_list=['r2_score', 'mean_absolute_error',
        #                                                   'root_mean_squared_error', 'rmse_over_stdev'])

        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_all = np.sqrt(mean_squared_error(np.concatenate([y_train, y_test]), y_pred_all))
        self.rmse_test.append(rmse_test)
        self.rmse_train.append(rmse_train)
        self.rmse_all.append(rmse_all)

        self.Xtrain_size.append(X_train.shape[0])

        # Make a copy of the candidate data so the original one
        # does not get modified during hypotheses generation
        candidate_data_copy = candidate_data.copy()
        candidate_data_copy["distance_to_ideal"] = np.abs(self.target_prop_val - y_pred)

        if self.ranking_method == "minimize":
            candidate_data_copy = candidate_data_copy.sort_values(
                by=["distance_to_ideal"]
            )
        elif self.ranking_method == "ascending":
            candidate_data_copy = candidate_data_copy.sort_values(
                by=y_pred, ascending=True
            )
        elif self.ranking_method == "descending":
            candidate_data_copy = candidate_data_copy.sort_values(
                by=y_pred, ascending=False
            )

        hypotheses = self._query_hypotheses(candidate_data_copy, seed_data)

        # Saving the X data to files
        #if self.iter % 50 == 0:
        #    pd.DataFrame(X_train, columns=features_columns).to_csv('X_train.csv', index=False)
        #    pd.DataFrame(X_test, columns=features_columns).to_csv('X_test.csv', index=False)
        #    pd.DataFrame(X_train_comps).to_csv('X_train_comps.csv', index=False)
        #    pd.DataFrame(X_test_comps).to_csv('X_test_comps.csv', index=False)
        #    pd.Series(y_train).to_csv('y_train.csv', index=False)
        #    pd.Series(y_test).to_csv('y_test.csv', index=False)

        #self.iter += 1

        return hypotheses


class GPMultiAgent(HypothesisAgent):
    """
    A Gaussian process lower confidence bound derived multi-fidelity agent.
    This agent operates under a total acquisition budget. It acquires
    candidates factoring in GPR predicted uncertainties in the LCB setting
    and hallucination of information gain from DFT acquisitions analogous
    to work of Desautels et al. in batch mode LCB.  The agent aims for
    prioritizing exploitation primarily with high-fidelity experimental measurements,
    offloading exploratory (higher risk) acquisitions first to lower-fidelity computations.
    """

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        chemsys_col="reduced_formula",
        features=None,
        fidelities=("theory_data", "expt_data"),
        target_prop=None,
        target_prop_val=1.8,
        total_budget=None,
        preprocessor=StandardScaler(),
        gp_max_iter=200,
        alpha=1.0,
        rank_thres=10,
        unc_percentile=5,
    ):
        """
        Args:
            candidate_data (df)      Candidate search space for the campaign.
            seed_data (df)           Seed (training) data for the campaign.
            chemsys_col (str)        The name of the composition column.
            features (tuple of str)  Column name of the features used in machine learning.
            fidelities (tuple)       Fidelity levels of the datasets. The strings in the tuple should be arranged
                                     from low to high fidelity. The value of fidelity features should
                                     be one-hot encoded.
            target_prop (str)        The property machine learning model is predicting, given feature space.
            target_prop_val (float)  The ideal value of the target property.
            total_budget (int)       The budget for the hypotheses queried.
            gp_max_iter (int)        Number of maximum iterations for GP optimization.
            preprocessor             A preprocessor that preprocess the features. It can be None, a single
                                     processor, or a pipeline. The default is StandardScaler().
            alpha (float)            The mixing parameter for uncertainties. It controls the
                                     trade-off between exploration and exploitation. Defaults to 1.0.
            rank_thres (int)         A threshold help to decide if lower fidelity data is worth acquiring.
            unc_percentile (int)     A number between 0 and 100, and used to calculate an uncertainty threshold
                                     at that percentile value. The threshold is used to decide if the
                                     agent is quering theory or experimental hypotheses.
        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.chemsys_col = chemsys_col
        self.features = features
        self.fidelities = fidelities
        self.target_prop = target_prop
        self.target_prop_val = target_prop_val
        self.total_budget = total_budget
        self.preprocessor = preprocessor
        self.gp_max_iter = gp_max_iter
        self.alpha = alpha
        self.rank_thres = rank_thres
        self.unc_percentile = unc_percentile
        super(GPMultiAgent).__init__()

    def _halluciate_lower_fidelity(self, seed_data, candidate_data, low_fidelity_data):
        # make copy of seed and candidate data, so we don't mess with the original one
        new_candidate_data = candidate_data.copy()
        low_fidelity = low_fidelity_data.copy()

        low_fidelity[self.target_prop] = low_fidelity_data["y_pred"]
        low_fidelity = low_fidelity.drop(columns=["y_pred"])
        new_seed_data = pd.concat([seed_data, low_fidelity])
        new_candidate_data = new_candidate_data.drop(low_fidelity.index)
        pred_candidate_data = self._train_and_predict(new_candidate_data, new_seed_data)
        return pred_candidate_data

    def _get_rank_after_hallucination(
        self, seed_data, candidate_data, orig_idx, low_fidelity
    ):
        halluciated_candidates = self._halluciate_lower_fidelity(
            seed_data, candidate_data, low_fidelity
        )
        halluciated_candidates = halluciated_candidates.loc[
            halluciated_candidates[self.fidelities[1]] == 1
        ]
        halluciated_candidates = halluciated_candidates.sort_values("pred_lcb")
        rank_after_hallucination = halluciated_candidates.index.get_loc(orig_idx)
        return rank_after_hallucination

    def _train_and_predict(self, candidate_data, seed_data):
        features_columns = self.features + list(self.fidelities)
        X_train, y_train, X_test, y_test = process_data(
            candidate_data,
            seed_data,
            features_columns,
            self.target_prop,
            y_reshape=True,
            preprocessor=self.preprocessor,
        )
        gp = GPy.models.GPRegression(X_train, y_train)
        gp.optimize("bfgs", max_iters=self.gp_max_iter)
        y_pred, var = gp.predict(X_test)

        # Make a copy of the candidate data so the original one
        # does not get modified during hypotheses generation
        candidate_data_copy = candidate_data.copy()
        dist_to_ideal = np.abs(self.target_prop_val - y_pred)
        pred_lcb = dist_to_ideal - self.alpha * var**0.5
        candidate_data_copy["pred_lcb"] = pred_lcb
        candidate_data_copy["pred_unc"] = var**0.5
        candidate_data_copy["y_pred"] = y_pred
        return candidate_data_copy

    def _query_hypotheses(self, candidate_data, seed_data):
        high_fidelity_candidates = candidate_data.loc[
            candidate_data[self.fidelities[1]] == 1
        ]
        high_fidelity_candidates = high_fidelity_candidates.sort_values("pred_lcb")

        # edge case: top the campaign if there are no high fidelity candidates
        if len(high_fidelity_candidates) == 0:
            return None

        else:
            selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)
            unc_thres = np.percentile(
                np.array(high_fidelity_candidates.pred_unc), self.unc_percentile
            )

            # query hypothesis
            for idx, candidate in high_fidelity_candidates.iterrows():
                if len(selected_hypotheses) < self.total_budget:
                    chemsys = candidate[self.chemsys_col]
                    low_fidelity = candidate_data.loc[
                        (candidate_data[self.chemsys_col] == chemsys)
                        & (candidate_data[self.fidelities[0]] == 1)
                    ]

                    # exploit if uncertainty is low or the low fidelity data is not available
                    if (candidate.pred_unc <= unc_thres) or (len(low_fidelity) == 0):
                        selected_hypotheses = selected_hypotheses.append(candidate)
                    # explore
                    else:
                        orig_rank = high_fidelity_candidates.index.get_loc(idx)
                        new_rank = self._get_rank_after_hallucination(
                            seed_data, candidate_data, idx, low_fidelity
                        )

                        delta_rank = new_rank - orig_rank
                        if delta_rank <= self.rank_thres:
                            selected_hypotheses = selected_hypotheses.append(candidate)
                        else:
                            selected_hypotheses = selected_hypotheses.append(
                                low_fidelity
                            )
        return selected_hypotheses

    def get_hypotheses(self, candidate_data, seed_data):
        """
        Selects candidate data for experiment using agent methods

        Args:
            candidate_data (pd.DataFrame): candidate data
            seed_data (pd.DataFrame): known data on which to base the selection
                of candidates

        Returns:
            (pd.DataFrame): selected candidates e.g. for experiment

        """
        candidate_data = self._train_and_predict(candidate_data, seed_data)
        hypotheses = self._query_hypotheses(candidate_data, seed_data)
        return hypotheses


class MultiFidelityMLAgent(HypothesisAgent):

    def __init__(
        self,
        candidate_data=None,
        seed_data=None,
        features=None,
        fidelities=("theory_data", "expt_data"),
        target_prop=None,
        target_prop_val=None,
        preprocessor_lf=None,
        preprocessor_hf=None,
        model=RandomForestRegressor(),
        model_LF=RandomForestRegressor(),
        ranking_method="exploit",
        total_budget=None,
        highFi_query_frac=None,
        lowFi_per_highFi=1,
        composition_column='Composition',
        LF_feature_column='ML PBE bandgap',
        fidelity='multi'
    ):
        """
        Args:
            candidate_data (df)      Candidate search space for the campaign.
            seed_data (df)           Seed (training) data for the campaign.
            features (tuple of str)  Name of the feature columns used in machine learning model training.
            fidelities (tuple)       Fidelity levels of the datasets. The strings in the tuple should be arranged
                                     from low to high fidelity.
            target_prop (str)        The property machine learning model is learning, given feature space.
            target_prop_val (float)  The ideal value of the target property.
            preprocessor_lf          A preprocessor that preprocess the features for the low fidelity ML model. It can be None, a single
                                     processor, or a pipeline. The default is sklearn StandardScaler.
            preprocessor_hf          A preprocessor that preprocess the features for the high fidelity ML model. It can be None, a single
                                     processor, or a pipeline. The default is sklearn StandardScaler.
            model                    A sklearn supervised machine learning regressor. Used for the HF data predictions
            model_LF                 A sklearn supervised machine learning regressor. Used for the LF data predictions
            ranking_method (str)     Ranking method of candidates based on ML predictions. Select one of the
                                     following: "exploit". As of this writing, this approach only uses purely exploitation.
            total_budget (int)       The number of the hypotheses at a given iteration of the campaign.
            highFi_query_frac        The fraction of the total budget used for high fidelity hypotheses queries.
                                     The value should be >0 and <=1.
            lowFi_per_highFi (int)   The number of low fidelity candidate selected to support each
                                     high fidelity candidates that predicted to be good, but the agent
                                     does not want to generate that experimental hypotheses yet.
            composition_column (str) Name of the input dataframe column containing material compositions
            target_column (str)      Name of the input dataframe column containing target property values
            LF_feature_column (str)  Name of the input dataframe column containing the ML-predicted LF values
            fidelity (str)           Choose from "single" or "multi". Determines whether to perform multifidelity search with LF and HF ML models ("multi"),
                                    or reduce to search over HF data only ("single").\

        """
        self.candidate_data = candidate_data
        self.seed_data = seed_data
        self.features = features
        self.fidelities = fidelities
        self.target_prop = target_prop
        self.target_prop_val = target_prop_val
        self.preprocessor_lf = preprocessor_lf
        self.preprocessor_hf = preprocessor_hf
        self.model = model
        self.model_LF = model_LF
        self.ranking_method = ranking_method
        self.total_budget = total_budget
        self.highFi_query_frac = highFi_query_frac
        self.lowFi_per_highFi = lowFi_per_highFi
        self.composition_column = composition_column
        self.LF_feature_column = LF_feature_column
        self.fidelity = fidelity
        self.num_LF_train = list() # List of number of materials used in LF ML model training
        self.num_HF_train = list() # List of number of materials used in HF ML model training
        self.LF_train_rmse = list() # List of LF ML model train RMSE values
        self.LF_test_rmse = list() # List of LF ML model test RMSE values
        self.HF_train_rmse = list() # List of HF ML model train RMSE values
        self.HF_test_rmse = list() # List of HF ML model test RMSE values
        self.LF_full_rmse = list() # List of LF ML model full data (train+test) RMSE values
        self.HF_full_rmse = list() # List of HF ML model full data (train+test) RMSE values
        self.all_HF_idx = list() # List of HF dataframe entry indices
        self.all_LF_idx = list() # list of LF dataframe entry indices
        super(MultiFidelityMLAgent).__init__()

    def _process_data_MultiMLagent(self, candidate_data, seed_data, label, features_HF, features_LF=None):
        """
        Helper function that process data for Multifidelity ML model and returns dataframes of training and testing data
        for both the LF and HF models.
        """

        # separate the candidate space into high and low fidelity candidates
        high_fidelity_candidates = candidate_data.loc[candidate_data[self.fidelities[1]] == 1]
        high_fidelity_seed = seed_data.loc[seed_data[self.fidelities[1]] == 1]

        if self.fidelity == 'multi':
            low_fidelity_candidates = candidate_data.loc[candidate_data[self.fidelities[0]] == 1]
            low_fidelity_seed = seed_data.loc[seed_data[self.fidelities[0]] == 1]

            X_train_LF = get_features_from_df(low_fidelity_seed, features_LF)
            X_train_LF_comps = low_fidelity_seed[self.composition_column]
            y_train_LF = np.array(low_fidelity_seed[label])
            X_test_LF = get_features_from_df(low_fidelity_candidates, features_LF)
            X_test_LF_comps = low_fidelity_candidates[self.composition_column]
            y_test_LF = np.array(low_fidelity_candidates[label])

        else:
            X_train_LF = None
            X_train_LF_comps = None
            y_train_LF = None
            X_test_LF = None
            X_test_LF_comps = None
            y_test_LF = None

        X_train_HF = get_features_from_df(high_fidelity_seed, features_HF)
        X_train_HF_comps = high_fidelity_seed[self.composition_column]
        y_train_HF = np.array(high_fidelity_seed[label])
        X_test_HF = get_features_from_df(high_fidelity_candidates, features_HF)
        X_test_HF_comps = high_fidelity_candidates[self.composition_column]
        y_test_HF = np.array(high_fidelity_candidates[label])

        if self.preprocessor_lf is not None:
            self.preprocessor_lf = self.preprocessor_lf.fit(X_train_LF)
            X_train_LF = pd.DataFrame(self.preprocessor_lf.transform(X_train_LF), columns=features_LF, index=X_train_LF.index)
        # Case where there is no test data b/c out of candidates
        if self.fidelity == 'multi':
            if X_test_LF.shape[0] == 0:
                X_test_LF = X_train_LF
                y_test_LF = y_train_LF
            else:
                if self.preprocessor_lf is not None:
                    X_test_LF = pd.DataFrame(self.preprocessor_lf.transform(X_test_LF), columns=features_LF, index=X_test_LF.index)

        if self.preprocessor_hf is not None:
            self.preprocessor_hf = self.preprocessor_hf.fit(X_train_HF)
            X_train_HF = pd.DataFrame(self.preprocessor_hf.transform(X_train_HF), columns=features_HF, index=X_train_HF.index)
        if X_test_HF.shape[0] == 0:
            X_test_HF = X_train_HF
            y_test_HF = y_train_HF
        else:
            if self.preprocessor_hf is not None:
                X_test_HF = pd.DataFrame(self.preprocessor_hf.transform(X_test_HF), columns=features_HF, index=X_test_HF.index)

        X_all_HF = pd.DataFrame(np.concatenate([X_train_HF, X_test_HF]), columns=features_HF)
        y_all_HF = pd.DataFrame(np.concatenate([y_train_HF, y_test_HF]), columns=[self.target_prop])

        if self.fidelity == 'multi':
            X_all_LF = pd.DataFrame(np.concatenate([X_train_LF, X_test_LF]), columns=features_LF)
            y_all_LF = pd.DataFrame(np.concatenate([y_train_LF, y_test_LF]), columns=[self.target_prop])
        else:
            X_all_LF = None
            y_all_LF = None

        return X_train_LF, y_train_LF, X_test_LF, y_test_LF, X_train_HF, y_train_HF, X_test_HF, y_test_HF, X_all_LF, \
               X_all_HF, y_all_LF, y_all_HF, X_train_LF_comps, X_test_LF_comps, X_train_HF_comps, X_test_HF_comps

    def _query_hypotheses(self, candidate_data, seed_data):
        """
        Query hypotheses given candidate and seed data via exploitation.
        """
        # separate the candidate space into high and low fidelity candidates
        high_fidelity_candidates = candidate_data.loc[candidate_data[self.fidelities[1]] == 1]
        low_fidelity_candidates = candidate_data.loc[candidate_data[self.fidelities[0]] == 1]

        # edge cases: end campaign if there are no high fidelity candidate
        # use the entire query budget if there are only high fidelity candidate
        if len(high_fidelity_candidates) == 0:
            return None

        elif (len(high_fidelity_candidates) != 0) & (len(low_fidelity_candidates) == 0):
            # Still just query the highFi budget instead of the total budget
            highFi_budget = int(self.total_budget * self.highFi_query_frac)
            selected_hypotheses = high_fidelity_candidates.head(highFi_budget)

        else:
            selected_hypotheses = pd.DataFrame(columns=candidate_data.columns)

            # Query high fidelity candidates first
            highFi_budget = int(self.total_budget * self.highFi_query_frac)

            selected_hypotheses = selected_hypotheses.append(high_fidelity_candidates.head(highFi_budget))

            # Need to first query LF values that are same as HF. Then add extra LF as the budget allows
            lf_idx = list()
            hf_idx = list()
            if len(selected_hypotheses) < self.total_budget:
                selected_HF_comps = selected_hypotheses[self.composition_column]
                for comp in selected_HF_comps:
                    selected_hypotheses = selected_hypotheses.append(low_fidelity_candidates[low_fidelity_candidates[self.composition_column] == comp])
                    lf_idx.append(low_fidelity_candidates[low_fidelity_candidates[self.composition_column] == comp].index)
                    hf_idx.append(high_fidelity_candidates[high_fidelity_candidates[self.composition_column] == comp].index)
                    self.all_LF_idx.append(low_fidelity_candidates[low_fidelity_candidates[self.composition_column] == comp].index)
                    self.all_HF_idx.append(high_fidelity_candidates[high_fidelity_candidates[self.composition_column] == comp].index)

            # query low fidelity candidate for remaining budget
            lowFi_candidates_copy = low_fidelity_candidates.copy()
            if (len(selected_hypotheses) < self.total_budget) & (len(lowFi_candidates_copy) != 0):
                # remained_highFi_cands_fea = highFi_cands_fea.drop(selected_hypotheses.index)
                num_left = self.total_budget - len(selected_hypotheses)
                if low_fidelity_candidates.shape[0] > num_left:
                    selected_hypotheses = selected_hypotheses.append(low_fidelity_candidates[highFi_budget:highFi_budget+num_left])
                else:
                    selected_hypotheses = selected_hypotheses.append(low_fidelity_candidates)

        return selected_hypotheses

    def get_hypotheses(self, candidate_data, seed_data):
        """
        Gets hypotheses using agent.

        Args:
            candidate_data (pd.DataFrame): dataframe of candidates
            seed_data (pd.DataFrame): dataframe of known data

        Returns:
            (pd.DataFrame): dataframe of selected candidates

        """

        features_HF = copy(self.features)
        if self.fidelity == 'multi':
            features_LF = copy(features_HF)
            if self.LF_feature_column in features_LF:
                features_LF.remove(self.LF_feature_column)
        elif self.fidelity == 'single':
            features_LF = None

        # Split the data into training (from seed data) and testing (from candidate data). Need to do this for both
        # the LF set and the HF set.
        X_train_LF, y_train_LF, X_test_LF, y_test_LF, X_train_HF, y_train_HF, X_test_HF, y_test_HF, \
        X_all_LF, X_all_HF, y_all_LF, y_all_HF, X_train_LF_comps, X_test_LF_comps, X_train_HF_comps, X_test_HF_comps = self._process_data_MultiMLagent(
            candidate_data,
            seed_data,
            self.target_prop,
            features_HF=features_HF,
            features_LF=features_LF)

        # Separately preprocess the LF and HF data, because they may not be the same set of materials as the campaign goes on.
        if self.fidelity == 'multi':
            # Fitting the ML model for predicting the LF data
            self.model_LF.fit(X_train_LF[features_LF], y_train_LF)
            y_pred_LF = self.model_LF.predict(X_test_LF[features_LF])
            y_pred_LF_train = self.model_LF.predict(X_train_LF[features_LF])
            y_pred_LF_all = self.model_LF.predict(X_all_LF[features_LF])
            rmse_LF = np.sqrt(mean_squared_error(y_pred_LF, y_test_LF))
            rmse_LF_train = np.sqrt(mean_squared_error(y_pred_LF_train, y_train_LF))
            rmse_LF_all = np.sqrt(mean_squared_error(y_pred_LF_all, y_all_LF))
            self.num_LF_train.append(X_train_LF.shape[0])
            self.LF_train_rmse.append(rmse_LF_train)
            self.LF_test_rmse.append(rmse_LF)
            self.LF_full_rmse.append(rmse_LF_all)

        if self.fidelity == 'multi':
            if self.LF_feature_column not in self.features:
                self.features.append(self.LF_feature_column)

        # Predicting the LF bandgap values for the HF compositions
        if self.fidelity == 'multi':
            y_pred_LF_onHF = self.model_LF.predict(X_train_HF[features_LF])
            y_pred_LF_onHF_test = self.model_LF.predict(X_test_HF[features_LF])

            # Adding in the ML-predicted LF bandgap to the HF data (the bandgap is not normalized)
            X_train_HF['ML PBE bandgap'] = y_pred_LF_onHF
            X_test_HF['ML PBE bandgap'] = y_pred_LF_onHF_test

        # Fitting the ML model for the HF data, with the LF bandgap as an additional feature (if self.fidelity == multi)
        if self.fidelity == 'multi' and self.LF_feature_column not in features_HF:
            features_HF.append(self.LF_feature_column)

        self.model.fit(X_train_HF[features_HF], y_train_HF)
        y_pred = self.model.predict(X_test_HF[features_HF])
        y_pred_train = self.model.predict(X_train_HF[features_HF])
        # NOTE: need to do this b/c the X_HF data updated with new ML PBE bandgaps
        X_ALL_NEW = pd.concat([X_train_HF, X_test_HF])
        Y_ALL_NEW = np.concatenate([y_train_HF, y_test_HF])

        # This was failing because the X_all_HF doesn't have updated ML PBE gap values...
        y_pred_all = self.model.predict(X_ALL_NEW[features_HF])
        rmse_HF_all = np.sqrt(mean_squared_error(y_pred_all, Y_ALL_NEW))

        rmse_HF = np.sqrt(mean_squared_error(y_pred, y_test_HF))
        rmse_HF_train = np.sqrt(mean_squared_error(y_pred_train, y_train_HF))

        self.num_HF_train.append(X_train_HF.shape[0])
        self.HF_train_rmse.append(rmse_HF_train)
        self.HF_test_rmse.append(rmse_HF)
        self.HF_full_rmse.append(rmse_HF_all)

        # Make a copy of the candidate data so the original one
        # does not get modified during hypotheses generation
        candidate_data_copy = candidate_data.copy()

        if self.fidelity == 'multi':
            if self.preprocessor_lf is not None:
                y_pred_LF_candidate = self.model_LF.predict(self.preprocessor_lf.transform(candidate_data_copy[features_LF]))
            else:
                y_pred_LF_candidate = self.model_LF.predict(candidate_data_copy[features_LF])

            # HERE remove addition of ML bandgap
            candidate_data_copy[self.LF_feature_column] = y_pred_LF_candidate
            candidate_data_copy["distance_to_ideal_LF"] = abs(self.target_prop_val - y_pred_LF_candidate)

        if self.preprocessor_hf is not None:
            y_pred_candidate = self.model.predict(self.preprocessor_hf.transform(candidate_data_copy[features_HF]))
        else:
            y_pred_candidate = self.model.predict(candidate_data_copy[features_HF])

        candidate_data_copy["distance_to_ideal"] = abs(self.target_prop_val - y_pred_candidate) # y_pred_candidate

        candidate_data_tosave = copy(candidate_data_copy)

        if self.fidelity == 'multi':
            candidate_data_tosave['LF pred'] = y_pred_LF_candidate
            candidate_data_tosave['HF pred'] = y_pred_candidate

        if self.ranking_method == "exploit":
            candidate_data_copy = candidate_data_copy.sort_values(by=["distance_to_ideal"])
        elif self.ranking_method == "explore":
            gpr = GaussianProcessRegressor(n_restarts_optimizer=10, kernel=ConstantKernel()*Matern()+WhiteKernel())
            gpr.fit(X_train_HF[features_HF], y_train_HF)
            y_pred_gpr, ebar_gpr = gpr.predict(candidate_data_copy[features_HF], return_std=True)
            candidate_data_copy['distance_to_ideal'] = ebar_gpr
            candidate_data_tosave['GPR ebar'] = ebar_gpr
            candidate_data_copy = candidate_data_copy.sort_values(by=["distance_to_ideal"], ascending=False)
            candidate_data_tosave = candidate_data_tosave.sort_values(by=["GPR ebar"], ascending=False)
        elif self.ranking_method == "random":
            candidate_data_copy = candidate_data_copy.sample(frac=1)
            candidate_data_tosave = candidate_data_tosave.sample(frac=1)
        #elif self.ranking_method == 'expected_improvement':
        #    print('Here, doing expected improvement')
        #    gpr = GaussianProcessRegressor(n_restarts_optimizer=10, kernel=ConstantKernel()*Matern()+WhiteKernel())
        #    gpr.fit(X_train_HF[features_HF], y_train_HF)
        #    y_pred_gpr, ebar_gpr = gpr.predict(candidate_data_copy[features_HF], return_std=True)
        #    #try:
        #    #    mean, std = optimizer.predict(X, return_std=True)
        #    y_pred_gpr, ebar_gpr = y_pred_gpr.reshape(-1, 1), ebar_gpr.reshape(-1, 1)
        #    #except NotFittedError:
        #    #    mean, std = np.zeros(shape=(X.shape[0], 1)), np.ones(shape=(X.shape[0], 1))
        #
        #    EI_vals = EI(y_pred_gpr, ebar_gpr, max(y_train_HF), tradeoff=0.5)
        #    #EI_vals = EI()
        #    candidate_data_copy['distance_to_ideal'] = EI_vals
        #    candidate_data_tosave['GPR EI values'] = EI_vals
        #    candidate_data_copy = candidate_data_copy.sort_values(by=["distance_to_ideal"], ascending=False)

        hypotheses = self._query_hypotheses(candidate_data=candidate_data_copy, seed_data=seed_data)

        return hypotheses