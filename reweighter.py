import numpy as np
from hep_ml import reweight
import pickle


class Reweighter(reweight.GBReweighter):
    """
    A reweighter class inherited from hep_ml.reweight.GBReweighter,
    with extended functions to predict weights and save to / load
    from pickle object.
    
    """
    
    def __init__(self, n_estimators=40, learning_rate=0.2, max_depth=3, min_samples_leaf=200, loss_regularization=5, gb_args=None):
        super().__init__(n_estimators, learning_rate, max_depth, min_samples_leaf, loss_regularization, gb_args)

    def predict_matched_total_weights(self, original, original_weight=None, target_weight=None) -> np.ndarray:
        new_weights = self.predict_weights(original, original_weight=original_weight)
        if target_weight is None:
            # Ensure sum(new_weights) = len(original) 
            new_weights = new_weights * len(new_weights)/np.sum(new_weights)
        else:
            new_weights = (np.sum(target_weight)/(np.sum(new_weights)))*new_weights
        return new_weights

    def save_to_pickle(self, path):
        with open(path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(path):
        with open(path, 'rb') as input:
            reweighter = pickle.load(input)
        return reweighter