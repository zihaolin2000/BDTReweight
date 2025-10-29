import numpy as np
from hep_ml import reweight
import pickle
from numpy.typing import ArrayLike


class Reweighter(reweight.GBReweighter):
    """
    A reweighter class inherited from hep_ml.reweight.GBReweighter,
    with extended functions to predict weights and save to / load
    from pickle object.
    
    """
    
    def __init__(self, n_estimators=40, learning_rate=0.2, max_depth=3, min_samples_leaf=200, loss_regularization=5, gb_args=None):
        super().__init__(n_estimators, learning_rate, max_depth, min_samples_leaf, loss_regularization, gb_args)

    def predict_matched_total_weights(self, original : np.ndarray, original_weight : ArrayLike = None, target_weight : ArrayLike = None) -> ArrayLike:
        """
        hep_ml.reweight's GBReweighter.predict_weights() doesn't
        preserve the total weights after reweight. In this modified
        version, the total weights are  either preserved, or matched
        to target total weights.
        
        Parameters
        ----------
        original : np.ndarray
            The source sample arrays of neutrino MC variables. 
        original_weight : ArrayLike, optional
            The old weights of source sample events.
        target_weight : ArrayLike, optional
            The weights or target sample events. If provided,
            source sample predicted weights' total magnitude
            will be matched to sum(target_weight).

        Returns
        ----------
        ArrayLike
        """
        new_weights = self.predict_weights(original, original_weight=original_weight)
        if target_weight is None:
            # Ensure sum(new_weights) = len(original) 
            new_weights = new_weights * len(new_weights)/np.sum(new_weights)
        else:
            new_weights = (np.sum(target_weight)/(np.sum(new_weights)))*new_weights
        return new_weights

    def save_to_pickle(self, filepath : str):
        """
        Save Reweighter object via pickle.

        Parameters
        ----------
        filepath : str
            The file path to save to.
 
        Returns
        ----------
        None
        """
        with open(filepath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_from_pickle(filepath):
        """
        Load Reweighter object via pickle.

        Parameters
        ----------
        filepath : str
            The file path to load from.
 
        Returns
        ----------
        Reweighter
        """
        with open(filepath, 'rb') as input:
            reweighter = pickle.load(input)
        return reweighter