import pandas as pd
import numpy as np
import awkward as ak
from .utilities import normalize_vectors
from .nuisance_flat_tree import NuisanceFlatTree

def transform_momentum_to_reaction_frame(df : pd.DataFrame, selector_lepton : str = 'leading_muon', particle_names : list = []) -> pd.DataFrame:
    """
    Convert particle momentum from lab frame to reaction frame.
    In both frames, neutrino direction is +z. Reaction frame is
    defined by rotating lab frame about z-axis, such that the
    coplane of neutrino and lepton directions form the yz plane.
    Lepton transverse direction is chosen as -y direction.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing final state particle 3-momenta.
        Assume columns contain particle names end with
        '_px', '_py', and '_pz'.
    selector_lepton : str
        The final state lepton that defines reaction frame.
    particle_names : list of str
        The list of names of final state particles whose
        3-momenta will be transformed. 

    Returns
    ----------
    pd.DataFrame
    """

    df_new = df.copy()

    #______tranform lepton's momenta to reaction frame______________
    # reaction frame lepton px is 0 by construction
    df_new[f'{selector_lepton}_px'] = np.zeros(len(df))
    # reaction frame lepton py has magnitude of tranverse momentum,
    # i.e, norm([px, py]), point in -y direction
    df_new[f'{selector_lepton}_py'] = - np.linalg.norm(df[[f'{selector_lepton}_px', f'{selector_lepton}_py']], axis=1)
    # reaction frame lepton pz stays unchanged
    df_new[f'{selector_lepton}_pz'] = df[f'{selector_lepton}_pz']

    #______tranform other particles' momenta to reaction frame______
    # take negative of lepton transverse momentum as y-vector
    transverse_plane_y = - df[[f'{selector_lepton}_px', f'{selector_lepton}_py']].values
    # normalize to get unit y-vector
    transverse_plane_y = normalize_vectors(transverse_plane_y)
    # x-vector is simply y-vector rotated counterclock wise by 90
    # degrees
    transverse_plane_x = np.array([-transverse_plane_y[:,1],transverse_plane_y[:,0]]).T
    for particle_name in particle_names:
        transverse_P = df[[f'{particle_name}_px', f'{particle_name}_py']].values
        # reaction frame particle px, py is the projection of 
        # transverse momentum onto unit x, y vector
        df_new[f'{particle_name}_px'] = np.sum(transverse_P * transverse_plane_x, axis=1)
        df_new[f'{particle_name}_py'] = np.sum(transverse_P * transverse_plane_y, axis=1)
        # reaction frame particle pz stays unchanged
        df_new[f'{particle_name}_pz'] = df[f'{particle_name}_pz']
    
    return df_new

def create_dataframe_from_nuisance(tree : NuisanceFlatTree, variable_exprs : list = [], mask : np.ndarray = None) -> pd.DataFrame:
    """
    Create a dataframe from NuisanceFlatTree with list of 
    variable expressions specified for event-level quantity.

    Parameters
    ----------
    tree : NuisanceFlatTree
        NUISANCE flat tree object.
    variable_exprs : list of str
        List of strings for variable expressions in the form of
        'selector_particle_variable', see description at
        NuisanceFlatTree.get_event_variable().

    Returns
    ----------
    pd.DataFrame
    """

    if mask is None:
        mask = np.full(len(tree._flattree_vars), True)

    df = pd.DataFrame()
    for expr in variable_exprs:
        variable = tree.get_event_variable(expr, mask = mask)
        # convert to numpy. Replace ak.None to np.NaN.
        np_variable = ak.fill_none(variable, np.nan).to_numpy()
        df[expr] = np_variable
    
    return df
