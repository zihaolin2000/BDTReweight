import numpy as np

PARTICLE_PDG = {
    'proton':2212,
    'neutron':2112,
    'muon':13,
    'electron':11,
    'numu':14,
    'nue':12,
    'gamma':22,
    'pi+':211,
    'pi-':-211,
    'pi0':111
}

PARTICLE_MASS = {
    'proton':0.93827,
    'neutron':0.93957,
    'muon':0.10566,
    'electron':0.00511,
    'numu':0.0,
    'nue':0.0,
    'gamma':0.0,
    'pi+':0.13957,
    'pi-':0.13957,
    'pi0':0.13498
}

def particle_pdg_lookup(particle : str) -> int:
    """
    Return the pdg code of given particle name.

    Parameters
    ----------
    particle : str
        Particle name.

    Returns
    ----------
    int
    """
    return PARTICLE_PDG[particle]

def particle_mass_lookup(particle : str) -> float:
    """
    Return the mass (in GeV/c^2) of given particle name.

    Parameters
    ----------
    particle : str
        Particle name.

    Returns
    ----------
    float
    """
    return PARTICLE_MASS[particle]

def angle_between_vectors(v1s : np.ndarray, v2s : np.ndarray) -> np.ndarray:
    pass

def TKI_variables(lepton_Ps : np.ndarray, p_nucleon_Ps : np.ndarray) -> tuple:
    pass

def normalize_vectors(vectors : np.ndarray) -> np.ndarray:
    """
    Treat each entry in vectors as a vector and normalize it to unit.

    Parameters
    ----------
    vectors : np.ndarray
        2d array where each entry is a physical vector. 

    Returns
    ----------
    np.ndarray
    """
    return vectors / (np.linalg.norm(vectors, axis=1)[:,None])

def scalar_component_vectors(v1s : np.ndarray, v2s : np.ndarray) -> np.ndarray:
    """
    Treat entry v1,v2 in v1s,v2s as physical vector pairs, and
    calculate scalar projection of v1 onto v2 for pair.

    Parameters
    ----------
    v1s : np.ndarray
        2d array where each entry is a physical vector.
    v2s : np.ndarray
        2d array where each entry is a physical vector. Must has the
        same shape as v1s.

    Returns
    ----------
    np.ndarray
    """
    v2s_unit = normalize_vectors(v2s)
    return np.sum(v1s * v2s_unit, axis=1)

