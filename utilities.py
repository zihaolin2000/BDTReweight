import numpy as np
from numpy.typing import ArrayLike

PARTICLE_PDG = {
    'proton':2212,
    'neutron':2112,
    'muon':13,
    'electron':11,
    'numu':14,
    'nue':12,
    'photon':22,
    'pip':211,
    'pim':-211,
    'pi0':111,
    'genieBindino':2000000101 # GENIE special particle: binding energy subtracted from f/s nucleons
}

PARTICLE_MASS = {
    'proton':0.93827,
    'neutron':0.93957,
    'muon':0.10566,
    'electron':0.00511,
    'numu':0.0,
    'nue':0.0,
    'photon':0.0,
    'pip':0.13957,
    'pim':0.13957,
    'pi0':0.13498,
    'genieBindino':0.0
}

PARTICLE_LATEX_SYMBOL = {
    'proton':'p',
    'neutron':'n',
    'muon':'\mu^-',
    'electron':'\e^-',
    'numu':'\\nu_\mu',
    'nue':'\\nu_\e',
    'photon':'\gamma',
    'pip':'\pi^+',
    'pim':'\pi^-',
    'pi0':'\pi^0',
    'genieBindino':'\\text{genieBindino}'
}

VARIABLE_LATEX_SYMBOL = {
    'px':'p_x',
    'py':'p_y',
    'pz':'p_z',
    'E':'E',
    'KE':'T',
    'theta':'\\theta',
    'dalphat':'\delta\\alpha_T',
    'dphit':'\delta\phi_T',
    'dpt':'\delta p_T',
    'nu':'\\nu',
    'Q2':'Q^2',
    'q0':'q_0',
    'q3':'|\mathbf{q}|',
    'Enu_true':'E_{\\nu,\\text{true}}',
    'weight':'\\text{weight}'
}

VARIABLE_UNIT = {
    'px':'\\text{GeV}/c',
    'py':'\\text{GeV}/c',
    'pz':'\\text{GeV}/c',
    'E':'\\text{GeV}',
    'KE':'\\text{GeV}',
    'mass':'\\text{GeV}/c^2',
    'theta':'\\text{rad}',
    'dalphat':'\\text{rad}',
    'dphit':'\\text{rad}',
    'dpt':'\\text{GeV}/c',
    'nu':'\\text{GeV}',
    'Q2':'\\text{GeV}^{2}/c^2',
    'q0':'\\text{GeV}',
    'q3':'\\text{GeV}/c',
    'Enu_true':'\\text{GeV}'
}

def particle_variable_to_latex(expr : str, add_unit : bool = True) -> str:
    """
    Return the latex symbol string of given particle variable
    expression.

    Parameters
    ----------
    expr : str
        string of particle variable expression.
    add_unit : bool
        If True, '(unit)' will be appended to the symbol.
        default: True. 
    
    Returns
    ----------
    str
        Latex string of particle variable.
    """

    if expr in VARIABLE_LATEX_SYMBOL.keys():
        unit = f' $\left({VARIABLE_UNIT[expr]}\\right)$' if (add_unit is True and expr in VARIABLE_UNIT.keys()) else ''
        return f'${VARIABLE_LATEX_SYMBOL[expr]}$' + unit
        
    selector, particle, variable = expr.split('_')
    particle_symbol = PARTICLE_LATEX_SYMBOL.get(particle, 'particle')
    variable_symbol = VARIABLE_LATEX_SYMBOL.get(variable, 'variable')
    if variable == 'KE':
        variable_symbol += '_{' + particle_symbol + '}'

    if selector in ['leading', 'subleading']:
        if particle in ['proton', 'neutron']:
            latex = f'{selector} {particle} ${variable_symbol}$'
        else:
            latex = f'{particle} ${variable_symbol}$'
    elif selector == 'total':
        latex = '$\sum_{\\text{' + particle + '}}$ ${' + variable_symbol + '}$'
    unit = f' $\left({VARIABLE_UNIT[variable]}\\right)$' if (add_unit is True and variable in VARIABLE_UNIT.keys()) else ''
    return latex + unit

def diff_xsec_latex_wrt_variable(expr : str, add_unit : bool = True) -> str:
    """
    Return the latex symbol string of differential cross-section
    w.r.t. given particle variable expression.

    Parameters
    ----------
    expr : str
        string of particle variable expression.
    add_unit : bool
        If True, '(unit)' will be appended to the symbol.
        default: True. 

    Returns
    ----------
    str
        Latex string of differential cross-section w.r.t. particle
        variable.
    """
    if expr in VARIABLE_LATEX_SYMBOL.keys():
        unit = ' $\left(\\frac{\\text{cm}^2}{'+VARIABLE_UNIT[expr]+'}\\right)$' if (add_unit is True and expr in VARIABLE_UNIT.keys()) else ''
        return '$\\frac{d\sigma}{d'+VARIABLE_LATEX_SYMBOL[expr]+'}$' + unit
    else:
        _, _, variable = expr.split('_')
        unit = ' $\left(\\frac{\\text{cm}^2}{'+VARIABLE_UNIT[variable]+'}\\right)$' if (add_unit is True and variable in VARIABLE_UNIT.keys()) else ''
        return '$\\frac{d\sigma}{d'+VARIABLE_LATEX_SYMBOL[variable]+'}$' + unit

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

def angle_between_vectors(v1s : np.ndarray, v2s : np.ndarray) -> ArrayLike:
    pass

def TKI_variables(lepton_Ps : np.ndarray, p_nucleon_Ps : np.ndarray) -> ArrayLike:
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

def scalar_component_vectors(v1s : np.ndarray, v2s : np.ndarray) -> ArrayLike:
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

def cosine_theta_vectors(v1s : np.ndarray, v2s : np.ndarray) -> ArrayLike:
    """
    Treat entry v1,v2 in v1s,v2s as physical vector pairs, and
    calculate cos(theta) for angle theta between v1,v2.

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
        float array of cos(theta).
    """
    return np.sum(normalize_vectors(v1s) * normalize_vectors(v2s), axis = 1)

def efficiency(cosT : ArrayLike, Tp : ArrayLike):
    """
    Return toy model efficiency values for given proton cosine(theta), cosT, and
    kinetic energy, Tp, see arxiv.org/abs/2510.07463.

    Parameters
    ----------
    cosT : ArrayLike
        cosine of proton angle w.r.t. neutrino direction.
    Tp : ArrayLike
        Kinetic energy of proton in GeV.

    Returns
    ----------
    np.array
    """
    return np.minimum(np.maximum((Tp * cosT - 0.060) / 0.060, 0), 1.0)

def MNEff_evaluate(df = None, xybins = (np.linspace(0,0.6,20),np.linspace(0,2,20)), reweight=False, Xsec_columns=('dpt','pT_muon')):
    # TODO: Change Styling. Add helper info. 

    if 'pT_muon' not in df.columns:
        df['pT_muon'] = - df['leading_muon_py']
    if 'eff' not in df.columns:
        P_protons = np.array(df[['leading_proton_px', 'leading_proton_py', 'leading_proton_pz']])
        P_zvector = np.zeros_like(P_protons)
        P_zvector[:,2] = 1
        cos_proton = cosine_theta_vectors(P_protons, P_zvector)
        T_proton = np.sqrt(df['leading_proton_px']**2 + df['leading_proton_py']**2 + df['leading_proton_pz']**2)
        print('cos_proton', cos_proton)
        print('T_proton', T_proton)
        df['eff'] = efficiency(cos_proton, T_proton)
    xcol, ycol = Xsec_columns
    N, dpt_edges, pT_edges = np.histogram2d(df[xcol],df[ycol],bins=xybins,weights=df['weight'])
    N = np.zeros(N.shape)
    Nerr = np.zeros(N.shape)
    M = np.zeros(N.shape)
    Merr = np.zeros(N.shape)
    K = np.zeros(N.shape)
    R = np.zeros(N.shape)
    Rerr = np.zeros(N.shape)
    for i in range(len(dpt_edges)-1):
        for j in range(len(pT_edges)-1):
            df_bin = df.loc[(df[xcol]>=dpt_edges[i])&(df[xcol]<dpt_edges[i+1])
                &(df[ycol]>=pT_edges[j])&(df[ycol]<pT_edges[j+1])].copy()
            if reweight == False:
                df_bin = df_bin.copy()
                df_bin['weight'] = 1.0
                
            N[i,j] = df_bin['weight'].sum()

            Nerr[i,j] = np.sqrt((df_bin['weight']**2).sum())
            M[i,j] = (df_bin['eff']*df_bin['weight']).sum()
            Merr[i,j] = np.sqrt((( df_bin['eff'] * df_bin['weight'] )**2).sum())

            K[i,j] = len(df_bin)

            R[i,j] = M[i,j]/N[i,j]
            cov = np.sum(df_bin['eff']*df_bin['weight']**2)

            Rerr[i,j] = R[i,j]*np.sqrt(
                (Nerr[i,j]/N[i,j])**2
                +(Merr[i,j]/M[i,j])**2
                -2*cov/(N[i,j]*M[i,j])
            )

    return M, N, R, dpt_edges, pT_edges, Nerr, Merr, Rerr
