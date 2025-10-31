import uproot
import numpy as np
from numpy.typing import ArrayLike
import awkward as ak
from .utilities import particle_mass_lookup, particle_pdg_lookup, cosine_theta_vectors

class NuisanceFlatTree:
    """
    A calss to help manipulate interaction event info from NUISANCE
    flat tree.
        
    Attributes
    ----------
    _flattree_vars : ak.Array
        Tree 'FlatTree_VARS' from NUISANCE flat tree root file that
        stores event quantities, such as 'Mode', 'pdg', 'px', etc.
    _total_xsec : float
        Total cross-section in unit of cm^2.
    """

    def __init__(self, rf_path : str | list, **kwargs):
        """
        Initialize the NuisanceFlatTree object with given arguments.

        Parameters
        ----------
        rf_path : str | list
            NUISANCE flat tree root file path, or list of paths
            (str).
        **kwargs : dict, optional
            kwargs for uproot.TTree.arrays() for additional
            filtering.
        
        Returns
        ----------
        None
        """
        if type(rf_path) is str:
            self._flattree_vars = uproot.open(rf_path)['FlatTree_VARS'].arrays(library='ak', **kwargs)
            self._total_xsec = np.sum(self._flattree_vars['fScaleFactor'])
        else:
            trees = []
            # Assuming all samples are generated from the same
            # generator with same preset, and total cross-section
            # in each file is the same. Need to re-evaluate
            # fScaleFactor to preserve total cross-section.
            total_xsec = 0.0
            for path in rf_path:
                tree = uproot.open(path)['FlatTree_VARS'].arrays(library='ak', **kwargs)
                if len(trees) == 0:
                    total_xsec = np.sum(tree['fScaleFactor'])
                trees.append(tree)
            # Concatenate all arrays
            self._flattree_vars = ak.concatenate(trees)
            # Re-evaluate fScaleFactor
            self._flattree_vars['fScaleFactor'] = total_xsec / len(self._flattree_vars)
            self._total_xsec = total_xsec

    def get_tree_array_copy(self) -> ak.highlevel.Array:
        """
        Get a copy of NUISANCE flat tree awkward array.

        Parameters
        ----------
        None

        Returns
        ----------
        ak.Array
            awkward array of flat tree.
        """
        return ak.copy(self._flattree_vars)

    def get_total_xsec(self) -> float:
        """
        Get total cross-section of flat tree events.

        Parameters
        ----------
        None

        Returns
        ----------
        float
            Events total cross-section in unit of cm^2.
        """
        return self._total_xsec
    
    def get_conversion_factor_eventrate_to_xsec(self) -> float:
        """
        Get the conversion factor converting event rate to 
        cross-section. In NUISANCE, fScaleFactor is the same for all
        events in flat tree.

        Parameters
        ----------
        None

        Returns
        ----------
        float
            Conversion factor from event rate to cross-section.
        """
        return self._flattree_vars['fScaleFactor'][0]

    def get_mask_final_state_allowed_pdg(self, pdg_list : list) -> ArrayLike:
        """
        Get the boolean mask for events whose final states
        exclusively have only the specified allowed pdg.

        Parameters
        ----------
        pdg_list : list
            List of integer pdg values.

        Returns
        ----------
        ArrayLike
            Boolean mask for events with specified final state pdgs.
        """
        mask = ak.full_like(self._flattree_vars['pdg'], False, dtype=bool)
        for pdg in pdg_list:
            mask = mask | (self._flattree_vars['pdg']==pdg)
        return ak.all(mask,axis=1)

    def get_mask_flagCCQELike(self) -> ArrayLike:
        """
        Get the boolean mask: flagCCQELike (from NUISANCE flat tree
        directly). 

        Parameters
        ----------
        None

        Returns
        ----------
        ArrayLike
            Boolean mask for CCQE-like events.
        """
        return np.array(self._flattree_vars['flagCCQELike'])

    def get_mask_target_nucleus_A_Z(self, A : int, Z : int) -> ArrayLike:
        """
        Get the boolean mask for events where target nuclei has
        specified atomic number A and Z. 

        Parameters
        ----------
        A : int
            Mass number.
        Z : int
            Atomic number.

        Returns
        ----------
        ArrayLike
            Boolean mask for events with target nuclei of A Z.

        """
        is_A = self._flattree_vars['tgta'] == A
        is_Z = self._flattree_vars['tgtz'] == Z
        return np.logical_and(is_A, is_Z)

    def get_mask_topology(self, particle_counts : dict = {}, KE_thresholds : dict = {}) -> ArrayLike:
        """
        Get the boolean mask for events of entered topology.

        Parameters
        ----------
        particle_counts : dict
            Names and counting rules to specify a event final state
            topology.
            keys : str
                Particle names.
            values : str
                Counting logical rules to applied to number of
                particles.
                Must follow python syntax, as it will be evaluated
                by eval() later.
            For example, 
                {'proton':'>=1', 'muon':'==1', 'pip':'<=1', 
                'neutron':'>0'}
        KE_thresholds : dict, optional
            Kinetic energy thresholds applied to particles. Only
            particles with KE >= KE_threshold are counted.
            keys : str
                String of particle names.
            values : float
                Kinetic energy thresholds (GeV).
            For example,
                {'proton':0.05, neutron:'0.01'}
            If not specified, a threshold of 0.0 GeV is assumed.
        
        Returns
        ----------
        ArrayLike
            Boolean mask for events of entered topology.
            
        """
        count_masks = []
        for particle, rule in particle_counts.items():
            # Default threshold is 0.0 GeV
            KE_threshold = KE_thresholds.get(particle, 0.0)
            n_particles = self.get_n_particles(particle, KE_threshold=KE_threshold)
            count_masks.append(eval(f'n_particles {rule}'))
        # Combine all particle count masks using logical AND
        final_mask = np.full(len(self._flattree_vars), True)
        for count_mask in count_masks:
            final_mask = final_mask & count_mask
        return final_mask

    def get_indices_genie2_drop_fsibug_events(self) -> ArrayLike:
        """
        Get indices of good events with no elastic FSI bug for GENIE
        v2.

        Parameters
        ----------
        None

        Returns
        ----------
        ArrayLike
            int indices of mask.
        """
        indices_1p0n = self.mask_to_indices((self.get_mask_topology({'muon':'==1','proton':'==1','neutron':'==0'})) & (self._flattree_vars['tgta'] > 2))
        arr = self.get_tree_array_copy()
        arr = arr[indices_1p0n]

        v1s = np.array(np.stack([arr['px_vert'][:,5], arr['py_vert'][:,5], arr['pz_vert'][:,5]])).T
        v2s = np.array(np.stack([arr['px'][:,1], arr['py'][:,1], arr['pz'][:,1]])).T
        costheta = cosine_theta_vectors(v1s, v2s)

        # Elastic FSI bug changes FSI proton angle unphysically.
        indices_FSIbug = indices_1p0n[costheta < 0.9999996] # Give a machine tolerance
        indices_tree = np.arange(0, len(self._flattree_vars))
        indices_good = np.delete(indices_tree, indices_FSIbug)
        return indices_good

    def mask_to_indices(self, mask : ArrayLike) -> ArrayLike:
        """
        Convert boolean mask to integer indices of events.

        Parameters
        ----------
        mask : ArrayLike
            Boolean masking for events. 
        Returns
        ----------
        ArrayLike
            int indices of mask.
        """
        return np.where(mask)[0]

    def update_tree_with_mask(self, mask : ArrayLike) -> None:
        """
        Update tree content by filtering out masked events.
        This will discard unmasked entries.

        Parameters
        ----------
        mask : 1d int or bool tuple
            Masking applied to slef._flattree_vars. 

        Returns
        ----------
        None
        """
        self._flattree_vars = self._flattree_vars[mask]

    def get_n_particles(self, particle : str, KE_threshold : float = 0.0, mask : ArrayLike = None) -> ak.highlevel.Array:
        """
        Count the number of particles of quest from each event.

        Parameters
        ----------
        particle : str
            Particle of quest. 
            Particle options: 'muon','electron','proton','neutron',
            'photon','pip','pim','pi0'
        KE_threshold : float
            A kinetic energy threshold applied to particle. Only 
            particles with KE >= KE_threshold are counted. 
        mask : 1d int or bool tuple, optional
            Masking applied to slef._flattree_vars before selection.

        Returns
        ----------
        ak.Array
            Per-event 1-dimensional ak.Array of number of particles.
        """
        num_particles = []

        if particle not in ['muon', 'electron', 'proton', 'neutron', 'photon', 'pip', 'pim', 'pi0']:
            raise ValueError(f'Particle not registered: {particle}')
    
        # default masking: all entries
        if mask is None:
            mask = np.full(len(self._flattree_vars), True)

        # create particle mask by matching pdg 
        is_particle = self._flattree_vars[mask]['pdg'] == particle_pdg_lookup(particle)
        if KE_threshold == 0.0:
            # no KE_threshold applied. Count particles directly.
            num_particles = ak.sum(is_particle, axis = 1)
        else:
            # sum the number of particles that has KE above
            # KE_threshold
            particle_KE = self._flattree_vars[mask]['E'][is_particle] - particle_mass_lookup(particle)
            above_KE_threshold = particle_KE  >=  KE_threshold
            num_particles = ak.sum(above_KE_threshold, axis = 1)

        return num_particles
    
    def get_event_variable(self, expr : str, mask : ArrayLike = None) -> ak.highlevel.Array:
        """
        Get the event-level quantity from NUISANCe flat tree 
        (self._flattree_vars).

        Parameters
        ----------
        expr : str
            Quantity expression to quest event info. 
            Supported expr: 
            NUISANCE keys of per-event quantity, or expr of
            form 'selector_particle_variable'.
            Selector options: 'leading','subleading','total'
            Particle options: 'muon','electron','proton','neutron',
                'photon','pip','pim','pi0'
            Variable options: 'px','py','pz','E','KE'
        mask : 1d int or bool tuple, optional
            Masking applied to self._flattree_vars before selection.

        Returns
        ----------
        ak.Array
            Per-event 1-dimensional Awkward Array of the requested
            quantity.
        """

        # default masking: all entries
        if mask is None:
            mask = np.full(len(self._flattree_vars), True) 

        selected = []

        if expr in self._flattree_vars.fields:
            # expr is key of any event-level quantity provided in
            # NUISANCE flattree event record format. 
            # Options:
            # ['Mode', 'cc', 'PDGnu', 'Enu_true', 'tgt', 'tgta',
            # 'tgtz', 'PDGLep', 'ELep', 'CosLep', 'Q2', 'q0', 'q3',
            #  'Enu_QE', 'Q2_QE', 'W_nuc_rest', 'W', 'W_genie', 'x',
            #  'y', 'Eav', 'EavAlt', 'CosThetaAdler', 'PhiAdler',
            # 'dalphat', 'dpt', 'dphit', 'pnreco_C', 'nfsp', 'px',
            # 'py', 'pz', 'E', 'pdg', 'pdg_rank', 'ninitp',
            # 'px_init', 'py_init', 'pz_init', 'E_init', 'pdg_init',
            #  'nvertp', 'px_vert', 'py_vert', 'pz_vert', 'E_vert',
            # 'pdg_vert', 'Weight', 'InputWeight', 'RWWeight',
            # 'fScaleFactor', 'CustomWeight', 'CustomWeightArray',
            # 'flagCCINC', 'flagNCINC', 'flagCCQE', 'flagCC0pi',
            # 'flagCCQELike', 'flagNCEL', 'flagNC0pi', 'flagCCcoh',
            # 'flagNCcoh', 'flagCC1pip', 'flagNC1pip', 'flagCC1pim',
            # 'flagNC1pim', 'flagCC1pi0', 'flagNC1pi0',
            # 'flagCC0piMINERvA', 'flagCC0Pi_T2K_AnaI',
            # 'flagCC0Pi_T2K_AnaII']
            selected = self._flattree_vars[mask][expr]
        else:
            # expr is in the form of 'selector_particle_variable'
            selector, particle, variable = expr.split('_')
            if selector not in ['leading','subleading', 'total']:
                raise ValueError(f'Selector not registered: {selector}')
            if particle not in ['muon', 'electron', 'proton', 'neutron', 'photon', 'pip', 'pim', 'pi0']:
                raise ValueError(f'Particle not registered: {particle}')
            if variable not in ['px', 'py', 'pz', 'E', 'KE']:
                raise ValueError(f'Variable not registered: {variable}')

            # create particle mask by matching pdg 
            is_particle = self._flattree_vars[mask]['pdg'] == particle_pdg_lookup(particle)

            if selector == 'leading':
                if variable == 'E':
                    selected = ak.max(self._flattree_vars[mask]['E'][is_particle], axis=1)
                elif variable == 'KE':
                    selected = ak.max(self._flattree_vars[mask]['E'][is_particle], axis=1) - particle_mass_lookup(particle)
                else:
                    # sort E to find the index of leading particle
                    leading_idx = ak.argmax(self._flattree_vars[mask]['E'][is_particle], axis=1, keepdims=True)
                    selected = ak.firsts(self._flattree_vars[mask][variable][is_particle][leading_idx])
            elif selector == 'subleading':
                # sort the particle energy in descending order
                # (0: highest, 1: second highest...)
                order = ak.argsort(self._flattree_vars[mask]['E'][is_particle], axis=1, ascending=False)
                # use ak.pad_none to fill None when there's no 2nd
                # particle
                if variable == 'KE':
                    selected = ak.pad_none(self._flattree_vars[mask]['E'][is_particle][order],2)[:,1] - particle_mass_lookup(particle)
                else:
                    selected = ak.pad_none(self._flattree_vars[mask][variable][is_particle][order],2)[:,1]
            else: # selector == 'leading'
                # when final state has no requested particle, the sum
                # is set to None
                if variable == 'KE':
                    # subtract mass * no. of particle from total E
                    selected = ak.sum(self._flattree_vars[mask]['E'][is_particle], axis=1, mask_identity=True) - particle_mass_lookup(particle)*ak.sum(is_particle, axis=1)
                else:
                    selected = ak.sum(self._flattree_vars[mask][variable][is_particle], axis=1, mask_identity=True)

        return selected

