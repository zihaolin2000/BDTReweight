import uproot
import numpy as np

class NUISANCEEvent:
    """Container for one entry of a NUISANCE flat tree."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.SelectedSample = -1  # Placeholder for SelectedSample attribute

        # muon acceptance
        self.pmu_threshold = 0.0  # GeV/c
        self.costhetamu_threshold = 0.5
        # piplus acceptance
        self.ppiplus_threshold = 0.2  # GeV/c
        self.costhetapiplus_threshold = 0.5
        # piminus acceptance
        self.ppiminus_threshold = 0.2  # GeV/c
        self.costhetapiminus_threshold = 0.5
        # proton acceptance
        self.pproton_threshold = 0.45  # GeV/c
        self.costhetaproton_threshold = 0.5

    def __repr__(self):
        keys = list(self.__dict__.keys())
        sample = ", ".join(f"{k}={getattr(self,k)!r}" for k in keys[:5])
        if len(keys) > 5:
            sample += ", ..."
        return f"<NUISANCEEvent {sample}>"

    def Pmu(self):
        """Calculate muon momentum from Px, Py, Pz."""
        # loop through the pdg attribute (is a list) to find the muon (pdg = 13)
        pdg = np.array(self.pdg)
        muon_mask = (pdg == 13) | (pdg == -13)
        if not np.any(muon_mask):
            return None  # No muon found
        Px = np.array(self.px)[muon_mask]
        Py = np.array(self.py)[muon_mask]
        Pz = np.array(self.pz)[muon_mask]

        if len(Px) > 1:
            # Take leading muon if multiple muons found
            max_index = np.argmax(np.sqrt(Px**2 + Py**2 + Pz**2))
            return np.sqrt(Px[max_index]**2 + Py[max_index]**2 + Pz[max_index]**2)

        return np.sqrt(Px[0]**2 + Py[0]**2 + Pz[0]**2)

    def CosThetamu(self):
        """Calculate muon cos(theta) from Px, Py, Pz."""
        pdg = np.array(self.pdg)
        muon_mask = (pdg == 13) | (pdg == -13)
        if not np.any(muon_mask):
            return None  # No muon found
        Px = np.array(self.px)[muon_mask]
        Py = np.array(self.py)[muon_mask]
        Pz = np.array(self.pz)[muon_mask]
        if len(Px) > 1:
            # If multiple muons, take leading muon
            max_index = np.argmax(np.sqrt(Px**2 + Py**2 + Pz**2))
            P = np.sqrt(Px[max_index]**2 + Py[max_index]**2 + Pz[max_index]**2)
            return Pz[max_index] / P

        P = np.sqrt(Px[0]**2 + Py[0]**2 + Pz[0]**2)
        return Pz[0] / P

    def vectorPmu(self):
        """Return the muon momentum vector (Px, Py, Pz) as a list."""



        pdg = np.array(self.pdg)
        muon_mask = (pdg == 13) | (pdg == -13)
        if not np.any(muon_mask):
            return None  # No muon found
        Px = np.array(self.px)[muon_mask]
        Py = np.array(self.py)[muon_mask]
        Pz = np.array(self.pz)[muon_mask]
        if len(Px) > 1:
            # If multiple muons, take leading muon
            max_index = np.argmax(np.sqrt(Px**2 + Py**2 + Pz**2))
            return [Px[max_index], Py[max_index], Pz[max_index]]

        return [Px[0], Py[0], Pz[0]]

    def print_gamma_info(self):
        """Print information about gamma particles in the event."""
        pdg = np.array(self.pdg)
        gamma_mask = (pdg == 22)
        if not np.any(gamma_mask):
            # print("No gamma particles found in this event.")
            return
        Px = np.array(self.px)[gamma_mask]
        Py = np.array(self.py)[gamma_mask]
        Pz = np.array(self.pz)[gamma_mask]
        E = np.array(self.E)[gamma_mask]
        for i in range(len(Px)):
            P = np.sqrt(Px[i]**2 + Py[i]**2 + Pz[i]**2)
            print(f"Gamma {i+1}: E={E[i]}")

    def PerformSelection(self, isFHC):
        """Perform a simple signal selection based on presence of muon and pion

        Parameters
        ----------
        isFHC : bool
            True if the beam mode is FHC, False for RHC.

        Returns
        -------
        SelectedSample code : int

        """
        if isFHC:
            self.PerformSelectionFHC()
        else:
            self.PerformSelectionRHC()

        return self.SelectedSample

    def PerformSelectionFHC(self):
        """Perform selection for FHC mode."""
        pdg = np.array(self.pdg)
        has_muon = np.any(pdg == 13)
        has_antimuon = np.any(pdg == -13)
        if has_antimuon and has_muon:
            return -1  # Not selected: both muon and antimuon present

        if has_muon:
            muon_pmu = self.Pmu()
            muon_costhetamu = self.CosThetamu()
            has_muon = has_muon and (muon_pmu > self.pmu_threshold) and (muon_costhetamu > self.costhetamu_threshold)

        num_piplus = 0
        piplus_mask = (pdg == 211)
        if np.any(piplus_mask):
            Px_piplus = np.array(self.px)[piplus_mask]
            Py_piplus = np.array(self.py)[piplus_mask]
            Pz_piplus = np.array(self.pz)[piplus_mask]
            for i in range(len(Px_piplus)):
                P_piplus = np.sqrt(Px_piplus[i]**2 + Py_piplus[i]**2 + Pz_piplus[i]**2)
                costheta_piplus = Pz_piplus[i] / P_piplus
                if P_piplus > self.ppiplus_threshold and costheta_piplus > self.costhetapiplus_threshold:
                    num_piplus += 1

        if has_muon:
            if num_piplus == 0:
                self.SelectedSample = 1
            elif num_piplus == 1:
                self.SelectedSample = 2
            else:
                self.SelectedSample = 3
        else:
            self.SelectedSample = -1  # Not selected

    def PerformSelectionRHC(self):
        """Perform selection for RHC mode."""
        pdg = np.array(self.pdg)
        has_antimuon = np.any(pdg == -13)
        has_muon = np.any(pdg == 13)
        if has_antimuon and has_muon:
            return -1  # Not selected: both muon and antimuon present

        if has_muon:
            muon_pmu = self.Pmu()
            muon_costhetamu = self.CosThetamu()
            has_muon = has_muon and (muon_pmu > self.pmu_threshold) and (muon_costhetamu > self.costhetamu_threshold)

        if has_antimuon:
            antimuon_mask = (pdg == -13)
            Px_antimuon = np.array(self.px)[antimuon_mask]
            Py_antimuon = np.array(self.py)[antimuon_mask]
            Pz_antimuon = np.array(self.pz)[antimuon_mask]
            if len(Px_antimuon) > 1:
                raise ValueError("Multiple antimuons found in event.")
            P_antimuon = np.sqrt(Px_antimuon[0]**2 + Py_antimuon[0]**2 + Pz_antimuon[0]**2)
            costheta_antimuon = Pz_antimuon[0] / P_antimuon
            has_antimuon = has_antimuon and (P_antimuon > self.pmu_threshold) and (costheta_antimuon > self.costhetamu_threshold)

        num_piplus = 0
        piplus_mask = (pdg == 211)
        if np.any(piplus_mask):
            Px_piplus = np.array(self.px)[piplus_mask]
            Py_piplus = np.array(self.py)[piplus_mask]
            Pz_piplus = np.array(self.pz)[piplus_mask]
            for i in range(len(Px_piplus)):
                P_piplus = np.sqrt(Px_piplus[i]**2 + Py_piplus[i]**2 + Pz_piplus[i]**2)
                costheta_piplus = Pz_piplus[i] / P_piplus
                if P_piplus > self.ppiplus_threshold and costheta_piplus > self.costhetapiplus_threshold:
                    num_piplus += 1

        num_piminus = 0
        piminus_mask = (pdg == -211)
        if np.any(piminus_mask):
            Px_piminus = np.array(self.px)[piminus_mask]
            Py_piminus = np.array(self.py)[piminus_mask]
            Pz_piminus = np.array(self.pz)[piminus_mask]
            for i in range(len(Px_piminus)):
                P_piminus = np.sqrt(Px_piminus[i]**2 + Py_piminus[i]**2 + Pz_piminus[i]**2)
                costheta_piminus = Pz_piminus[i] / P_piminus
                if P_piminus > self.ppiminus_threshold and costheta_piminus > self.costhetapiminus_threshold:
                    num_piminus += 1

        if has_antimuon:
            if num_piminus == 0:
                self.SelectedSample = 59
            elif num_piminus == 1:
                self.SelectedSample = 60
            else:
                self.SelectedSample = 61
        elif has_muon:
            # Check for nu_mu background
            has_muon = np.any(pdg == 13)
            if has_muon:
                if num_piplus == 0:
                    self.SelectedSample = 71
                elif num_piplus == 1:
                    self.SelectedSample = 72
                else:
                    self.SelectedSample = 73
            else:
                self.SelectedSample = -1 # Not selected
        else:
            self.SelectedSample = -1  # Not selected


    def MINERvACCQELikeSelection(self, neutrinoMode=True, remove_proton_KE=False):
        """Perform CCQE-Like selection as it is done in CCQENuCuts.cxx in the MAT (MINERVA analysis toolkit) code.

        cpp version of the code:
        {
          int genie_n_muons         = 0;
          int genie_n_mesons        = 0;
          int genie_n_heavy_baryons_plus_pi0s = 0;
          int genie_n_photons       = 0;
          int genie_n_protons       = 0; //antinu

          for(int i = 0; i < mc_nFSPart; ++i) {
            int pdg =  mc_FSPartPDG[i];
            double energy = mc_FSPartE[i];
            double proton_E = 1058.272;
            //removing the 1020 MeV proton KE cut as per Minerba's Suggestion.
            if(remove_proton_KE)proton_E=938.28;
            //The photon energy cut is hard-coded at 10 MeV at present. We're happy to make it general, if the need arises !
            if( abs(pdg) == 13 ) genie_n_muons++;
            else if( pdg == 22 && energy >10 ) genie_n_photons++;
            else if( abs(pdg) == 211 || abs(pdg) == 321 || abs(pdg) == 323 || pdg == 111 || pdg == 130 || pdg == 310 || pdg == 311 || pdg == 313 ) genie_n_mesons++;
            else if( pdg == 3112 || pdg == 3122 || pdg == 3212 || pdg == 3222 || pdg == 4112 || pdg == 4122 || pdg == 4212 || pdg == 4222 || pdg == 411 || pdg == 421 || pdg == 111 ) genie_n_heavy_baryons_plus_pi0s++;
            else if( pdg == 2212 && energy > proton_E ) genie_n_protons++; //antinu
          }

          //Definition of CCQE-like: 1 muon (from neutrino) and no mesons/heavy baryons in final state
          //Any number of final state nucleons (protons or neutrons) allowed
          //Photons from nuclear de-excitation are kept. These tend to be < 10 MeV. Events with photons from other sources are excluded.
          //GENIE simulates nuclear de-excitations only for Oxygen atoms at present.
          if(neutrinoMode){
            if( genie_n_muons         == 1 &&
            genie_n_mesons        == 0 &&
            genie_n_heavy_baryons_plus_pi0s == 0 &&
            genie_n_photons       == 0 ) return true;
          }
          else{
            if( genie_n_muons         == 1 &&
            genie_n_mesons        == 0 &&
            genie_n_heavy_baryons_plus_pi0s == 0 &&
            genie_n_photons       == 0 ) return true;
          }
          return false;
        }


        Returns
        -------
        bool
            True if event passes CCQE-Like selection, False otherwise.
        """
        pdg = np.array(self.pdg)
        E = np.array(self.E)

        genie_n_muons = np.sum(np.abs(pdg) == 13)

        photon_mask = (pdg == 22)
        genie_n_photons = np.sum(photon_mask & (E > 10))

        meson_pdgs = {211, -211, 321, -321, 323, -323, 111, 130, 310, 311, -311, 313, -313}
        genie_n_mesons = np.sum(np.isin(pdg, list(meson_pdgs)))

        heavy_baryon_pdgs = {3112, 3122, 3212, 3222, 4112, 4122, 4212, 4222, 411, -411, 421, -421}
        genie_n_heavy_baryons_plus_pi0s = np.sum(np.isin(pdg, list(heavy_baryon_pdgs)) | (pdg == 111))

        proton_mask = (pdg == 2212)
        proton_E = 938.28
        genie_n_protons = np.sum(proton_mask)  

        if neutrinoMode:
            if genie_n_muons == 1 and genie_n_mesons == 0 and genie_n_heavy_baryons_plus_pi0s == 0 and genie_n_photons == 0:
                return True
        else:
            if genie_n_muons == 1 and genie_n_mesons == 0 and genie_n_heavy_baryons_plus_pi0s == 0 and genie_n_photons == 0:
                return True

        return False

    def print_final_state_particles(self):
        """Printout number of muons, neutrons, protons, pions, and photons in the final state."""
        pdg = np.array(self.pdg)
        E = np.array(self.E)

        genie_n_muons = np.sum(np.abs(pdg) == 13)

        photon_mask = (pdg == 22)
        genie_n_photons = np.sum(photon_mask)
        genie_n_photons_above10MeV = np.sum(photon_mask & (E > 10))

        meson_pdgs = {211, -211, 321, -321, 323, -323, 111, 130, 310, 311, -311, 313, -313, 111}
        genie_n_mesons = np.sum(np.isin(pdg, list(meson_pdgs)))

        heavy_baryon_pdgs = {3112, 3122, 3212, 3222, 4112, 4122, 4212, 4222, 411, -411, 421, -421}
        genie_n_heavy_baryons_plus_pi0s = np.sum(np.isin(pdg, list(heavy_baryon_pdgs)))

        proton_mask = (pdg == 2212)
        genie_n_protons = np.sum(proton_mask)

        neutron_mask = (pdg == 2112)
        genie_n_neutrons = np.sum(neutron_mask)

        print(f"Final state particles:")
        print(f"  Muons: {genie_n_muons}")
        print(f"  Photons: {genie_n_photons} ({genie_n_photons_above10MeV} above 10 MeV)")
        print(f"  Mesons: {genie_n_mesons}")
        print(f"  Heavy baryons: {genie_n_heavy_baryons_plus_pi0s}")
        print(f"  Protons: {genie_n_protons}")
        print(f"  Neutrons: {genie_n_neutrons}")





class NUISANCEFile:
    """Iterator over NUISANCE flat tree events."""
    def __init__(self, filepath, treename, relevant_keys, step=1000):
        """
        Parameters
        ----------
        filepath : str
            Path to the ROOT file
        treename : str
            Name of the TTree (usually 'flat_tree')
        step : int
            Number of entries to read per chunk
        """
        self.file = uproot.open(filepath)
        self.tree = self.file[treename]
        self.step = step
        if relevant_keys is None:
            self.keys = self.tree.keys()
        else:
            self.keys = relevant_keys
        self._nentries = self.tree.num_entries

    def __iter__(self):
        """Return an iterator over all events."""
        for arrays in self.tree.iterate(self.keys, step_size=self.step, library="np"):
            n = len(next(iter(arrays.values())))
            for i in range(n):
                data = {k: arrays[k][i] for k in self.keys}
                yield NUISANCEEvent(**data)

    def __getitem__(self, index):
        """Return an event by index.
        
        Parameters
        ----------
        index : int
            Event index (0-based). Supports negative indexing.
            
        Returns
        -------
        NUISANCEEvent
            Event at the specified index.
        """
        # Handle negative indices
        if index < 0:
            index = self._nentries + index
        
        # Check bounds
        if index < 0 or index >= self._nentries:
            raise IndexError(f"Index {index} out of range for tree with {self._nentries} entries")
        
        # Read single entry
        arrays = self.tree.arrays(self.keys, entry_start=index, entry_stop=index+1, library="np")
        
        # Create event from the first (and only) entry
        data = {k: arrays[k][0] for k in self.keys}
        return NUISANCEEvent(**data)

    def __len__(self):
        return self._nentries

    def get_tree(self):
        return self.tree

    def close(self):
        self.file.close()


# Sample codes mapping
SampleCodes = {
    -1: "not selected",
    # "classic" FHC samples
    1: "numu CC 0pi in FHC",
    2: "numu CC 1pi in FHC",
    3: "numu CCother in FHC",
    # more detailed T2K samples (FHC)
    7:  "CC 0pi 0p 0gamma in FHC",
    8:  "CC 0pi Np 0gamma in FHC",
    10: "CC 1pi 0gamma in FHC",
    12: "CC other 0gamma in FHC",
    13: "CC other gamma in FHC",
    # RHC samples
    59: "antinumu CC 0pi in RHC",
    60: "antinumu CC 1pi in RHC",
    61: "antinumu CCother in RHC",
    71: "numu bkg CC 0pi in RHC",
    72: "numu bkg CC 1pi in RHC",
    73: "numu bkg CCother in RHC"
}


