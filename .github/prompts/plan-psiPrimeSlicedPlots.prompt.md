## Plan: Add Psi-Prime Sliced Reweighting Plots

Add 24 new post-reweighting diagnostic plots inside the existing per-process loop in train_by_reaction.py: 13 psi-prime distributions in muon pT slices and 11 psi-prime distributions in recoil slices. Compute psi-prime from final-state kinematics using the provided C++ formula translated to Python, with constants converted from MeV to GeV. Reuse existing plotting style by calling draw_source_target_distributions_and_ratio on sliced DataFrames and writing outputs to the same pics/<target_model_name>/ directory.

**Steps**
1. Define psi-prime calculation helpers in a reusable location (*phase: kinematics helpers*).
2. Add constants for S_RE, k_F, Eshift in GeV using user-confirmed values: 28 MeV, 228 MeV, 20 MeV (*depends on 1*).
3. Implement Python translation of ComputePsiPrime(q0, q3_mag) and GetPsiPrimeREFromFSKinematics(recoil, muon_px, muon_py, muon_pz), including safe handling for invalid sqrt/division regions to avoid crashes on edge events (*depends on 1,2*).
4. In the process loop of train_by_reaction.py, compute per-event muon pT and recoil-in-MeV columns for source and target subsets, and compute psi-prime column for both source and target using recoil in GeV for the formula (*phase: per-process derived variables*).
5. Add fixed bin-edge arrays exactly as specified:
6. Muon pT bins in GeV: [0, 0.075, 0.15, 0.25, 0.325, 0.4, 0.475, 0.55, 0.7, 0.85, 1, 1.25, 1.75, 2.5].
7. Recoil bins in MeV: [0, 20, 40, 80, 120, 160, 240, 320, 400, 600, 800, 1400].
8. Build a small plotting helper in train_by_reaction.py that takes slice masks (for source/target), calls draw_source_target_distributions_and_ratio with variables=['psi_prime'], and saves output with descriptive names that include process, category, slice type, and bin range (*depends on 4,5-7*).
9. Generate 13 pT-sliced plots: each mask is pT in [edge_i, edge_{i+1}) (last bin right-inclusive), with all recoil included (*depends on 8*).
10. Generate 11 recoil-sliced plots: each mask is recoil_mev in [edge_i, edge_{i+1}) (last bin right-inclusive), with all pT included (*depends on 8*).
11. Reuse existing weights exactly:
12. source_weights=source_train_p['init_wgt']
13. target_weights=target_train_p['weight']
14. new_source_weights=all_weights
15. Add guard logging for empty/near-empty slices and skip plotting those slices safely if a source or target slice has zero events (*parallel with 9 and 10*).
16. Keep output path unchanged (pics/<target_model_name>/) and save with filenames like PsiPrime_ptSlice_... and PsiPrime_recoilSlice_... alongside current plots (*parallel with 9 and 10*).

**Relevant files**
- /Users/lorenzo/Minerva/fork_reweighting/BDTReweight/train_by_reaction.py — main integration point inside per-process loop after existing plot calls around [train_by_reaction.py](train_by_reaction.py#L251) and [train_by_reaction.py](train_by_reaction.py#L268).
- /Users/lorenzo/Minerva/fork_reweighting/BDTReweight/analysis.py — reuse [draw_source_target_distributions_and_ratio](analysis.py#L133) without changing its core behavior.
- /Users/lorenzo/Minerva/fork_reweighting/BDTReweight/utilities.py — optional reusable location for psi-prime math helpers if extracted from script-level code.

**Verification**
1. Run the existing training command with representative source/target files and confirm no runtime exceptions during the new slice plotting section.
2. Confirm exactly 24 additional output images per process/category combination when all slices are populated (13 pT slices + 11 recoil slices).
3. Spot-check that recoil slicing uses MeV labels/ranges while psi-prime computation uses recoil in GeV.
4. Verify pT definition is sqrt(px^2 + py^2) in GeV and slice boundaries match the requested edges.
5. Check that each plot compares Source, Target, and Source (Reweighted) and includes ratio panel via draw_source_target_distributions_and_ratio.
6. Inspect logs for skipped empty slices; ensure skips are expected and clearly reported.
7. If available, compare 1-2 bins against a manual calculation to validate psi-prime implementation numerically.

**Decisions**
- Use psi-prime as x variable.
- Use user-provided constants: S_RE=28 MeV, k_F=228 MeV, Eshift=20 MeV; convert to GeV internally.
- Use recoil in MeV only for slicing; recoil in GeV for psi-prime formula.
- Produce 24 plots total: 13 pT-sliced + 11 recoil-sliced.
- Save in existing pics/<target_model_name>/ output directory.

**Further Considerations**
1. Function placement recommendation: implement psi-prime helpers in utilities.py for reuse and cleaner train_by_reaction.py; fallback is local helper functions near the plotting block for minimal diff.
2. Numeric stability recommendation: clamp tiny negative values before sqrt (from floating precision) and return NaN for unphysical tau/denominator domains, then exclude NaN psi-prime entries from histograms.
3. Labeling recommendation: include explicit bin labels in plot suptitle and filename for direct traceability during physics validation.
