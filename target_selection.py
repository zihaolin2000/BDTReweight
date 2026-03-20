import sys
import os
sys.path.append('/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/')

import argparse
from nuisance_event import NUISANCEEvent, NUISANCEFile
import numpy as np
from tqdm import tqdm
import uproot
from BDTReweight.nuisance_flat_tree import NuisanceFlatTree


def perform_selection(input_file, output_file, sample, max_events=-1):


    treename='FlatTree_VARS'
    f = NUISANCEFile(filepath=input_file, treename=treename, relevant_keys=None)
    num_entries = len(f)

    # flat_tree = NuisanceFlatTree(input_file)

    print(f"Performing selection on {max_events} events from file {input_file}. \nSample: {sample}")

    if max_events < 0:
        max_events = num_entries
    selection_mask = np.zeros(num_entries, dtype=bool)

    pbar = tqdm(total=100)
    for i, event in enumerate(f):
        # limit number of events for testing
        if i > max_events and max_events > 0:
            break
        # progress bar
        if (i + 1) % (max_events // 100) == 0:
            pbar.update(1)
        # Perform selection
        if sample == "CCQELike":
            selected = event.MINERvACCQELikeSelection()
        else:
            raise ValueError(f"Unknown sample type: {sample}")

        selection_mask[i] = selected
    # flagCCQELike_mask = flat_tree.get_mask_flagCCQELike()
    # falgCC0piMINERVa_mask = flat_tree.get_mask_flagCC0piMINERVA()
    # print out final state particle contents for those events that pass the CCQELike selection but fail the falgCC0piMINERVa_mask or flagCCQELike_mask
    # for i in range(min(num_entries,10)):
    #     if selection_mask[i] and (not flagCCQELike_mask[i] or not falgCC0piMINERVa_mask[i]):
    #         event = f[i]
    #         print(f"Event {i} passes CCQELike selection but fails flagCCQELike or falgCC0piMINERVa:")
    #         print(f" My selection: {selection_mask[i]}, flagCCQELike: {flagCCQELike_mask[i]}, falgCC0piMINERVa: {falgCC0piMINERVa_mask[i]}")
    #         event.print_final_state_particles()

    pbar.close()


    print(f"Selected {np.sum(selection_mask)} out of {max_events} events for sample {sample}.\nConverting branhces to awkward arrays and applying selection mask.")
    # Apply selection mask to all branches
    filtered_branches = {}
    tree = f.get_tree()
    n_branches = len(f.keys)
    for branch_name in f.keys:
        print(f"  Processing branch {branch_name} ({len(filtered_branches)+1}/{n_branches})...")
        branch_array = tree[branch_name].array(library="ak")
        if branch_name == 'fScaleFactor':
            filtered_branches[branch_name] = branch_array[selection_mask] * num_entries/max_events
            print(f" Multiplying fScaleFactor: {filtered_branches[branch_name][0]}={branch_array[selection_mask][0]} * {num_entries/max_events}")
        else:
            filtered_branches[branch_name] = branch_array[selection_mask]
        # correct fScaleFactor branch: to account for the fact that we only looped through max_events events

    print(f"Filtered branches created. Writing to output file {output_file}...")
    # Write to new ROOT file
    with uproot.recreate(output_file) as fout:
        # fout.mktree(treename, filtered_branches)
        fout[treename] = filtered_branches


    print(f"Selection complete. Output written to {output_file}. Selected {np.sum(selection_mask)} out of {max_events} events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform selection on input ROOT file.')
    parser.add_argument('--input_file', '-i', type=str, required=True, help='Path to the input NUISANCE flat tree file.')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Path to the output ROOT file.')
    parser.add_argument('--sample', '-s', type=str, default='CCQELike', help='Sample type (e.g., "CCQELike").')
    parser.add_argument('--max_events', '-m', type=int, default=-1, help='Maximum number of events to process (default: -1 for all events).')

    args = parser.parse_args()


    perform_selection(args.input_file, args.output_file, args.sample, args.max_events)
