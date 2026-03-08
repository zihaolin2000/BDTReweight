import argparse
from nuisance_event import NUISANCEEvent, NUISANCEFile
import numpy as np
from tqdm import tqdm
import uproot

def perform_selection(input_file, output_file, sample, max_events=-1):

    treename='FlatTree_VARS'
    f = NUISANCEFile(filepath=input_file, treename=treename, relevant_keys=None)
    num_entries = len(f)

    selection_mask = np.zeros(num_entries, dtype=bool)

    pbar = tqdm(total=100)
    for i, event in enumerate(f):
        # limit number of events for testing
        if i > max_events and max_events > 0:
            break
        # progress bar
        if (i + 1) % (num_entries // 100) == 0:
            pbar.update(1)
        # Perform selection
        if sample == "CCQELike":
            selected = event.MINERvACCQELikeSelection()
        else:
            raise ValueError(f"Unknown sample type: {sample}")

        selection_mask[i] = selected
    flagCCQELike_mask = f.get_mask_flagCCQELike()
    falgCC0piMINERVa_mask = f.get_mask_flagCC0piMINERVA()
    # print out final state particle contents for those events that pass the CCQELike selection but fail the falgCC0piMINERVa_mask or flagCCQELike_mask
    for i in range(num_entries):
        if selection_mask[i] and (not flagCCQELike_mask[i] or not falgCC0piMINERVa_mask[i]):
            event = f.get_event(i)
            print(f"Event {i} passes CCQELike selection but fails flagCCQELike or falgCC0piMINERVa:")
            print(f"  Final state particles: {event.final_state_particles}")
            print(f"  flagCCQELike: {flagCCQELike_mask[i]}, falgCC0piMINERVa: {falgCC0piMINERVa_mask[i]}")
            print(f" My selection: {selection_mask[i]}, flagCCQELike: {flagCCQELike_mask[i]}, falgCC0piMINERVa: {falgCC0piMINERVa_mask[i]}")

    pbar.close()

    # Apply selection mask to all branches
    filtered_branches = {}
    tree = f.get_tree()
    for branch_name in f.keys:
        branch_array = tree[branch_name].array(library="ak")
        filtered_branches[branch_name] = branch_array[selection_mask]

    # Write to new ROOT file
    with uproot.recreate(output_file) as fout:
        fout.mktree(treename, filtered_branches)

    print(f"Selection complete. Output written to {output_file}. Selected {np.sum(selection_mask)} out of {num_entries} events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform selection on input ROOT file.')
    parser.add_argument('--input_file', '-i', type=str, required=True, help='Path to the input NUISANCE flat tree file.')
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Path to the output ROOT file.')
    parser.add_argument('--sample', '-s', type=str, default='CCQELike', help='Sample type (e.g., "CCQELike").')

    args = parser.parse_args()

    perform_selection(args.input_file, args.output_file, args.sample)