import sys
from pathlib import Path
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

# Change this path to your working directory where BDTReweight is installed:
# sys.path.append('/Users/lorenzo/Minerva/reweighting_workdir')
sys.path.append('/Users/lorenzo/Minerva/fork_reweighting/')

from BDTReweight.nuisance_flat_tree import NuisanceFlatTree


S_RE_GEV = 0.028
DEFAULT_PT_BINS = np.linspace(0.0, 2.5, 60)
DEFAULT_RECOIL_BINS = np.linspace(0.0, 1.4, 60)
DEFAULT_Q2_BINS = np.linspace(0.0, 2.5, 60)
DEFAULT_OMEGA_BINS = np.linspace(0.0, 1.5, 60)
DEFAULT_LEVELS = [0.05, 0.2, 0.5, 0.8]


def _array_from_tree(expr: str, tree: NuisanceFlatTree, mask=None) -> np.ndarray:
	arr = tree.get_event_variable(expr, mask=mask)
	arr = ak.fill_none(arr, np.nan)
	return ak.to_numpy(arr)


def _fetch_weights(tree: NuisanceFlatTree, weight_branch: str, mask=None) -> np.ndarray:
	if weight_branch.lower() == "none":
		return np.ones(len(tree.get_event_variable("Mode", mask=mask)), dtype=float)
	try:
		w = _array_from_tree(weight_branch, tree, mask=mask)
	except Exception:
		w = np.ones(len(tree.get_event_variable("Mode", mask=mask)), dtype=float)
	w = np.nan_to_num(w, nan=0.0)
	return w


def _compute_kinematics(tree: NuisanceFlatTree, mask=None):
	mu_px = _array_from_tree("leading_muon_px", tree, mask=mask)
	mu_py = _array_from_tree("leading_muon_py", tree, mask=mask)
	mu_pz = _array_from_tree("leading_muon_pz", tree, mask=mask)
	mu_e = _array_from_tree("leading_muon_E", tree, mask=mask)

	pT = np.sqrt(np.nan_to_num(mu_px, nan=0.0) ** 2 + np.nan_to_num(mu_py, nan=0.0) ** 2)
	recoil = _array_from_tree("total_proton_KE", tree, mask=mask)
	recoil = np.nan_to_num(recoil, nan=0.0)
	omega = recoil + S_RE_GEV
	q2 = pT * pT + (mu_e - mu_pz + omega) - omega * omega
	omega_true = _array_from_tree("q0", tree, mask=mask)
	q2_true = _array_from_tree("Q2", tree, mask=mask)

	return {
		"pT": pT,
		"recoil": recoil,
		"omega": omega,
		"q2": q2,
		"omega_true": omega_true,
		"q2_true": q2_true,
	}


def _hist2d_density(x, y, x_edges, y_edges, weights, normalize=True):
	finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
	if not np.any(finite):
		return None
	hist, _, _ = np.histogram2d(x[finite], y[finite], bins=[x_edges, y_edges], weights=weights[finite])
	if normalize:
		areas = np.outer(np.diff(x_edges), np.diff(y_edges))
		with np.errstate(invalid="ignore", divide="ignore"):
			hist = hist / areas
		total = np.sum(hist)
		if total > 0:
			hist = hist / total
	return hist


def _plot_contours(datasets, x_key, y_key, x_edges, y_edges, xlabel, ylabel, title, outpath, level_fracs):
	fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
	x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
	y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

	handles = []
	colors = plt.cm.tab10.colors
	for idx, data in enumerate(datasets):
		hist = data["histograms"].get((x_key, y_key))
		if hist is None:
			continue
		max_val = np.nanmax(hist)
		if not np.isfinite(max_val) or max_val <= 0:
			continue
		levels = [max_val * frac for frac in level_fracs]
		cs = ax.contour(x_centers, y_centers, hist.T, levels=levels, colors=[colors[idx % len(colors)]], linewidths=1.5)
		handles.append(plt.Line2D([], [], color=colors[idx % len(colors)], label=data["label"], linewidth=1.5))

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_xlim(x_edges[0], x_edges[-1])
	ax.set_ylim(y_edges[0], y_edges[-1])
	ax.grid(True, alpha=0.3)
	if handles:
		ax.legend(handles=handles, loc="best", title="Model")
	ax.set_title(title)
	outpath.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(outpath)
	plt.close(fig)


def _build_histograms(kinematics, weights, bins):
	histograms = {}
	histograms[("pT", "q2")] = _hist2d_density(kinematics["pT"], kinematics["q2"], bins["pT"], bins["q2"], weights)
	histograms[("pT", "omega")] = _hist2d_density(kinematics["pT"], kinematics["omega"], bins["pT"], bins["omega"], weights)
	histograms[("recoil", "q2")] = _hist2d_density(kinematics["recoil"], kinematics["q2"], bins["recoil"], bins["q2"], weights)
	histograms[("recoil", "omega")] = _hist2d_density(kinematics["recoil"], kinematics["omega"], bins["recoil"], bins["omega"], weights)
	histograms[("pT","q2_true")] = _hist2d_density(kinematics["pT"], kinematics["q2_true"], bins["pT"], bins["q2_true"], weights)
	histograms[("pT","omega_true")] = _hist2d_density(kinematics["pT"], kinematics["omega_true"], bins["pT"], bins["omega_true"], weights)
	histograms[("recoil","q2_true")] = _hist2d_density(kinematics["recoil"], kinematics["q2_true"], bins["recoil"], bins["q2_true"], weights)
	histograms[("recoil","omega_true")] = _hist2d_density(kinematics["recoil"], kinematics["omega_true"], bins["recoil"], bins["omega_true"], weights)
	return histograms


def _parse_bins(arg_val: str, default_edges: np.ndarray) -> np.ndarray:
	if arg_val is None:
		return default_edges
	if ":" in arg_val:
		start, stop, n = arg_val.split(":")
		return np.linspace(float(start), float(stop), int(n))
	edges = [float(x) for x in arg_val.split(",")]
	return np.asarray(edges, dtype=float)


def main():
	parser = argparse.ArgumentParser(description="Overlay momentum-transfer contours from multiple NUISANCE files.")
	parser.add_argument("--inputs", nargs="+", required=True, help="List of NUISANCE flat tree ROOT files.")
	parser.add_argument("--labels", nargs="+", help="Optional labels for each input (same order).")
	parser.add_argument("--outdir", default="momentum_transfer_plots", help="Directory to store output plots.")
	parser.add_argument("--pt-bins", dest="pt_bins", default=None, help="Bin edges for pT (start:stop:n or comma list).")
	parser.add_argument("--recoil-bins", dest="recoil_bins", default=None, help="Bin edges for recoil (start:stop:n or comma list).")
	parser.add_argument("--q2-bins", dest="q2_bins", default=None, help="Bin edges for Q2 (start:stop:n or comma list).")
	parser.add_argument("--omega-bins", dest="omega_bins", default=None, help="Bin edges for omega (start:stop:n or comma list).")
	parser.add_argument("--levels", nargs="+", type=float, default=DEFAULT_LEVELS, help="Contour levels as fractions of the histogram maximum.")
	parser.add_argument("--weight-branch", default="Weight", help="Branch name for event weights (use 'none' for unweighted).")
	parser.add_argument("--ccqelike", action="store_true", help="Apply CCQELike mask.")

	args = parser.parse_args()

	inputs = args.inputs
	if args.labels and len(args.labels) != len(inputs):
		raise ValueError("--labels must match the number of --inputs")
	labels = args.labels if args.labels else [Path(p).stem for p in inputs]

	bins = {
		"pT": _parse_bins(args.pt_bins, DEFAULT_PT_BINS),
		"recoil": _parse_bins(args.recoil_bins, DEFAULT_RECOIL_BINS),
		"q2": _parse_bins(args.q2_bins, DEFAULT_Q2_BINS),
		"omega": _parse_bins(args.omega_bins, DEFAULT_OMEGA_BINS),
	}
	# True kinematics use the same binning as reco variables unless explicitly provided later.
	bins["q2_true"] = bins["q2"]
	bins["omega_true"] = bins["omega"]

	datasets = []
	for path, label in zip(inputs, labels):
		tree = NuisanceFlatTree(path)
		mask = tree.get_mask_flagCCQELike() if args.ccqelike else None
		kin = _compute_kinematics(tree, mask=mask)
		# print(f"Kinematics dictionary keys for {label}: {list(kin.keys())}")
		weights = _fetch_weights(tree, args.weight_branch, mask=mask)
		histograms = _build_histograms(kin, weights, bins)
		datasets.append({"label": label, "histograms": histograms})

	outdir = Path(args.outdir)
	_plot_contours(
		datasets,
		"pT",
		"q2",
		bins["pT"],
		bins["q2"],
		xlabel=r"$p_T$ (GeV)",
		ylabel=r"$Q^2_{reco}$ (GeV$^2$)",
		title=r"$p_T$ vs $Q^2_{reco}$",
		outpath=outdir / "pT_vs_Q2_contours.png",
		level_fracs=args.levels,
	)
	_plot_contours(
		datasets,
		"pT",
		"omega",
		bins["pT"],
		bins["omega"],
		xlabel=r"$p_T$ (GeV)",
		ylabel=r"$\omega_{reco}$ (GeV)",
		title=r"$p_T$ vs $\omega_{reco}$",
		outpath=outdir / "pT_vs_omega_contours.png",
		level_fracs=args.levels,
	)
	_plot_contours(
		datasets,
		"recoil",
		"q2",
		bins["recoil"],
		bins["q2"],
		xlabel="Recoil (GeV)",
		ylabel=r"$Q^2_{reco}$ (GeV$^2$)",
		title=r"Recoil vs $Q^2_{reco}$",
		outpath=outdir / "recoil_vs_Q2_contours.png",
		level_fracs=args.levels,
	)
	_plot_contours(
		datasets,
		"recoil",
		"omega",
		bins["recoil"],
		bins["omega"],
		xlabel="Recoil (GeV)",
		ylabel=r"$\omega_{reco}$ (GeV)",
		title=r"Recoil vs $\omega_{reco}$",
		outpath=outdir / "recoil_vs_omega_contours.png",
		level_fracs=args.levels,
	)
	_plot_contours(
		datasets,
		"pT",
		"q2_true",
		bins["pT"],
		bins["q2_true"],
		xlabel=r"$p_T$ (GeV)",
		ylabel=r"$Q^2_{true}$ (GeV$^2$)",
		title=r"$p_T$ vs $Q^2_{true}$",
		outpath=outdir / "pT_vs_Q2_true_contours.png",
		level_fracs=args.levels,
	)
	_plot_contours(
		datasets,
		"pT",
		"omega_true",
		bins["pT"],
		bins["omega_true"],
		xlabel=r"$p_T$ (GeV)",
		ylabel=r"$\omega_{true}$ (GeV)",
		title=r"$p_T$ vs $\omega_{true}$",
		outpath=outdir / "pT_vs_omega_true_contours.png",
		level_fracs=args.levels,
	)
	_plot_contours(
		datasets,
		"recoil",
		"q2_true",
		bins["recoil"],
		bins["q2_true"],
		xlabel="Recoil (GeV)",
		ylabel=r"$Q^2_{true}$ (GeV$^2$)",
		title=r"Recoil vs $Q^2_{true}$",
		outpath=outdir / "recoil_vs_Q2_true_contours.png",
		level_fracs=args.levels,
	)
	_plot_contours(
		datasets,
		"recoil",
		"omega_true",
		bins["recoil"],
		bins["omega_true"],
		xlabel="Recoil (GeV)",
		ylabel=r"$\omega_{true}$ (GeV)",
		title=r"Recoil vs $\omega_{true}$",
		outpath=outdir / "recoil_vs_omega_true_contours.png",
		level_fracs=args.levels,
	)


if __name__ == "__main__":
	main()
