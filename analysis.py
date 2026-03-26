import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
import awkward as ak
import matplotlib.pyplot as plt
from hep_ml.metrics_utils import ks_2samp_weighted
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap
from .utilities import normalize_vectors, MNEff_evaluate
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

    # guard against zero transverse momentum to avoid NaNs
    norms = np.linalg.norm(transverse_plane_y, axis=1, keepdims=True)
    zero_pt = norms[:, 0] == 0
    safe_norms = np.where(norms == 0, 1.0, norms)
    transverse_plane_y = transverse_plane_y / safe_norms
    if np.any(zero_pt):
        transverse_plane_y[zero_pt] = np.array([0.0, -1.0])

    # x-vector is simply y-vector rotated clock wise by 90 degrees
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

def create_dataframe_from_nuisance(tree : NuisanceFlatTree, variable_exprs : list = [], mask : ArrayLike = None) -> pd.DataFrame:
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

def calculate_weighted_diff_histogram_and_stat_errors(var : ArrayLike, weights : ArrayLike, scale_factor : float, bins : ArrayLike) -> tuple:
    """
    Calculate the weighted differential counts with respect to 
    variable var, dcounts / dvar, and statistical errors,
    for given bins, weights, and scale factor.
    When var is array of particle physical quantity x, and
    scale_factor is 'fScaleFactor' from NUISANCE flat tree, this
    function returns differential cross section dsigma / dx.

    Parameters
    ----------
    var : ArrayLike
        Variables to be counted to make differential histograms. 

    weights : ArrayLike
        Weight for each event.

    scale_factor : float
        A factor to scale up or down the histogram and error bars.

    bins : ArrayLike
        Bin edges for histogram.

    Returns
    ----------
    tuple
        Pair of differential histogram and error bars.
    """
    bin_widths = np.diff(bins)
    counts, _ = np.histogram(var, bins=bins, weights=weights)
    diff_counts = scale_factor * counts / bin_widths
    sum_w2, _ = np.histogram(var, bins=bins, weights=weights**2)
    errors = np.sqrt(sum_w2) * scale_factor / bin_widths
    return diff_counts, errors

def draw_source_target_distributions_and_ratio(source : pd.DataFrame, target : pd.DataFrame, variables : list = [], bottom_adjust : float = 0.1,
        label_subplot_abc : bool = True, legends : list = ['', '', ''], KS_test : bool = True, source_weights : ArrayLike = None,
        new_source_weights : ArrayLike = None, target_weights : ArrayLike = None, scale_source : float = 1.0, scale_target : float = 1.0,
    xlabels : list = None, ylabels : list = None, quantile_range : tuple = [0.005, 0.995], shape_only = False,
    variable_bins : dict = None) -> None:
    """
    Draw distributions of variables of source, source reweighted, and
    target sample in grids of subplots. 

    Parameters
    ----------
    source : pd.DataFrame
        The dataframe containing physical quantities of source sample.
    target : pd.DataFrame
        The dataframe containing physical quantities of source sample.
    variables : list
        List of physical quantities to be plotted, such as
        'leading_proton_px', 'subleading_proton_KE', 'weight', etc.
    bottom_adjust : float
        Value passed to plt.subplot_adjust() to adjust figure bottowm.
        Default: 0.1
    label_subplot_abc : bool
        If True, label subplots with a., b., c., ... on top right.
        Default: True
    legends : list, optional
        List of string of legends corresponding to source sample,
        source sample reweighted, and target sample.
        Default: ['', '', '']
    KS_test : bool, optional
        If True, a KS test score between source (or source reweighted)
        and target distributions is printed on top right of subplot.
        Default: True 
    source_weights : ArrayLike
        Array of old weights for source sample events.
    new_source_weights : ArrayLike
        Array of new weights for source sample events.
    target_weights : ArrayLike
        Array of weights for target sample events.
    scale_source : float, optional
        Scale factor for source sample.
    scale_target : float, optional
        Scale factor for target sample.
    xlabels : list, optional
        strings of x-axis labels.
    ylabels : list, optional
        strings of y-axis labels.
    quntile_range : tuple, optional
        float values (0.0 ~ 1.0) to specify the quantiles of data
        to be plotted. Use this to constrain plot range for better
        visualization.
        Default: [0.005, 0.995]
    shape_only : bool, optional
        If True, normalize histograms to unit area to compare shapes only.

    Returns
    ----------
    None
    """

    # create grids of subplots
    n_plots = len(variables)
    has_weight = 'weight' in variables

    # Use adaptive grid sizing to avoid large unused areas in the canvas.
    # Each non-weight variable uses 1 grid unit; 'weight' uses 2 units.
    plot_units = [2 if v == 'weight' else 1 for v in variables]
    total_units = sum(plot_units)
    max_unit = max(plot_units) if len(plot_units) > 0 else 1

    if has_weight:
        col_candidates = [c for c in (2, 3, 4) if c >= max_unit]
    else:
        col_candidates = [1, 2, 3]

    best_cols = None
    best_rows = None
    best_empty = None
    for c in col_candidates:
        rows = int(np.ceil(total_units / c))
        empty = rows * c - total_units
        if best_empty is None or empty < best_empty or (empty == best_empty and c < best_cols):
            best_cols = c
            best_rows = rows
            best_empty = empty

    n_cols = best_cols
    n_rows = max(best_rows, 1)

    if n_plots == 1 and not has_weight:
        # Single-variable plots should be compact and readable (4:3)
        # without introducing unused outer-grid rows.
        figwidth_in = 8
        figheight_in = 6
    else:
        figwidth_in = 5 * n_cols
        figheight_in = 3 * n_rows
    
    fig = plt.figure(figsize=[figwidth_in, figheight_in], dpi=200)
    alpha = 0.5
    outer_grid = GridSpec(n_rows, n_cols, figure=fig, wspace=0.3, hspace=0.32)
    handles, labels = [], []
    
    # Track grid position accounting for weight taking 2 columns
    plot_row, plot_col = 0, 0
    
    # loop through variables and plot
    for idx, variable in enumerate(variables):
        # For weight, allocate 2 columns; for others, allocate 1
        if variable == 'weight':
            col_span = 2
        else:
            col_span = 1
        
        # Check if we need to wrap to next row
        if plot_col + col_span > n_cols:
            plot_row += 1
            plot_col = 0
        
        row, col = plot_row, plot_col
        plot_col += col_span

        if variable != 'weight':
            # plot histogram and ratio of source / target
            if col_span == 1:
                inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[row, col], height_ratios=[3, 1], hspace=0.0)
            else:
                inner_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[row, col:col+col_span], height_ratios=[3, 1], hspace=0.0)
            ax_main = fig.add_subplot(inner_grid[0])
            ax_ratio = fig.add_subplot(inner_grid[1], sharex=ax_main)
        else:
            # plot log scale histogram for source sample new weights; don't plot ratio
            inner_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[row, col:col+col_span], wspace=0.35)
            ax_main = fig.add_subplot(inner_grid[0])
            ax_log = fig.add_subplot(inner_grid[1])
            ax_ratio = None

        if label_subplot_abc:
            ax_main.text(1.00, 1.02, f'{chr(ord("a")+idx)}.', transform=ax_main.transAxes, ha='right', va='bottom')

        if variable == 'weight':
            # plot source sample new weights (left) and log(weights) (right) side-by-side
            
            # Left: normalized weights histogram (log scale)
            ax_main.hist(new_source_weights * len(new_source_weights)/np.sum(new_source_weights),
                         log=True, bins=30, alpha=alpha, color='goldenrod')
            ax_main.tick_params(which='both', direction='in')
            ax_main.set_xlim(0, None)
            ax_main.set_ylim(0, None)
            ax_main.set_xlabel('new weights')
            ax_main.set_ylabel('counts (log scale)')
            
            # Right: log(weights) histogram (log scale)
            log_weights = np.log(new_source_weights)
            ax_log.hist(log_weights, bins=200, alpha=alpha, color='skyblue')
            ax_log.tick_params(which='both', direction='in')
            ax_log.set_xlim(None, None)
            ax_log.set_ylim(0, None)
            ax_log.set_xlabel('log(weights)')
            ax_log.set_ylabel('counts')
            
            continue

        # Plot the selected quantile rage of data to depict majority of data
        if variable in ['total_proton_KE', 'dpt', 'dphit', 'dalphat', 'leading_proton_KE', 'leading_neutron_KE']:
            x_min = 0.0
        else:
            x_min = min(np.quantile(source[variable], quantile_range[0]),np.quantile(target[variable], quantile_range[0]))
        x_max = min(np.quantile(source[variable], quantile_range[1]),np.quantile(target[variable], quantile_range[1]))

        # plot histogram with either user-provided bins or evenly spaced bins
        if variable_bins is not None and variable in variable_bins:
            bins = np.asarray(variable_bins[variable], dtype=float)
        else:
            bins = np.linspace(x_min, x_max, 30)
        bin_widths = np.diff(bins)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        # a helper function to plot, also returns differential counts and statistical errors
        def hist_plot(data, weights, scale, color, label='', offset=0, ax=ax_main):
            diff_counts, errors = calculate_weighted_diff_histogram_and_stat_errors(data, weights=weights, scale_factor=scale, bins=bins)

            if shape_only:
                area = np.sum(diff_counts * bin_widths)
                if area > 0:
                    diff_counts = diff_counts / area
                    errors = errors / area

            ax.step(bins, np.append(diff_counts, diff_counts[-1]), where='post', label=label, color=color, alpha=alpha)
            ax.errorbar(bin_centers + offset * bin_widths, diff_counts, yerr=errors,
                        fmt=".", color=color, capsize=1.5, markersize=2, alpha=alpha)
            return diff_counts, errors

        # plot source before reweight
        h1, e1 = hist_plot(source[variable], source_weights, scale_source, 'green', label=legends[0], offset=-0.3)
        # plot source after reweight
        h2, e2 = hist_plot(source[variable], new_source_weights, scale_source, 'blue', label=legends[1], offset=0.3)
        # plot target
        h3, e3 = hist_plot(target[variable], target_weights, scale_target, 'red', label=legends[2], offset=0)


        if ylabels is not None:
            ax_main.set_ylabel(ylabels[idx])
        else:
            ax_main.set_ylabel('diff counts')
        ax_main.set_xlim(bins[0], bins[-1])
        ax_main.set_ylim(0, None)
        ax_main.tick_params(which='both', direction='in', top=True, right=True)
        ax_main.minorticks_on()
        
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        ax_main.yaxis.set_major_formatter(fmt)
        plt.setp(ax_main.get_xticklabels(), visible=False)


        # ratio plot of source / target
        if variable != 'weight' and ax_ratio:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.true_divide(h1, h3)
                error_ratio = ratio * np.sqrt((e1 / h1)**2 + (e3 / h3)**2)
                # avoid divide by zeros
                valid = (h1 > 0) & (h3 > 0) & np.isfinite(ratio) & np.isfinite(error_ratio) & (error_ratio >= 0)
            ax_ratio.errorbar(bin_centers[valid], ratio[valid], yerr=error_ratio[valid], label='ratio $s / t$',
                              fmt='.', color='orange', markersize=3,capsize=2,alpha=alpha)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.true_divide(h2, h3)
                error_ratio = ratio * np.sqrt((e2 / h2)**2 + (e3 / h3)**2)
                # Mask valid entries only
                valid = (h2 > 0) & (h3 > 0) & np.isfinite(ratio) & np.isfinite(error_ratio) & (error_ratio >= 0)

            ax_ratio.errorbar(bin_centers[valid], ratio[valid], yerr=error_ratio[valid], label='ratio $s\' / t$',
                              fmt='.', color='purple', markersize=3,capsize=2,alpha=alpha)

            ax_ratio.axhline(1, color='gray', linestyle='-',alpha=alpha)
            ax_ratio.set_ylabel('ratio', fontsize=8)
            ax_ratio.set_yticks([0,1,2])

            ax_ratio.set_ylim(0,2)
            ax_ratio.yaxis.tick_right()
            ax_ratio.yaxis.set_label_position("right")
            ax_ratio.tick_params(which='both', direction='in')

            if xlabels is not None:
                ax_ratio.set_xlabel(xlabels[idx])
            else:
                ax_ratio.set_xlabel(variable)

        if idx == 0:
            h, l = ax_main.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

            if ax_ratio is not None:
                h, l = ax_ratio.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
        
        if KS_test:
            ks_score1 = ks_2samp_weighted(source[variable], target[variable], weights1=source_weights, weights2=target_weights)
            ks_score2 = ks_2samp_weighted(source[variable], target[variable], weights1=new_source_weights, weights2=target_weights)
            KS_text = '$D_{\\text{KS}}$'+f'\nbefore: {ks_score1:.3f}\nafter: {ks_score2:.3f}'
            # KS_line, = ax_main.plot([0], [0],color='white',alpha=0.0, label=KS_text)
            # explicitly print handle/label on subplot ax_main
            ax_main.text(0.98, 0.95, KS_text, transform=ax_main.transAxes,ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.0))



    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
    fig.subplots_adjust(bottom=bottom_adjust)
    # plt.show()
    return fig

def draw_2Dxsec_and_efficiency(df_genie2=[],df_genie3=[],xybins=(np.linspace(0,0.6,20),np.linspace(0,2,20)),fScale_genie2=0,fScale_genie3=0,
               Xsec_columns=('dpt','pT_muon'), xylabels=('$\delta p_T \ (\\text{GeV}/c)$','$p^\mu_T \ (\\text{GeV}/c)$')):
    M2, N2, eff2, xedges, yedges, N2err, M2err, R2err = MNEff_evaluate(df=df_genie2,xybins=xybins,reweight=False,Xsec_columns=Xsec_columns)
    M2_rwt, N2_rwt, eff2_rwt, _, _, N2rwt_err, M2rwt_err, R2rwt_err = MNEff_evaluate(df=df_genie2, reweight=True,xybins=xybins,Xsec_columns=Xsec_columns)
    M3, N3, eff3, _, _, N3err, M3err, R3err = MNEff_evaluate(df=df_genie3,xybins=xybins,reweight=False,Xsec_columns=Xsec_columns)
    datas = [N2, M2, eff2, N3, M3, eff3, N2_rwt, M2_rwt, eff2_rwt]
    titles = ['$N_{v2}$',
              '$M_{v2}$',
              '$\phi_{v2}=M_{v2}/N_{v2}$',
               '$N_{v3}$', 
               '$M_{v3}$', 
               '$\phi_{v3}=M_{v3}/N_{v3}$', 
               '$N_{v2}\'$', 
               '$M_{v2}\'$', 
               '$\phi_{v2}\'=M_{v2}\'/N_{v2}\'$'
        ]
    # titles = ['','','','','','','','','']
    xlabel, ylabel = xylabels
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(11, 9),dpi=300)  # 3x3 grid
    vmax = 1.9e-38
    for i, (ax, data, title) in enumerate(zip(axs.flat, datas, titles)):
        X, Y = np.meshgrid(xedges, yedges)
        area=(xedges[1]-xedges[0])*(yedges[1]-yedges[0])
        # cmap = LinearSegmentedColormap.from_list("black_white_red", ['black','white','red'], N=256)
        cmap = 'viridis'
        if i in [0,1]:
            mesh = ax.pcolormesh(X, Y, data.T*fScale_genie2/area, shading='auto', cmap=cmap, vmin=0.0,vmax=vmax)

        elif i in [3,4]:
            mesh = ax.pcolormesh(X, Y, data.T*fScale_genie3/area, shading='auto', cmap=cmap,vmin=0.0,vmax=vmax)
        elif i in [6,7]:
            # mesh = ax.pcolormesh(X, Y, data.T*(len(df_genie3)*fScale_genie3/len(df_genie2))/area, shading='auto', cmap='viridis')  # Transpose counts
            mesh = ax.pcolormesh(X, Y, data.T/area, shading='auto', cmap=cmap,vmin=0.0,vmax=vmax)
        else:
            mesh = ax.pcolormesh(X, Y, data.T, shading='auto', cmap=cmap,vmin=0.0,vmax=1.0) # efficiency plot

        cbar = plt.colorbar(mesh, ax=ax)
        if i in [0,1,3,4,6,7]:
            cbar.set_label('$\\frac{d^2\\sigma}{d\delta p_T d p^T_{\mu}} \ \left(\\frac{\\text{cm}^2}{(\\text{GeV}/c)^2}\\right)$')
        else:
            cbar.set_label('efficiency')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{chr(ord("a")+i)}. '+title)
        ax.set_yticks(yedges)
        ax.set_yticklabels([f"{edge:.2f}" for edge in yedges])

        # plt.show()
    plt.tight_layout()
    # plt.show()
    
    
    

        
    datas = [eff2/eff3,eff2_rwt/eff3]
    titles = ['Ratio $\phi_{v2}/\phi_{v3}$','Ratio $\phi_{v2}\'/\phi_{v3}$']

    # vmin,vmax=(0.0, np.max([datas[0],datas[1]]))
    # vmin,vmax=(0.0, 3.0)
    vmin,vmax=(0.0, 2.0)


    print('vmin, vmax:',vmin,vmax)

        

    dpt_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
    # print('dpt bincenters:',dpt_bin_centers)
    eff23_ratio = eff2/eff3
    ratio23_err = eff23_ratio * np.sqrt((R2err/eff2)**2+(R3err/eff3)**2)

    
    # eff2_rwt/eff3 histogram slices
    eff2rwt3_ratio = eff2_rwt/eff3
    ratio2rwt3_err = eff2rwt3_ratio * np.sqrt((R2rwt_err/eff2_rwt)**2+(R3err/eff3)**2)



    # new: 2025 July 25 ______________________________________________________________________________________________________
    fig = plt.figure(figsize=(10, 8.5), dpi=300)
    outer = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.07)

    # Top-left
    ax00 = fig.add_subplot(outer[0, 0])

    data=eff2/eff3
    X, Y = np.meshgrid(xedges, yedges)
    cmap = LinearSegmentedColormap.from_list("black_white_red", ['black','white','red'], N=256)
    # cmap = 'viridis'

    mesh = ax00.pcolormesh(X, Y, data.T, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)  # Transpose counts
    cbar = plt.colorbar(mesh, ax=ax00)
    cbar.set_ticks(np.linspace(vmin, vmax, 11))  # optional: set colorbar ticks at 0.0, 0.1, ..., 1.0
    # cbar.set_label('ratio')
    ax00.set_xlabel(xlabel)
    ax00.set_ylabel(ylabel)
    ax00.set_title(
        'a.',
        # 'a. Ratio $\phi_{v2}/\phi_{v3}$',
        fontsize=11)
    ax00.set_yticks(yedges)
    ax00.set_yticklabels([f"{edge:.2f}" for edge in yedges])

    # Top-right
    ax01 = fig.add_subplot(outer[0, 1])
    data=eff2_rwt/eff3
    X, Y = np.meshgrid(xedges, yedges)
    mesh = ax01.pcolormesh(X, Y, data.T, vmin=vmin, vmax=vmax, shading='auto', cmap=cmap)  # Transpose counts
    cbar = plt.colorbar(mesh, ax=ax01)
    cbar.set_ticks(np.linspace(vmin, vmax, 11))  # optional: set colorbar ticks at 0.0, 0.1, ..., 1.0
    # cbar.set_label('ratio')
    ax01.set_xlabel(xlabel)
    ax01.set_ylabel(ylabel)
    ax01.set_title(
        'b.',
        # 'b. Ratio $\phi_{v2}\'/\phi_{v3}$',
        fontsize=11)
    ax01.set_yticks(yedges)
    ax01.set_yticklabels([f"{edge:.2f}" for edge in yedges])



    half = eff2rwt3_ratio.shape[1]//2

    # Bottom-left: nested grid 
    inner = GridSpecFromSubplotSpec(eff2rwt3_ratio.shape[1]-half, 1, subplot_spec=outer[1, 0], hspace=0.05)
    for index in range(half,eff2rwt3_ratio.shape[1]):
        ax = fig.add_subplot(inner[index-half])
        i = eff2rwt3_ratio.shape[1]-1-index

        # draw a line at y=1
        ax.axhline(y=1, color='black', linestyle='--',alpha=0.5)
        # draw a invisible line to label "c." for paper...
        ax.axhline(y=0, color='black', linestyle='-',alpha=0.0,label='c.')


        # eff2_rwt/eff3
        slice_counts = eff2rwt3_ratio.T[i, :]
        errors = ratio2rwt3_err.T[i,:]

        slice_counts[np.isnan(slice_counts)] = 0
        ax.step(xedges, np.append(slice_counts, slice_counts[-1]), where='post',label='$\phi_{v2}\'/\phi_{v3}$',color='purple',alpha=0.7)
        ax.errorbar(dpt_bin_centers - 0.1*np.diff(xedges), slice_counts, yerr=errors, capsize = 2,fmt='none',color='purple',markersize=2,alpha=0.7)

        yge2 = np.any((slice_counts+errors) >= 2)

        # eff2/eff3
        slice_counts = eff23_ratio.T[i, :]
        errors = ratio23_err.T[i,:]
        #fill nan with zero:
        slice_counts[np.isnan(slice_counts)] = 0
        ax.step(xedges, np.append(slice_counts, slice_counts[-1]), where='post',label='$\phi_{v2}/\phi_{v3}$',color='orange',alpha=0.7)
        ax.errorbar(dpt_bin_centers, slice_counts, yerr=errors,  markersize=2,capsize = 2,fmt='none',color='orange',alpha=0.7)


        # add a text line at top left
        ax.text(0.95, 0.9, f'{round(yedges[i],2)} $\le ~ p^T_\mu ~ <$ {round(yedges[i+1],2)} (GeV/$c$)', fontsize=8,
                horizontalalignment='right',verticalalignment='top', transform=ax.transAxes)

        ax.set_xlim(0,xedges[-1])
        
        # yge2 = yge2 or np.any((slice_counts+errors) >= 1.3)
        # if yge2:
        #     ax.set_ylim(0.7,None)
        # else:
        #     ax.set_ylim(0.7,1.3)
        points = slice_counts+errors
        points[~np.isfinite(points)] = 1.0
        ymax = np.max(points-1)*1.1

        # yrange = max(np.max(points-1),np.max(1-points))*5
        # ax.set_ylim(1-yrange,1+yrange)
        points = slice_counts-errors
        points[~np.isfinite(points)] = 1.0
        ymin = np.max(1-points)*1.1

        yrange = max(ymin,ymax)
        ax.set_ylim(1-yrange,1+ yrange)

        # ax.legend()
        
        if index < eff2rwt3_ratio.shape[1] - 1:
            ax.set_xticklabels([])

        if index == eff2rwt3_ratio.shape[1]-1:
            ax.set_xlabel(xlabel)

        # ax.set_yticks([0,1])
        # ax.set_yticklabels([0,1])



        if index == half:
            ax.legend(loc='lower center', bbox_to_anchor=(1.1, 1.3), ncol=3, frameon=False,fontsize=11)
        
        bbox = outer[1, 0].get_position(fig)
        ax.set_position([
            bbox.x0,
            ax.get_position().y0,
            bbox.width*0.8,
            ax.get_position().height
        ])

    # Bottom-right: nested grid 
    inner = GridSpecFromSubplotSpec(half, 1, subplot_spec=outer[1, 1], hspace=0.05)
    for index in range(0,half):
        ax = fig.add_subplot(inner[index])
        i = eff2rwt3_ratio.shape[1]-1-index

        # draw a line at y=1
        ax.axhline(y=1, color='black', linestyle='--',alpha=0.5)
        # draw a invisible line to label "c." for paper...
        ax.axhline(y=0, color='black', linestyle='-',alpha=0.0,label='c.')


        # eff2_rwt/eff3
        slice_counts = eff2rwt3_ratio.T[i, :]
        errors = ratio2rwt3_err.T[i,:]

        slice_counts[np.isnan(slice_counts)] = 0
        ax.step(xedges, np.append(slice_counts, slice_counts[-1]), where='post',label='$\phi_{v2}\'/\phi_{v3}$',color='purple',alpha=0.7)
        ax.errorbar(dpt_bin_centers - 0.1*np.diff(xedges), slice_counts, yerr=errors, capsize = 2,fmt='none',color='purple',markersize=2,alpha=0.7)

        yge2 = np.any((slice_counts+errors) >= 2)

        # eff2/eff3
        slice_counts = eff23_ratio.T[i, :]
        errors = ratio23_err.T[i,:]
        #fill nan with zero:
        slice_counts[np.isnan(slice_counts)] = 0
        ax.step(xedges, np.append(slice_counts, slice_counts[-1]), where='post',label='$\phi_{v2}/\phi_{v3}$',color='orange',alpha=0.7)
        ax.errorbar(dpt_bin_centers, slice_counts, yerr=errors,  markersize=2,capsize = 2,fmt='none',color='orange',alpha=0.7)


        # add a text line at top right
        ax.text(0.05, 0.9, f'{round(yedges[i],2)} $\le ~ p^T_\mu ~ <$ {round(yedges[i+1],2)} (GeV/$c$)', fontsize=8,
                horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)

        ax.set_xlim(0,xedges[-1])
        
        # yge2 = yge2 or np.any((slice_counts+errors) >= 1.3)
        # if yge2:
        #     ax.set_ylim(0.7,None)
        # else:
        #     ax.set_ylim(0.7,1.3)
        points = slice_counts+errors
        points[~np.isfinite(points)] = 1.0
        ymax = np.max(points-1)*2

        yrange = max(np.max(points-1),np.max(1-points))*1.1
        # ax.set_ylim(1-yrange,1+yrange)
        points = slice_counts-errors
        points[~np.isfinite(points)] = 1.0
        ymin = np.max(1-points)*1.1

        yrange = max(ymin,ymax)
        ax.set_ylim(1-yrange,1+ yrange)        
        
        
        # ax.legend()
        
        if index < half - 1:
            ax.set_xticklabels([])

        if index == half-1:
            ax.set_xlabel(xlabel)

        # ax.set_yticks([0,1])
        # ax.set_yticklabels([0,1])



        # if index == 0:
        #     ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, 
        #             #   fontsize=20, 
        #               frameon=False)
        
        bbox = outer[1, 1].get_position(fig)
        ax.set_position([
            bbox.x0,
            ax.get_position().y0,
            bbox.width*0.8,
            ax.get_position().height
        ])



    plt.show()
