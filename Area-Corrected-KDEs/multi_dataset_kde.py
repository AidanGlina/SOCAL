"""
Pooling several datasets for the area-corrected KDE analysis.

Each dataset keeps its own shapes, study area and reference distances. For
every name, points from all datasets are combined, and the expected
(uniform-placement) distribution is the mix of the datasets' references
weighted by that name's share of points in each dataset. This matches each
name to its own mix of datasets, so differences in shapes or expression
levels between datasets don't distort the result, and no points have to be
up- or down-weighted.

Requires area_corrected_kde.py in the same folder. Results feed straight into
its downstream functions (cluster_genes, plot_cluster_profiles,
plot_cluster_heatmap, plot_ridgeline, rho_distance_matrix,
plot_inside_outside). For the gene plot and the inside-vs-outside scatter,
use the _multi versions here.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from area_corrected_kde import (_cluster_color, _cluster_name, _kde_grid, _kde_smoother,
                                _xwindow)


def area_corrected_kdes_multi(datasets, T, n_bins=4096, ref_floor=1e-3, boundary=None):
    """
    Area-corrected KDEs pooled across several datasets, each with its own
    shapes, study area and reference.

    datasets : dict {label: (df, ref_d)}; each df has columns name, distance
               (plus x, y), and ref_d comes from make_reference_distances
               run on THAT dataset's shapes and bounds. Distances must be in
               the same physical units in every dataset.
    T, n_bins, ref_floor, boundary : as in area_corrected_kdes (use
               boundary=0 for unassigned transcripts against cell polygons)

    Pooling: for each name, points from all datasets are combined, and the
    expected (uniform-placement) distribution is the mix of each dataset's
    reference weighted by that name's share of points in that dataset. So a
    dataset with more area at some distance, or more points for that name,
    counts for more there, and differences in shapes or expression levels
    between datasets don't distort the result.

    Returns names, grid, rho, ref, bw, dens, per_dataset
        rho, dens   : as in area_corrected_kdes, pooled over datasets
        ref         : 1D average reference over names, used as the distance
                      weights by cluster_genes / rho_distance_matrix, so
                      those work unchanged
        per_dataset : dict with
                      'rho'         -> {label: rho from that dataset alone}
                      'n'           -> DataFrame of points (d <= T) per name
                                       and dataset
                      'ref_by_name' -> 2D (names x grid) reference each
                                       name's rho was actually divided by
                                       (used by plot_gene_multi)
        Grid points are masked unless every dataset has area there.
    """
    labels_ = list(datasets)
    dfs = {k: datasets[k][0].dropna(subset=['name', 'distance']) for k in labels_}
    refs = {k: np.asarray(datasets[k][1], float) for k in labels_}
    names = pd.Index(pd.unique(np.concatenate(
        [dfs[k]['name'].to_numpy() for k in labels_])))
    m = len(names)

    # Shared bandwidth: median of per-(dataset, name) Silverman bandwidths
    bws = []
    for k in labels_:
        g = dfs[k].groupby('name')['distance']
        iqr = g.quantile(.75) - g.quantile(.25)
        bws.append(0.9 * np.minimum(g.std(), iqr / 1.34) * g.size() ** -0.2)
    bw_all = pd.concat(bws)
    bw = float(bw_all[bw_all > 0].median())

    # Shared grid over all data and references
    lo = min(min(dfs[k]['distance'].min(), refs[k].min()) for k in labels_) - 5 * bw
    hi = max(max(dfs[k]['distance'].max(), refs[k].max()) for k in labels_) + 5 * bw
    lo, dx, n_bins, grid = _kde_grid(lo, hi, n_bins, boundary)
    sigma = bw / dx
    if sigma < 5:
        print(f"Warning: bw/dx = {sigma:.1f}; consider increasing n_bins.")
    keep = grid <= T

    def to_bins(v):
        return np.clip(np.floor((v - lo) / dx).astype(int), 0, n_bins - 1)

    smooth = _kde_smoother(grid, sigma, boundary)

    dens_k, ref_k, n_k = {}, {}, {}
    for k in labels_:
        df_k, r = dfs[k], refs[k]
        codes = names.get_indexer(df_k['name'])
        d = df_k['distance'].to_numpy(float)
        cnt = np.bincount(codes * n_bins + to_bins(d), minlength=m * n_bins)
        dk = smooth(cnt.reshape(m, n_bins), axis=1)[:, keep]
        tot = dk.sum(axis=1, keepdims=True)
        dens_k[k] = dk / np.where(tot > 0, tot, np.nan) / dx
        rk = smooth(np.bincount(to_bins(r), minlength=n_bins))[keep]
        ref_k[k] = rk / (rk.sum() * dx)
        n_k[k] = np.bincount(codes[d <= T], minlength=m).astype(float)

    grid = grid[keep]
    N = pd.DataFrame(n_k, index=names)
    tot = N.sum(axis=1).to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        pi = N.to_numpy() / tot[:, None]                  # name x dataset

    dens = sum(pi[:, [j]] * np.nan_to_num(dens_k[k]) for j, k in enumerate(labels_))
    ref = sum(pi[:, [j]] * ref_k[k][None, :] for j, k in enumerate(labels_))
    dens[tot == 0] = np.nan
    ref[tot == 0] = np.nan

    ok = np.logical_and.reduce([ref_k[k] > ref_floor * ref_k[k].max()
                                for k in labels_])
    rho = np.full_like(dens, np.nan)
    rho[:, ok] = dens[:, ok] / ref[:, ok]

    rho_k = {}
    for k in labels_:
        rk = np.full((m, len(grid)), np.nan)
        rk[:, ok] = dens_k[k][:, ok] / ref_k[k][ok]
        rho_k[k] = rk
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ref_avg = np.nanmean(ref, axis=0)
    return names, grid, rho, ref_avg, bw, dens, {'rho': rho_k, 'n': N,
                                                 'ref_by_name': ref}


def inside_outside_enrichment_multi(datasets, T):
    """
    Per-name enrichment inside shapes vs outside, from raw counts (no KDE
    smoothing, so nothing leaks across the boundary). Only d <= T is used.

    datasets : dict {label: (df, ref_d)}, as for area_corrected_kdes_multi.
    Counts are pooled, and each name's expected inside share combines the
    datasets' inside areas, weighted by that name's points per dataset.
    Output columns match inside_outside_enrichment, so plot_inside_outside
    works unchanged; use plot_inside_vs_outside_multi for the 2D plot.

    Columns returned (index = name):
        n, n_in, n_out : points with d <= T, inside (d < 0), outside
        frac_in        : n_in / n
        area_in        : expected inside share under uniform placement
        rho_in/rho_out : relative density inside / outside (1 = as expected)
        log2_ratio     : log2(rho_out / rho_in); < 0 favours inside,
                         > 0 favours outside
        ci_low/ci_high : approximate 95% interval for log2_ratio
        z, pvalue      : Wald test of log2_ratio = 0
        log2_ratio_<label> : the same ratio from each dataset alone
    Intervals and p-values assume independent points; spatially clustered
    points (e.g. many per cell) make them too optimistic.
    """
    from scipy.stats import norm

    pairs = datasets

    parts, R_in_tot, R_out_tot = [], 0, 0
    for k, (dk, rk) in pairs.items():
        rk = np.asarray(rk, float)
        rk = rk[rk <= T]
        R_in, R_out = (rk < 0).sum(), (rk >= 0).sum()
        if R_in == 0 or R_out == 0:
            raise ValueError(
                f"Reference for dataset {k!r} has {R_in} points inside shapes and {R_out} "
                f"outside (d <= T). Both are needed. Make sure you passed ref_d "
                f"(the raw reference distances), not ref (the smoothed density "
                f"returned by area_corrected_kdes). Otherwise, check the bounds "
                f"and coordinate system used in make_reference_distances, and "
                f"that ref_d was made with the same shapes as df['distance'].")
        R_in_tot += R_in
        R_out_tot += R_out
        sub = dk.loc[dk['distance'] <= T]
        g = (sub['distance'] < 0).groupby(sub['name'])
        parts.append(pd.DataFrame({'n': g.size(), 'n_in': g.sum(),
                                   'a_in': R_in / len(rk), 'ds': k}))

    long = pd.concat(parts).rename_axis('name').reset_index()
    long['n'] = long['n'].astype(float)
    long['n_in'] = long['n_in'].astype(float)
    long['E_in'] = long['n'] * long['a_in']
    agg = long.groupby('name')[['n', 'n_in', 'E_in']].sum()
    n, n_in, E_in = agg['n'], agg['n_in'], agg['E_in']
    n_out, E_out = n - n_in, n - E_in

    out = pd.DataFrame({'n': n, 'n_in': n_in, 'n_out': n_out})
    out['frac_in'] = n_in / n
    out['area_in'] = E_in / n
    out['rho_in'] = n_in / E_in
    out['rho_out'] = n_out / E_out
    # 0.5 continuity correction keeps names with no inside/outside points finite
    lor = np.log((n_out + 0.5) / (n_in + 0.5)) + np.log(E_in / E_out)
    se = np.sqrt(1 / (n_in + 0.5) + 1 / (n_out + 0.5)
                 + 1 / R_in_tot + 1 / R_out_tot)
    out['log2_ratio'] = lor / np.log(2)
    out['ci_low'] = (lor - 1.96 * se) / np.log(2)
    out['ci_high'] = (lor + 1.96 * se) / np.log(2)
    out['z'] = lor / se
    out['pvalue'] = 2 * norm.sf(np.abs(out['z']))

    for k, gk in long.groupby('ds', sort=False):
        gk = gk.set_index('name')
        no, ni = gk['n'] - gk['n_in'], gk['n_in']
        a = gk['a_in']
        out[f'log2_ratio_{k}'] = (np.log((no + 0.5) / (ni + 0.5))
                                  + np.log(a / (1 - a))) / np.log(2)
    return out


def plot_inside_vs_outside_multi(stats, labels=None, genes=None, annotate=10,
                           log=True, floor=0.01, ax=None):
    """
    Relative density inside shapes (y) against outside shapes (x), per name.

    stats    : output of inside_outside_enrichment_multi
    labels   : optional cluster labels (from cluster_genes) to colour points
    genes    : names to always annotate and outline
    annotate : also annotate this many names with the largest |z|
    log      : log2 axes (2x enrichment and 2x depletion equally far from 1)
    floor    : values below this are drawn at the floor on log axes

    Grey bars are approximate 95% intervals. Each name has its own expected
    inside share (its mix of datasets), so unlike the single-dataset plot
    names don't all fall on one line, and no line is drawn.
    """
    from matplotlib.collections import LineCollection

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    st = stats
    a_in = st['area_in'].to_numpy()
    a_out = 1 - a_in
    one_line = np.ptp(a_in) < 1e-9          # single dataset: shared constraint
    clip = (lambda v: np.maximum(v, floor)) if log else (lambda v: v)
    xv, yv = clip(st['rho_out']), clip(st['rho_in'])

    # 95% Wilson interval for the inside share -> segment along the line
    n, p, zc = st['n'], st['frac_in'], 1.96
    c = (p + zc**2 / (2 * n)) / (1 + zc**2 / n)
    h = zc * np.sqrt(p * (1 - p) / n + zc**2 / (4 * n**2)) / (1 + zc**2 / n)
    lo, hi = (c - h).clip(0, 1), (c + h).clip(0, 1)
    segs = np.stack([np.column_stack([clip((1 - lo) / a_out), clip(lo / a_in)]),
                     np.column_stack([clip((1 - hi) / a_out), clip(hi / a_in)])],
                    axis=1)
    ax.add_collection(LineCollection(segs, colors='grey', lw=1.5, alpha=0.4,
                                     zorder=1))

    # Constraint line: a_in * rho_in + a_out * rho_out = 1. With several
    # datasets each name has its own expected inside share, so no single line.
    if one_line:
        t = np.linspace(1e-4, 1 - 1e-4, 400)
        ax.plot(clip((1 - t) / a_out[0]), clip(t / a_in[0]), color='grey',
                ls='--', lw=0.8, zorder=0)

    if labels is not None:
        lab = labels.reindex(st.index)
        for k in sorted(lab.dropna().unique()):
            m = (lab == k).to_numpy()
            ax.scatter(xv[m], yv[m], s=18, color=_cluster_color(k),
                       label=_cluster_name(k), zorder=2)
        m = lab.isna().to_numpy()
        if m.any():
            ax.scatter(xv[m], yv[m], s=12, color='lightgrey',
                       label='Not clustered', zorder=2)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(xv, yv, s=18, color='C0', zorder=2)

    to_label = set(st['z'].abs().nlargest(annotate).index) if annotate else set()
    if genes is not None:
        genes = [g for g in genes if g in st.index]
        ax.scatter(xv[genes], yv[genes], s=60, facecolors='none',
                   edgecolors='k', lw=1.2, zorder=3)
        to_label |= set(genes)
    for g in to_label:
        ax.annotate(str(g), (xv[g], yv[g]), xytext=(4, 3),
                    textcoords='offset points', fontsize=8, zorder=4)

    ax.axhline(1, color='grey', ls=':', lw=1.2, zorder=0)
    ax.axvline(1, color='grey', ls=':', lw=1.2, zorder=0)
    pad = 1.3
    xs = np.concatenate([xv, segs[:, :, 0].ravel()])
    ys = np.concatenate([yv, segs[:, :, 1].ravel()])
    if log:
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        fmt = plt.FuncFormatter(lambda v, _: f'{v:g}')
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(fmt)
        ax.set_xlim(min(xs.min(), 1) / pad, max(xs.max(), 1) * pad)
        ax.set_ylim(min(ys.min(), 1) / pad, max(ys.max(), 1) * pad)
    else:
        ax.set_xlim(0, max(xs.max(), 1) * 1.05)
        ax.set_ylim(0, max(ys.max(), 1) * 1.05)
    ax.set_xlabel('relative density outside shapes (1 = as expected)')
    ax.set_ylabel('relative density inside shapes (1 = as expected)')
    fig.tight_layout()
    return fig, ax


def plot_gene_multi(name, names, grid, rho, dens, per_dataset, ax=None, xlim=None):
    """
    Plot one name's uncorrected KDE, reference distribution and corrected
    relative density. Densities use the left axis; rho uses the right axis.

    Like plot_gene, for pooled results. Shows the reference this name was
    actually compared with, and adds each dataset's own corrected curve as a
    thin line so you can check that the datasets agree.

    per_dataset : the last value returned by area_corrected_kdes_multi
    xlim        : optional (low, high) range of distances to show
    """
    try:
        i = names.get_loc(name) if hasattr(names, 'get_loc') else list(names).index(name)
    except (KeyError, ValueError):
        raise KeyError(f"{name!r} not found in names") from None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    win, lims = _xwindow(grid, xlim)
    g = grid[win]
    h1, = ax.plot(g, dens[i][win], color='C0', lw=1.8, label='Uncorrected KDE')
    ref_i = per_dataset['ref_by_name'][i][win]
    h2 = ax.fill_between(g, ref_i, color='grey', alpha=0.25,
                         label='Uniform-placement reference')
    ax.set_xlabel('signed distance')
    ax.set_ylabel('density')
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    h3, = ax2.plot(g, rho[i][win], color='C3', lw=1.8,
                   label='Corrected (relative density)')
    ax2.axhline(1, color='C3', lw=0.7, alpha=0.5)
    handles = [h1, h2, h3]
    styles = ['--', '-.', ':', (0, (5, 1, 1, 1))]
    for j, (k, rk) in enumerate(per_dataset['rho'].items()):
        n_ik = per_dataset['n'].iloc[i][k]
        if n_ik > 0:
            h, = ax2.plot(g, rk[i][win], color='C3', lw=1, alpha=0.6,
                          ls=styles[j % len(styles)],
                          label=f'{k} (n = {int(n_ik):,})')
            handles.append(h)
    ax2.set_ylabel('relative density (1 = uniform)', color='C3')
    ax2.tick_params(axis='y', colors='C3')
    ax2.set_ylim(bottom=0)

    ax.axvline(0, color='grey', ls=':', lw=1.2)
    ax.set_xlim(*(lims if lims is not None else (grid[0], grid[-1])))
    ax.set_title(str(name), pad=28 + 14 * ((len(handles) - 1) // 3))
    ax.legend(handles=handles, loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    return fig, (ax, ax2)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
# from area_corrected_kde import *
# from multi_dataset_kde import *
#
# 1. Distances and a reference for each dataset, using ITS OWN shapes and
#    bounds. Coordinates must be in the same units (e.g. microns) everywhere.
# datasets = {}
# for label, pts, shapes, bounds in [('A', dfA, 'A.geojson', boundsA),
#                                    ('B', dfB, 'B.geojson', boundsB),
#                                    ('C', dfC, 'C.geojson', boundsC)]:
#     pts, sd = add_signed_distance(pts, shapes)
#     ref_d = make_reference_distances(sd, bounds)
#     np.save(f'ref_distances_{label}.npy', ref_d)
#     datasets[label] = (pts, ref_d)
#
# 2. Pooled area-corrected KDEs
# names, grid, rho, ref, bw, dens, per_ds = area_corrected_kdes_multi(datasets, T=20.0)
# plot_gene_multi('GeneA', names, grid, rho, dens, per_ds); plt.show()
#
# 3. Clustering and plots from area_corrected_kde work unchanged
# counts = per_ds['n'].sum(axis=1)
# labels, Z, scores = cluster_genes(names, rho, ref, counts=counts, min_count=200)
# plot_cluster_heatmap(labels, Z, names, grid, rho); plt.show()
# plot_ridgeline(names, grid, rho, genes=['GeneA', 'GeneB']); plt.show()
#
# 4. Inside vs outside, pooled
# io = inside_outside_enrichment_multi(datasets, T=20.0)
# plot_inside_outside(io, labels=labels); plt.show()
# plot_inside_vs_outside_multi(io, labels=labels); plt.show()
