import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import shapely
import geopandas as gpd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Signed distances to shape boundaries
# ---------------------------------------------------------------------------

def _distance_engine(polys, spacing):
    """Signed-distance function for one set of polygons (no labels).
    Missing or empty geometries are ignored; with none left, returns NaN."""
    polys = np.asarray(polys)
    polys = polys[~(shapely.is_missing(polys) | shapely.is_empty(polys))]
    if len(polys) == 0:
        return lambda x, y: np.full(len(np.asarray(x)), np.nan)
    bnd = shapely.boundary(polys)
    poly_tree = shapely.STRtree(polys)
    kd = None if spacing is None else cKDTree(
        shapely.get_coordinates(shapely.segmentize(bnd, spacing)))

    def signed_distance(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        pts = shapely.points(x, y)
        if kd is None:                                          # exact
            idx, dist = poly_tree.query_nearest(pts, return_distance=True,
                                                all_matches=False)
            d = np.full(len(x), np.nan)
            d[idx[0]] = dist
        else:                                                   # approximate
            d, _ = kd.query(np.column_stack([x, y]), workers=-1)
        i, j = poly_tree.query(pts, predicate='within')         # inside points
        depth = np.zeros(len(x))
        np.maximum.at(depth, i, shapely.distance(pts[i], bnd[j]))
        return np.where(depth > 0, -depth, d)

    return signed_distance


def make_signed_distance(shapes, spacing=None, label_col=None):
    """
    Build a fast signed-distance function for a set of polygons.

    shapes    : path to a GeoJSON file, or a GeoDataFrame / array of polygons
    spacing   : None -> exact distances.
                A number -> approximate outside distances (error <= ~spacing/2)
                using a KD-tree over boundary points spaced this far apart.
                Faster; keep it well below your KDE bandwidth.
    label_col : optional column in shapes. If given, each point is measured
                only against shapes with the same label as the point.

    Returns signed_distance(x, y), or signed_distance(x, y, labels) when
    label_col is given: distance to the nearest shape for points outside,
    minus the distance to the containing shape's boundary for points inside.
    Points whose label has no shapes get NaN.
    """
    if isinstance(shapes, str):
        shapes = gpd.read_file(shapes)
    if label_col is None:
        polys = (shapes.geometry.to_numpy() if hasattr(shapes, 'geometry')
                 else np.asarray(shapes))
        return _distance_engine(polys, spacing)

    if not hasattr(shapes, 'columns') or label_col not in shapes.columns:
        raise ValueError(f"label_col {label_col!r} not found in the shapes.")
    geoms = shapes.geometry.to_numpy()
    valid = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))
    groups = {lab: g.geometry.to_numpy()
              for lab, g in shapes.loc[valid].groupby(label_col)}
    engines = {}                       # built on first use, then reused

    def signed_distance(x, y, labels):
        x, y = np.asarray(x, float), np.asarray(y, float)
        codes, uniq = pd.factorize(np.asarray(labels))   # NaN label -> -1
        out = np.full(len(x), np.nan)
        order = np.argsort(codes, kind='stable')
        starts = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        missing = []
        for c, lab in enumerate(uniq):
            if lab not in groups:
                missing.append(lab)
                continue
            if lab not in engines:
                engines[lab] = _distance_engine(groups[lab], spacing)
            sel = order[starts[c]:starts[c + 1]]
            out[sel] = engines[lab](x[sel], y[sel])
        if missing:
            warnings.warn(
                f"{len(missing)} labels have no (non-empty) shapes; their "
                f"points get NaN: "
                f"{[str(m) for m in missing[:10]]}"
                f"{' ...' if len(missing) > 10 else ''}")
        return out

    return signed_distance


def add_signed_distance(df, shapes, spacing=None, x_col='x', y_col='y',
                        out_col='distance', label_col=None):
    """
    Return a copy of df with a signed-distance column added.

    df        : DataFrame with point coordinates in x_col / y_col
    shapes    : path to a GeoJSON file, or a GeoDataFrame of polygons
                (must use the same coordinate system as the points)
    label_col : optional column present in BOTH df and shapes; each point is
                measured only against shapes with the same label
    """
    signed_distance = make_signed_distance(shapes, spacing, label_col)
    out = df.copy()
    xy = out[x_col].to_numpy(), out[y_col].to_numpy()
    if label_col is None:
        out[out_col] = signed_distance(*xy)
    else:
        out[out_col] = signed_distance(*xy, out[label_col].to_numpy())
    return out, signed_distance


# ---------------------------------------------------------------------------
# Reference distribution and area-corrected KDEs
# ---------------------------------------------------------------------------

def _check_region(region, what='bounds'):
    """Return (sampler-ready region, area). Region = 4-tuple or polygon."""
    if hasattr(region, 'geom_type'):
        if region.area <= 0:
            raise ValueError(f"{what} polygon has zero area.")
        return region, region.area
    xmin, xmax, ymin, ymax = region
    if not (xmin < xmax and ymin < ymax):
        raise ValueError(
            f"{what} must be (xmin, xmax, ymin, ymax); got {tuple(region)}. "
            "Note: GeoDataFrame.total_bounds is (minx, miny, maxx, maxy).")
    return (xmin, xmax, ymin, ymax), (xmax - xmin) * (ymax - ymin)


def _sample_region(region, n, rng):
    """n uniform random points inside a 4-tuple box or a polygon."""
    if not hasattr(region, 'geom_type'):
        xmin, xmax, ymin, ymax = region
        return rng.uniform(xmin, xmax, n), rng.uniform(ymin, ymax, n)
    minx, miny, maxx, maxy = region.bounds
    shapely.prepare(region)
    xs, ys, got = [], [], 0
    while got < n:
        m = int((n - got) * 1.2 * (maxx - minx) * (maxy - miny) / region.area) + 16
        x = rng.uniform(minx, maxx, m)
        y = rng.uniform(miny, maxy, m)
        keep = shapely.contains_xy(region, x, y)
        xs.append(x[keep]); ys.append(y[keep]); got += keep.sum()
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n]


def label_bounds(df, label_col, x_col='x', y_col='y'):
    """{label: (xmin, xmax, ymin, ymax)} from the extent of each label's points."""
    g = df.groupby(label_col)
    return {lab: (float(r[(x_col, 'min')]), float(r[(x_col, 'max')]),
                  float(r[(y_col, 'min')]), float(r[(y_col, 'max')]))
            for lab, r in g[[x_col, y_col]].agg(['min', 'max']).iterrows()}


def make_reference_distances(dist_fn, bounds, n_ref=1_000_000, seed=0):
    """
    Signed distances of uniformly random points in the study area.
    Run ONCE (shapes and study area are shared by all names) and save.

    dist_fn : the function from make_signed_distance / add_signed_distance,
              i.e. computed exactly the way df['distance'] was
    bounds  : the study area, as (xmin, xmax, ymin, ymax) or a shapely polygon.
              With labels (dist_fn made with label_col): a dict
              {label: (xmin, xmax, ymin, ymax) or polygon} giving each label's
              own area, e.g. from label_bounds(df, label_col). Points are
              spread over the labels in proportion to their areas and each is
              measured against its own label's shapes.
    """
    rng = np.random.default_rng(seed)
    if isinstance(bounds, dict):
        regions, areas = {}, {}
        for lab, r in bounds.items():
            regions[lab], areas[lab] = _check_region(r, f"bounds[{lab!r}]")
        labs = list(regions)
        a = np.array([areas[l] for l in labs], float)
        n_each = np.floor(n_ref * a / a.sum()).astype(int)
        n_each[np.argsort(-(n_ref * a / a.sum() - n_each))[:n_ref - n_each.sum()]] += 1
        xs, ys, ls = [], [], []
        for lab, n in zip(labs, n_each):
            if n == 0:
                continue
            x, y = _sample_region(regions[lab], n, rng)
            xs.append(x); ys.append(y); ls.append(np.full(n, lab, dtype=object))
        ref_d = np.asarray(dist_fn(np.concatenate(xs), np.concatenate(ys),
                                   np.concatenate(ls)), float)
    else:
        region, _ = _check_region(bounds)
        x, y = _sample_region(region, n_ref, rng)
        ref_d = np.asarray(dist_fn(x, y), float)

    bad = np.isnan(ref_d)
    if bad.any():
        warnings.warn(f"Dropping {bad.sum():,} reference points with no "
                      "distance (labels without shapes).")
        ref_d = ref_d[~bad]
    if not (ref_d < 0).any():
        warnings.warn(
            "No reference points fell inside any shape. Check that bounds are "
            "(xmin, xmax, ymin, ymax) and in the same coordinates as the shapes.")
    return ref_d


def _kde_grid(lo, hi, n_bins, boundary=None):
    """Bin layout for the KDEs. With a boundary, the grid is shifted so a bin
    edge falls exactly on it (bins never straddle the boundary)."""
    dx = (hi - lo) / n_bins
    if boundary is not None:
        lo = boundary - np.ceil((boundary - lo) / dx) * dx
        n_bins = int(np.ceil((hi - lo) / dx))
    grid = lo + dx * (np.arange(n_bins) + 0.5)
    return lo, dx, n_bins, grid


def _kde_smoother(grid, sigma, boundary=None):
    """
    Gaussian smoothing of binned counts along the last (or given) axis.

    boundary=None : ordinary smoothing (points spread across every value).
    boundary=b    : b is treated as a hard edge. Values below and above b are
                    smoothed separately, with a local-linear boundary
                    correction on each side (Jones 1993): near the edge the
                    estimate accounts for both the part of the smoothing
                    window that falls on the other side and the slope of the
                    density, so points next to the edge keep their full
                    weight and steep profiles aren't flattened. Away from the
                    edge it is identical to ordinary smoothing. Applied the
                    same way to transcripts and reference.
    """
    def plain(a, axis=-1):
        return gaussian_filter1d(np.asarray(a, float), sigma, axis=axis,
                                 mode='constant', truncate=4)
    if boundary is None:
        return plain
    u = (grid - grid[0]) / (grid[1] - grid[0])     # position in bins
    sides = [grid < boundary, grid >= boundary]
    # Moments of the smoothing window that fall on each side, per position
    mom = []
    for side in sides:
        s0 = plain(side.astype(float))
        s1 = plain(side * u) - u * s0
        s2 = plain(side * u ** 2) - 2 * u * plain(side * u) + u ** 2 * s0
        mom.append((s0, s1, s2))

    def smooth(a, axis=-1):
        a = np.asarray(a, float)
        shape = [1] * a.ndim
        shape[axis] = -1
        ur = u.reshape(shape)
        out = np.zeros_like(a)
        for side, (s0, s1, s2) in zip(sides, mom):
            sm = side.reshape(shape)
            c0 = plain(a * sm, axis=axis)
            c1 = plain(a * sm * ur, axis=axis) - ur * c0
            det = (s0 * s2 - s1 ** 2).reshape(shape)
            with np.errstate(divide='ignore', invalid='ignore'):
                f = (s2.reshape(shape) * c0 - s1.reshape(shape) * c1) / det
                f_renorm = c0 / s0.reshape(shape)
            # fall back to plain renormalisation where the linear correction
            # is undefined or goes negative (very sparse data at the edge)
            f = np.where(np.isfinite(f) & (f >= 0), f, f_renorm)
            out += np.where(sm, np.nan_to_num(f), 0.0)
        return out
    return smooth


def _kde_variance_inflation(grid, sigma, boundary=None):
    """
    How much noisier the (boundary-corrected) smoothed estimate is at each
    grid position than in the interior: 1 away from edges, larger close to
    the boundary, where the estimate rests on data from one side only.
    Computed from the local-linear weights' sum of squares.
    """
    if boundary is None:
        return np.ones(len(grid))
    u = (grid - grid[0]) / (grid[1] - grid[0])

    def plain(a, s_):
        return gaussian_filter1d(np.asarray(a, float), s_, mode='constant', truncate=4)

    # squared Gaussian weights = scaled Gaussian with sigma / sqrt(2)
    c2 = 1 / (2 * np.sqrt(np.pi) * sigma)
    infl = np.zeros(len(grid))
    for side in (grid < boundary, grid >= boundary):
        s0 = plain(side, sigma)
        s1 = plain(side * u, sigma) - u * s0
        s2 = plain(side * u ** 2, sigma) - 2 * u * plain(side * u, sigma) + u ** 2 * s0
        t = sigma / np.sqrt(2)
        q0 = c2 * plain(side, t)
        q1 = c2 * plain(side * u, t) - u * q0
        q2 = c2 * (plain(side * u ** 2, t) - 2 * u * plain(side * u, t)) + u ** 2 * q0
        with np.errstate(divide='ignore', invalid='ignore'):
            v = (s2 ** 2 * q0 - 2 * s1 * s2 * q1 + s1 ** 2 * q2) / (s0 * s2 - s1 ** 2) ** 2
        infl = np.where(side, v / c2, infl)
    return np.where(np.isfinite(infl) & (infl > 0), infl, np.inf)


def area_corrected_kdes(df, T, ref_d, n_bins=4096, ref_floor=1e-3, boundary=None):
    """
    Area-corrected KDE of signed distance for each name.

    df       : DataFrame with columns name, x, y, distance
    T        : right-side threshold; results are returned for distance <= T
    ref_d    : reference distances from make_reference_distances
               (e.g. loaded with np.load)
    ref_floor: grid points where the reference is below this fraction of its
               max are masked (NaN), since the ratio is unstable there
    boundary : None (default) or a distance to treat as a hard edge, usually
               0. Use 0 when the points can only lie on one side of the shape
               boundary, or their density jumps there - for example
               unassigned transcripts against cell polygons, which are almost
               all outside cells. Without it the smoothing spreads points
               across 0 and the curve dips artificially just next to it.

    Returns names, grid, rho, ref, bw, dens
        rho[i]  : relative density for names[i] on grid
                  (1 = as common as uniform placement, >1 enriched, <1 depleted)
        ref     : reference (uniform-placement) density on grid
        dens[i] : uncorrected KDE for names[i] on grid
        ref and dens integrate to 1 over distance <= T.
    """
    df = df.dropna(subset=['name', 'distance'])
    codes, names = pd.factorize(df['name'])
    d = df['distance'].to_numpy(float)
    k = len(names)

    # 1. Shared bandwidth: median of per-name Silverman bandwidths
    g = df.groupby(codes)['distance']
    iqr = g.quantile(.75) - g.quantile(.25)
    bw_each = 0.9 * np.minimum(g.std(), iqr / 1.34) * g.size() ** -0.2
    bw = float(bw_each[bw_each > 0].median())

    # 2. Reference distances (precomputed)
    ref_d = np.asarray(ref_d, float)

    # Shared grid covering data and reference, padded on both ends
    lo = min(d.min(), ref_d.min()) - 5 * bw
    hi = max(d.max(), ref_d.max()) + 5 * bw
    lo, dx, n_bins, grid = _kde_grid(lo, hi, n_bins, boundary)
    sigma = bw / dx
    if sigma < 5:
        print(f"Warning: bw/dx = {sigma:.1f}; consider increasing n_bins.")

    def to_bins(v):
        return np.clip(np.floor((v - lo) / dx).astype(int), 0, n_bins - 1)

    smooth = _kde_smoother(grid, sigma, boundary)

    ref = smooth(np.bincount(to_bins(ref_d), minlength=n_bins))

    # 4. Per-name KDEs on the full grid, using ALL points (including > T)
    counts = np.bincount(codes * n_bins + to_bins(d), minlength=k * n_bins)
    dens = smooth(counts.reshape(k, n_bins), axis=1)

    # 3 & 5. Truncate at T (after smoothing) and normalize over d <= T
    keep = grid <= T
    grid, ref, dens = grid[keep], ref[keep], dens[:, keep]
    ref /= ref.sum() * dx
    tot = dens.sum(axis=1, keepdims=True)
    dens /= np.where(tot > 0, tot, np.nan) * dx

    # 6. Mask where the reference is near zero
    ok = ref > ref_floor * ref.max()

    # 7. Divide
    rho = np.full_like(dens, np.nan)
    rho[:, ok] = dens[:, ok] / ref[ok]
    return names, grid, rho, ref, bw, dens


def relative_to(names, rho, dens, n, bw, grid=None, boundary=None,
                reference='__ALL__', min_count=10):
    """
    Each name's curve relative to another name's, usually '__ALL__' (a sample
    of all unassigned transcripts):
        rel[i] = rho[i] / rho[reference]
    1 = placed like the reference at that distance, > 1 more often, < 1 less
    often.

    Where the reference itself is rare - for example inside cells, when only
    a few unassigned transcripts lie inside polygons - both curves are close
    to 0 and their ratio is meaningless (anything from 0 to infinity). So
    each name's distances are masked (NaN) unless the name would have at
    least min_count transcripts within one bandwidth of that distance if it
    were placed like the reference (n x reference density x 2 bw), counted
    as an effective number: next to a boundary the estimate rests on one
    side only and is noisier, so the count is divided by that extra
    variance. Sparse names are therefore masked over a wider range than
    abundant ones.

    names, rho, dens, bw : from area_corrected_kdes or
                area_corrected_kdes_multi
    n         : transcripts per name used for the curves (distance <= T):
                a Series indexed by name, or an array in the order of names.
                Multi version: per_dataset['n'].sum(axis=1) (or
                per_dataset['n'][sample] for one sample's curves). Single
                version: df.loc[df['distance'] <= T, 'name'].value_counts()
    grid, boundary : pass the grid and the same boundary used for the
                curves, so the extra noise right next to the boundary is
                taken into account
    Returns rel, same shape as rho.
    """
    names = pd.Index(names)
    i_ref = names.get_loc(reference)
    if hasattr(n, 'reindex'):
        n = n.reindex(names).fillna(0).to_numpy(float)
    n = np.asarray(n, float)
    d_ref = np.nan_to_num(dens[i_ref])
    infl = np.ones(len(d_ref))
    if grid is not None and boundary is not None:
        grid = np.asarray(grid, float)
        infl = _kde_variance_inflation(grid, bw / (grid[1] - grid[0]), boundary)
    share = d_ref * 2 * bw / infl
    expected = n[:, None] * share[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.asarray(rho, float) / np.asarray(rho, float)[i_ref][None, :]
    rel[(expected < min_count) | ~np.isfinite(rel)] = np.nan
    return rel


def rho_distance_matrix(rho, ref):
    """Reference-weighted mean absolute difference between rho curves."""
    ok = ~np.isnan(rho).all(axis=0)      # columns not masked by ref_floor
    w = ref[ok] / ref[ok].sum()
    return squareform(pdist(rho[:, ok] * w, 'cityblock'))


# ---------------------------------------------------------------------------
# Grouping names by the shape of their corrected KDE
# ---------------------------------------------------------------------------

def _cluster_color(k):
    return 'lightgrey' if k == 0 else f'C{(int(k) - 1) % 10}'


def _cluster_name(k):
    return 'Unassigned' if k == 0 else f'Cluster {int(k)}'


def _relabel_by_size(lab, min_size):
    """Clusters smaller than min_size -> 0; others renumbered 1.. by size."""
    lab = np.asarray(lab)
    out = np.zeros(len(lab), int)
    vals, cnt = np.unique(lab[lab > 0], return_counts=True)
    keep = [v for v, c in sorted(zip(vals, cnt), key=lambda t: -t[1])
            if c >= min_size]
    for new, v in enumerate(keep, start=1):
        out[lab == v] = new
    return out


def cluster_genes(names, rho, ref, method='complete', max_dist=0.5,
                  n_clusters=None, max_clusters=10, min_size=3,
                  min_cluster_size=5, n_pcs=10, counts=None, min_count=0,
                  log=True, log_floor=0.1, weight='reference'):
    """
    Group names by their corrected curves.

    Each curve is compared as log2(rho) at every grid point, weighted by
    available area. The distance between two names is then the weighted RMS
    difference between their curves in log2 units (0.5 ~ curves differ by
    about 1.4x on average).

    method :
      'complete' (default, strict) - complete linkage cut at max_dist: EVERY
                 pair of names in a cluster differs by at most max_dist.
                 Lower max_dist -> stricter, more and smaller clusters.
      'hdbscan'  - density-based clustering on the top n_pcs principal
                 components; finds groups of similar curves and leaves
                 names that fit no group unassigned. min_cluster_size sets
                 the smallest group.
      'ward'     - Ward linkage; n_clusters, or chosen by silhouette over
                 2..max_clusters. Tends to give few broad clusters.
    min_size   : clusters (from 'complete' or 'ward') smaller than this are
                 marked unassigned
    counts     : optional Series of points per name (df['name'].value_counts());
                 names with fewer than min_count points are left out entirely
    log, log_floor, weight : see notes above (weight='uniform' weights all
                 distances equally)

    Returns labels, Z, scores
        labels : Series (index = name, value = cluster). 1 is the largest
                 cluster; 0 = unassigned (didn't fit any cluster)
        Z      : linkage matrix (rows in the order of labels.index); used for
                 the heatmap ordering. For 'hdbscan' it's an average-linkage
                 tree for display only.
        scores : 'ward' with n_clusters=None -> silhouette by k;
                 'hdbscan' -> membership strength per name (0..1);
                 otherwise empty
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    names = pd.Index(names)
    # Genes with any estimate (and enough transcripts); then the distances
    # that every one of those genes has, so a gene masked at some distances
    # (e.g. by relative_to) narrows the distances compared rather than being
    # dropped
    fin = np.isfinite(rho)
    use = fin.any(axis=1)
    if counts is not None:
        use &= counts.reindex(names).fillna(0).to_numpy() >= min_count
    cols = fin[use].all(axis=0)
    if not cols.any():
        raise ValueError("No distance is defined for every gene being clustered; "
                         "raise min_count or check the curves for NaN.")

    X = rho[np.ix_(use, cols)]
    if log:
        X = np.log2(np.maximum(X, log_floor))
    w = ref[cols] if weight == 'reference' else np.ones(cols.sum())
    X = X * np.sqrt(w / w.sum())          # Euclidean = weighted RMS difference
    idx = names[use]
    scores = pd.Series(dtype=float)

    if method == 'complete':
        Z = linkage(X, method='complete')
        lab = (fcluster(Z, n_clusters, 'maxclust') if n_clusters
               else fcluster(Z, max_dist, 'distance'))
        lab = _relabel_by_size(lab, min_size)

    elif method == 'ward':
        from sklearn.metrics import silhouette_score
        Z = linkage(X, method='ward')
        if n_clusters is None:
            ks = range(2, min(max_clusters, len(X) - 1) + 1)
            scores = pd.Series({k: silhouette_score(X, fcluster(Z, k, 'maxclust'))
                                for k in ks}, name='silhouette')
            n_clusters = int(scores.idxmax())
        lab = _relabel_by_size(fcluster(Z, n_clusters, 'maxclust'), min_size)

    elif method == 'hdbscan':
        from sklearn.cluster import HDBSCAN
        Xc = X - X.mean(axis=0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        P = U[:, :n_pcs] * S[:n_pcs]
        hd = HDBSCAN(min_cluster_size=min_cluster_size).fit(P)
        lab = _relabel_by_size(hd.labels_ + 1, 1)     # -1 noise -> 0
        scores = pd.Series(hd.probabilities_, index=idx, name='membership')
        Z = linkage(X, method='average')

    else:
        raise ValueError("method must be 'complete', 'hdbscan' or 'ward'")

    labels = pd.Series(lab, index=idx, name='cluster')
    return labels, Z, scores


def _xwindow(grid, xlim):
    """
    Distances to show. xlim = None (all) or (low, high), where either end may
    be None to keep the default there. Returns (mask over grid, limits or
    None). Data are cropped to the window before plotting, so y ranges,
    colour scales and ridgeline heights fit what is shown.
    """
    if xlim is None:
        return np.ones(len(grid), bool), None
    lo, hi = xlim
    lo = grid[0] if lo is None else lo
    hi = grid[-1] if hi is None else hi
    if not lo < hi:
        raise ValueError(f"xlim must be (low, high) with low < high; got {xlim}")
    m = (grid >= lo) & (grid <= hi)
    if not m.any():
        raise ValueError(f"xlim {xlim} contains no distances (curves cover "
                         f"{grid[0]:.2f} to {grid[-1]:.2f})")
    return m, (lo, hi)


def plot_cluster_profiles(labels, names, grid, rho, ncols=3, log=True,
                          log_floor=0.1, xlim=None):
    """One panel per cluster: each member's corrected curve plus the median.
    Unassigned names (label 0) get their own panel at the end.
    xlim : optional (low, high) range of distances to show."""
    names = pd.Index(names)
    win, lims = _xwindow(grid, xlim)
    grid, rho = grid[win], rho[:, win]
    ks = sorted(k for k in labels.unique() if k != 0)
    if (labels == 0).any():
        ks.append(0)
    nrows = int(np.ceil(len(ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, k in zip(axes.flat, ks):
        rows = rho[names.get_indexer(labels.index[labels.values == k])]
        rows = np.where(np.isfinite(rows), rows, np.nan)
        if log:
            rows = np.maximum(rows, log_floor)
        alpha = max(0.05, min(0.6, 5 / len(rows)))
        ax.plot(grid, rows.T, color=_cluster_color(k) if k else 'grey',
                lw=0.6, alpha=alpha)
        if k != 0:
            with np.errstate(all='ignore'), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med = np.nanmedian(rows, axis=0)
            ax.plot(grid, med, color='k', lw=2)
        ax.axhline(1, color='k', lw=0.6, alpha=0.5)
        ax.axvline(0, color='grey', ls=':', lw=1.2)
        ax.set_title(f'{_cluster_name(k)}  (n = {len(rows)})')
        if log:
            ax.set_yscale('log', base=2)
    for ax in axes.flat[len(ks):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('signed distance')
    for ax in axes[:, 0]:
        ax.set_ylabel('relative density')
    if lims is not None:
        axes[0, 0].set_xlim(*lims)
    else:
        valid = grid[~np.isnan(rho).all(axis=0)]
        axes[0, 0].set_xlim(valid[0], valid[-1])
    fig.tight_layout()
    return fig, axes


def plot_cluster_heatmap(labels, Z, names, grid, rho, log_floor=0.1,
                         order='cluster', show_names=None, xlim=None):
    """
    Heatmap of log2(rho), red = enriched, blue = depleted, grey = no
    estimate at that distance (e.g. masked by relative_to).

    order : 'cluster' -> rows grouped by cluster (1, 2, ..., unassigned last),
                         tree order within each cluster; no dendrogram
            'tree'    -> rows in tree order with the dendrogram shown
    xlim  : optional (low, high) range of distances to show; the colour
            scale is set from the distances shown
    """
    from scipy.cluster.hierarchy import dendrogram, leaves_list

    names = pd.Index(names)
    win, lims = _xwindow(grid, xlim)
    cols = ~np.isnan(rho).all(axis=0) & win
    g = grid[cols]
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.log2(np.maximum(rho[names.get_indexer(labels.index)][:, cols],
                               log_floor))
    M = np.where(np.isfinite(M), M, np.nan)
    lab_all = labels.to_numpy()
    n = len(labels)

    fig = plt.figure(figsize=(10, min(12, 2 + 0.15 * n)))
    tree = order == 'tree'
    left = 0.205 if tree else 0.08
    ax_c = fig.add_axes([left, 0.1, 0.015, 0.8])
    ax_h = fig.add_axes([left + 0.02, 0.1, 0.845 - left - 0.02, 0.8])
    ax_cb = fig.add_axes([0.87, 0.1, 0.02, 0.8])

    if tree:
        ax_d = fig.add_axes([0.05, 0.1, 0.15, 0.8])
        dn = dendrogram(Z, orientation='left', no_labels=True, ax=ax_d,
                        color_threshold=0, above_threshold_color='k')
        rows = np.array(dn['leaves'][::-1])
        ax_d.axis('off')
    else:
        leaf_pos = np.empty(n, int)
        leaf_pos[leaves_list(Z)] = np.arange(n)
        key = np.where(lab_all == 0, lab_all.max() + 1, lab_all)
        rows = np.lexsort((leaf_pos, key))

    lab = lab_all[rows]
    colors = [plt.matplotlib.colors.to_rgb(_cluster_color(k)) for k in lab]
    ax_c.imshow(np.array(colors)[:, None, :], aspect='auto',
                interpolation='nearest')
    ax_c.axis('off')

    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('lightgrey')                       # no estimate at that distance
    im = ax_h.imshow(M[rows], aspect='auto', cmap=cmap, vmin=-vmax,
                     vmax=vmax, interpolation='nearest',
                     extent=[g[0], g[-1], n - 0.5, -0.5])
    if not tree:
        for b in np.flatnonzero(np.diff(lab)) + 0.5:
            ax_h.axhline(b, color='k', lw=0.8)
    ax_h.axvline(0, color='grey', ls=':', lw=1.2)
    if lims is not None:
        ax_h.set_xlim(*lims)
    ax_h.set_xlabel('signed distance')
    if show_names is None:
        show_names = n <= 60
    if show_names:
        ax_h.set_yticks(range(n), labels.index[rows], fontsize=7)
        ax_h.yaxis.tick_right()
        ax_cb.set_position([0.93, 0.1, 0.015, 0.8])
    else:
        ax_h.set_yticks([])
    fig.colorbar(im, cax=ax_cb, label='log2 relative density')
    return fig


# ---------------------------------------------------------------------------
# Inside (d < 0) vs outside (0 <= d <= T) enrichment
# ---------------------------------------------------------------------------

def inside_outside_enrichment(df, ref_d, T):
    """
    Per-name enrichment inside shapes vs outside, from raw counts (no KDE
    smoothing, so nothing leaks across the boundary). Only d <= T is used.

    Columns returned (index = name):
        n, n_in, n_out : points with d <= T, inside (d < 0), outside
        frac_in        : n_in / n
        rho_in/rho_out : relative density inside / outside
                         (share of points / share of area; 1 = as expected)
        log2_ratio     : log2(rho_out / rho_in); < 0 favours inside,
                         > 0 favours outside
        ci_low/ci_high : approximate 95% interval for log2_ratio
        z, pvalue      : Wald test of log2_ratio = 0
    Intervals and p-values assume independent points; spatially clustered
    points (e.g. many per cell) make them too optimistic.
    """
    from scipy.stats import norm

    ref_d = np.asarray(ref_d, float)
    ref_d = ref_d[ref_d <= T]
    R_in, R_out = (ref_d < 0).sum(), (ref_d >= 0).sum()
    if R_in == 0 or R_out == 0:
        raise ValueError(
            f"Reference has {R_in} points inside shapes and {R_out} outside "
            f"(d <= T). Both are needed. Make sure you passed ref_d (the raw "
            f"reference distances), not ref (the smoothed density returned by "
            f"area_corrected_kdes). Otherwise, check the bounds and coordinate system "
            f"used in make_reference_distances, and that ref_d was made with "
            f"the same shapes as df['distance'].")
    a_in, a_out = R_in / len(ref_d), R_out / len(ref_d)

    sub = df.loc[df['distance'] <= T]
    inside = sub['distance'] < 0
    g = inside.groupby(sub['name'])
    n_in = g.sum().astype(float)
    n = g.size().astype(float)
    n_out = n - n_in

    out = pd.DataFrame({'n': n, 'n_in': n_in, 'n_out': n_out})
    out['frac_in'] = n_in / n
    out['rho_in'] = (n_in / n) / a_in
    out['rho_out'] = (n_out / n) / a_out
    # 0.5 continuity correction keeps names with no inside/outside points finite
    lor = np.log((n_out + 0.5) / (n_in + 0.5)) - np.log(R_out / R_in)
    se = np.sqrt(1 / (n_in + 0.5) + 1 / (n_out + 0.5) + 1 / R_in + 1 / R_out)
    out['log2_ratio'] = lor / np.log(2)
    out['ci_low'] = (lor - 1.96 * se) / np.log(2)
    out['ci_high'] = (lor + 1.96 * se) / np.log(2)
    out['z'] = lor / se
    out['pvalue'] = 2 * norm.sf(np.abs(out['z']))
    return out


def plot_inside_outside(stats, y='count', labels=None, genes=None,
                        annotate=10, ax=None):
    """
    Enrichment inside vs outside shapes for every name.

    x : stats['log2_ratio'] = log2(relative density outside / inside)
        (< 0 favours inside, > 0 favours outside)
    y : 'count'  -> number of points (log scale), with 95% interval bars;
                    shows effect size against how much data supports it
        'pvalue' -> -log10 p-value (classic volcano)
    labels   : optional cluster labels (from cluster_genes) to colour points
    genes    : names to always annotate and outline
    annotate : also annotate this many names with the largest |z|
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    x = stats['log2_ratio']
    x_lo, x_hi = stats['ci_low'], stats['ci_high']
    if y == 'count':
        yv = stats['n']
        ax.errorbar(x, yv, xerr=[x - x_lo, x_hi - x],
                    fmt='none', ecolor='grey', elinewidth=0.6, alpha=0.4,
                    zorder=1)
        ax.set_yscale('log')
        fmt = plt.FuncFormatter(lambda v, _: f'{v:,.0f}')
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(fmt)
        ax.set_ylabel('number of points (d \u2264 T)')
    elif y == 'pvalue':
        yv = -np.log10(stats['pvalue'].clip(lower=1e-300))
        ax.set_ylabel('-log10 p-value')
    else:
        raise ValueError("y must be 'count' or 'pvalue'")

    if labels is not None:
        lab = labels.reindex(stats.index)
        for k in sorted(lab.dropna().unique()):
            m = (lab == k).to_numpy()
            ax.scatter(x[m], yv[m], s=18, color=_cluster_color(k),
                       label=_cluster_name(k), zorder=2)
        m = lab.isna().to_numpy()
        if m.any():
            ax.scatter(x[m], yv[m], s=12, color='lightgrey',
                       label='Not clustered', zorder=2)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(x, yv, s=18, color='C0', zorder=2)

    to_label = set(stats['z'].abs().nlargest(annotate).index) if annotate else set()
    if genes is not None:
        genes = [g for g in genes if g in stats.index]
        ax.scatter(x[genes], yv[genes], s=60, facecolors='none',
                   edgecolors='k', lw=1.2, zorder=3)
        to_label |= set(genes)
    for g in to_label:
        ax.annotate(str(g), (x[g], yv[g]), xytext=(4, 3),
                    textcoords='offset points', fontsize=8, zorder=4)

    ax.axvline(0, color='grey', ls=':', lw=1.2, zorder=0)
    ax.set_xlabel('log2 enrichment, outside vs inside\n'
                  '(< 0 favours inside,  > 0 favours outside)')
    fig.tight_layout()
    return fig, ax


def plot_inside_vs_outside(stats, labels=None, genes=None, annotate=10,
                           log=True, floor=0.01, ax=None):
    """
    Relative density inside shapes (y) against outside shapes (x), per name.

    stats    : output of inside_outside_enrichment
    labels   : optional cluster labels (from cluster_genes) to colour points
    genes    : names to always annotate and outline
    annotate : also annotate this many names with the largest |z|
    log      : log2 axes (2x enrichment and 2x depletion equally far from 1)
    floor    : values below this are drawn at the floor on log axes

    Grey bars are approximate 95% intervals. Because a name's share of points
    inside and outside must add up to 1, every name falls on the dashed line;
    its position along the line is what differs between names.
    """
    from matplotlib.collections import LineCollection

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    st = stats
    ok = st['n_in'] > 0
    a_in = float((st.loc[ok, 'frac_in'] / st.loc[ok, 'rho_in']).median())
    a_out = 1 - a_in
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

    # Constraint line: a_in * rho_in + a_out * rho_out = 1
    t = np.linspace(1e-4, 1 - 1e-4, 400)
    ax.plot(clip((1 - t) / a_out), clip(t / a_in), color='grey', ls='--',
            lw=0.8, zorder=0)

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


def plot_gene(name, names, grid, rho, ref, dens, ax=None, xlim=None):
    """
    Plot one name's uncorrected KDE, reference distribution and corrected
    relative density. Densities use the left axis; rho uses the right axis.
    xlim : optional (low, high) range of distances to show.
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
    h2 = ax.fill_between(g, np.asarray(ref)[win], color='grey', alpha=0.25,
                         label='Uniform-placement reference')
    ax.set_xlabel('signed distance')
    ax.set_ylabel('density')
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    h3, = ax2.plot(g, rho[i][win], color='C3', lw=1.8,
                   label='Corrected (relative density)')
    ax2.axhline(1, color='C3', lw=0.7, alpha=0.5)
    ax2.set_ylabel('relative density (1 = uniform)', color='C3')
    ax2.tick_params(axis='y', colors='C3')
    ax2.set_ylim(bottom=0)

    ax.axvline(0, color='grey', ls=':', lw=1.2)
    ax.set_xlim(*(lims if lims is not None else (grid[0], grid[-1])))
    ax.set_title(str(name), pad=28)
    ax.legend(handles=[h1, h2, h3], loc='lower center', ncol=3,
              bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    return fig, (ax, ax2)


def plot_ridgeline(names, grid, rho, genes=None, sort=True, height=0.9, xlim=None):
    """
    One row per name; dashed line marks rho = 1, dotted line marks distance 0.

    genes  : optional list of names to plot (default: all names)
    sort   : True  -> rows sorted by mean distance (lowest at the bottom)
             False -> rows in the order given in genes (first at the top)
    height : peak height relative to row spacing; all rows share one scale
    xlim   : optional (low, high) range of distances to show; heights and
             sorting are based on the distances shown
    """
    names = pd.Index(names)
    win, lims = _xwindow(grid, xlim)
    grid, rho = grid[win], rho[:, win]
    rho = np.where(np.isfinite(rho), rho, np.nan)
    if genes is None:
        idx = np.arange(len(names))
    else:
        genes = list(dict.fromkeys(genes))            # drop duplicates
        missing = [g for g in genes if g not in names]
        if missing:
            warnings.warn(f"Not found, skipped: {missing}")
        idx = names.get_indexer([g for g in genes if g in names])
        if len(idx) == 0:
            raise ValueError("None of the requested genes were found.")

    sub = rho[idx]
    if sort:
        w = np.nan_to_num(sub)
        with np.errstate(all='ignore'):
            order = np.argsort((w * grid).sum(axis=1) / w.sum(axis=1))
    else:
        order = np.arange(len(idx))[::-1]
    scale = height / np.nanmax(sub)

    n = len(idx)
    fig, ax = plt.subplots(figsize=(8, 0.4 * n + 1))
    for row, i in enumerate(order):
        z = n - row
        y = row + sub[i] * scale
        ax.fill_between(grid, row, y, alpha=0.7, zorder=z)
        ax.plot(grid, y, color='k', lw=0.8, zorder=z)
        ax.hlines(row + scale, grid[0], grid[-1], colors='grey',
                  linestyles='--', lw=0.5, zorder=z)
    ax.axvline(0, color='grey', ls=':', lw=1.2, zorder=n + 1)
    ax.set_yticks(range(n), names[idx][order])
    ax.set_xlabel('signed distance')
    if lims is not None:
        ax.set_xlim(*lims)
    else:
        valid = grid[~np.isnan(sub).all(axis=0)]
        ax.set_xlim(valid[0], valid[-1])
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
# df = pd.read_csv('points.csv')                    # columns: name, x, y
#
# 1. Distances (spacing=None -> exact)
# df, signed_distance = add_signed_distance(df, 'shapes.geojson')
#    or, measuring each point only against shapes with the same label:
# df, signed_distance = add_signed_distance(df, gdf, label_col='fov')
#
# 2. Reference, once (same shapes, same study area), then save
# ref_d = make_reference_distances(signed_distance,
#                                  bounds=(xmin, xmax, ymin, ymax))
#    or, with labels (each label's own area):
# ref_d = make_reference_distances(signed_distance,
#                                  bounds=label_bounds(df, 'fov'))
# np.save('ref_distances.npy', ref_d)
#
# 3. Area-corrected KDEs, comparison, plot
# ref_d = np.load('ref_distances.npy')
# names, grid, rho, ref, bw, dens = area_corrected_kdes(df, T=20.0, ref_d=ref_d)
# D = rho_distance_matrix(rho, ref)
# plot_ridgeline(names, grid, rho, genes=['GeneA', 'GeneB']); plt.show()
# plot_gene('GeneA', names, grid, rho, ref, dens); plt.show()
#
# 4. Group names by the shape of their corrected curves
# labels, Z, scores = cluster_genes(names, rho, ref, max_dist=0.5,
#                                   counts=df['name'].value_counts(),
#                                   min_count=200)          # strict (complete)
# labels, Z, scores = cluster_genes(names, rho, ref, method='hdbscan')
# plot_cluster_profiles(labels, names, grid, rho); plt.show()
# plot_cluster_heatmap(labels, Z, names, grid, rho); plt.show()
#
# 5. Inside vs outside enrichment
# io = inside_outside_enrichment(df, ref_d, T=20.0)
# plot_inside_outside(io, labels=labels); plt.show()
# plot_inside_vs_outside(io, labels=labels); plt.show()
# D = rho_distance_matrix(rho, ref)
# plot_ridgeline(names, grid, rho, genes=['GeneA', 'GeneB']); plt.show()
