from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

def _paper_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False
    })

def _save(fig, path=None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig

def plot_points_and_curve(points, curve, title="", closed=False, save_path=None):
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(curve[:,0], curve[:,1], linewidth=2.0, label="reconstruction")
    ax.scatter(points[:,0], points[:,1], s=20, zorder=3, label="input points")
    if closed:
        ax.plot([curve[-1,0], curve[0,0]], [curve[-1,1], curve[0,1]], linewidth=2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(frameon=False)
    return _save(fig, save_path)

def plot_method_comparison(points, curve_dict, title="", closed=False, save_path=None):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for name, curve in curve_dict.items():
        ax.plot(curve[:,0], curve[:,1], linewidth=2.0, label=name)
        if closed:
            ax.plot([curve[-1,0], curve[0,0]], [curve[-1,1], curve[0,1]], linewidth=2.0)
    ax.scatter(points[:,0], points[:,1], s=18, c="k", alpha=0.7, zorder=3, label="input")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    return _save(fig, save_path)

def plot_local_zoom(points, curve_dict, zoom_region, title="", closed=False, save_path=None):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for name, curve in curve_dict.items():
        ax.plot(curve[:,0], curve[:,1], linewidth=2.0, label=name)
        if closed:
            ax.plot([curve[-1,0], curve[0,0]], [curve[-1,1], curve[0,1]], linewidth=2.0)
    ax.scatter(points[:,0], points[:,1], s=18, c="k", alpha=0.7, zorder=3, label="input")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(frameon=False)
    x0, x1, y0, y1 = zoom_region
    inset = inset_axes(ax, width="34%", height="34%", loc="upper right")
    for curve in curve_dict.values():
        inset.plot(curve[:,0], curve[:,1], linewidth=1.6)
    inset.scatter(points[:,0], points[:,1], s=10, c="k", alpha=0.7)
    inset.set_xlim(x0, x1); inset.set_ylim(y0, y1)
    inset.set_xticks([]); inset.set_yticks([])
    ax.indicate_inset_zoom(inset, edgecolor="black")
    return _save(fig, save_path)

def plot_parameterization_comparison(points, curve_dict, title="", closed=False, save_path=None):
    _paper_style()
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    for ax, (name, curve) in zip(axs, curve_dict.items()):
        ax.plot(curve[:,0], curve[:,1], linewidth=2.0)
        ax.scatter(points[:,0], points[:,1], s=18, c="k", alpha=0.7)
        if closed:
            ax.plot([curve[-1,0], curve[0,0]], [curve[-1,1], curve[0,1]], linewidth=2.0)
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_path)

def plot_line_chart(df, x, y, hue, title="", save_path=None):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.2,5.2))
    for key, g in df.groupby(hue):
        g = g.sort_values(x)
        ax.plot(g[x].to_numpy(), g[y].to_numpy(), marker="o", linewidth=2.0, label=str(key))
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title); ax.legend(frameon=False)
    return _save(fig, save_path)

def plot_boxplot(df, x, y, title="", save_path=None):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.4,5.2))
    groups = [g[y].to_numpy() for _, g in df.groupby(x)]
    labels = [str(k) for k, _ in df.groupby(x)]
    ax.boxplot(groups, labels=labels, showfliers=True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
    return _save(fig, save_path)

def plot_violinplot(df, x, y, title="", save_path=None):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.4,5.2))
    groups = [g[y].to_numpy() for _, g in df.groupby(x)]
    labels = [str(k) for k, _ in df.groupby(x)]
    parts = ax.violinplot(groups, showmeans=True, showextrema=True, widths=0.85)
    for body in parts["bodies"]:
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
    return _save(fig, save_path)

def plot_spectrum(freqs, amps, title="", save_path=None):
    _paper_style()
    fig, ax = plt.subplots(figsize=(7.0,4.6))
    ax.stem(freqs, amps, basefmt=" ")
    ax.set_xlabel("frequency"); ax.set_ylabel("magnitude"); ax.set_title(title)
    return _save(fig, save_path)

def plot_epicycle_snapshot(chain, traced_curve=None, title="", save_path=None):
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.4,6.2))
    for seg in chain:
        center, end, radius = seg["center"], seg["end"], seg["radius"]
        ax.add_patch(plt.Circle(center, radius, fill=False, linestyle="--", alpha=0.5))
        ax.plot([center[0], end[0]], [center[1], end[1]], linewidth=1.8)
        ax.scatter([center[0]], [center[1]], s=10)
    if traced_curve is not None and len(traced_curve) > 0:
        ax.plot(traced_curve[:,0], traced_curve[:,1], linewidth=2.0, alpha=0.85)
    ax.set_title(title); ax.set_aspect("equal", adjustable="box")
    return _save(fig, save_path)

def save_epicycle_keyframes(approximator, k, n_frames, out_dir, prefix="epicycle"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    traced = []
    for i in range(n_frames):
        t = i / max(n_frames, 1)
        chain = approximator.epicycle_chain(t, k=k)
        traced.append(chain[-1]["end"])
        fig = plot_epicycle_snapshot(chain, traced_curve=np.asarray(traced), title=f"Epicycle keyframe {i+1}/{n_frames}, K={k}", save_path=out_dir / f"{prefix}_{i:03d}.png")
        plt.close(fig)

def animate_epicycle(approximator, k, n_frames=120, save_path=None):
    _paper_style()
    fig, ax = plt.subplots(figsize=(6.4,6.2))
    traced = []
    dense = approximator.dense_curve
    pad = 0.2 * max(np.ptp(dense[:,0]), np.ptp(dense[:,1]), 1.0)
    x0, x1 = dense[:,0].min() - pad, dense[:,0].max() + pad
    y0, y1 = dense[:,1].min() - pad, dense[:,1].max() + pad
    def update(i):
        ax.clear(); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal", adjustable="box")
        t = i / n_frames
        chain = approximator.epicycle_chain(t, k=k)
        traced.append(chain[-1]["end"])
        for seg in chain:
            center, end, radius = seg["center"], seg["end"], seg["radius"]
            ax.add_patch(plt.Circle(center, radius, fill=False, linestyle="--", alpha=0.5))
            ax.plot([center[0], end[0]], [center[1], end[1]], linewidth=1.5)
        trace = np.asarray(traced)
        if len(trace) > 1:
            ax.plot(trace[:,0], trace[:,1], linewidth=2.0)
        ax.set_title(f"Epicycle animation, K={k}, frame={i+1}/{n_frames}")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=50)
    if save_path is not None:
        path = Path(save_path); path.parent.mkdir(parents=True, exist_ok=True)
        try:
            anim.save(path, writer=PillowWriter(fps=20))
        except Exception:
            pass
    return anim
