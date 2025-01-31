"""Helper functions for plotting the datasets and model predictions."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns


# Colors
C_BLACK = '#000000'
C_WHITE = '#ffffff'
C_BLUE = '#093691'
C_RED = '#911b09'
C_BLACK_BLUEISH = '#011745'
C_BLACK_REDDISH = '#380801'
C_WHITE_BLUEISH = '#dce5f5'
C_WHITE_REDDISH = '#f5dcdc'

# Color maps
CMAP_BBR = matplotlib.colors.LinearSegmentedColormap.from_list(
  'blue_black_red',
  [C_WHITE_BLUEISH, C_BLUE, C_BLACK, C_RED, C_WHITE_REDDISH],
  N=200,
)
CMAP_BWR = matplotlib.colors.LinearSegmentedColormap.from_list(
  'blue_white_red',
  [C_BLACK_BLUEISH, C_BLUE, C_WHITE, C_RED, C_BLACK_REDDISH],
  N=200,
)
CMAP_WRB = matplotlib.colors.LinearSegmentedColormap.from_list(
  'white_red_black',
  [C_WHITE, C_RED, C_BLACK],
  N=200,
)

# General settings
plt.rcParams['font.family'] = 'serif'
SCATTER_SETTINGS = dict(marker='s', s=1, alpha=1, linewidth=0)
HATCH_SETTINGS = dict(facecolor='#b8b8b8', hatch='//////', edgecolor='#4f4f4f', linewidth=.0)

def plot_trajectory(u, x, t, idx_t, idx_s=0, symmetric=True, ylabels=None, domain=([0, 0], [1, 1])):
  """Plot the trajectory of a functions."""

  _WIDTH_PER_COL = 1.5
  _HEIGHT_PER_ROW = 1.5
  _WIDTH_MARGIN = .2
  _HEIGHT_MARGIN = .2
  _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
  _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .42 * _HEIGHT_PER_ROW
  _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (x.shape[2] ** .5)

  # Arrange the inputs
  n_vars = u.shape[-1]
  if isinstance(symmetric, bool):
    symmetric = [symmetric] * n_vars

  # Create the figure and the gridspec
  figsize=(_WIDTH_PER_COL*len(idx_t)+_WIDTH_MARGIN, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN)
  fig = plt.figure(figsize=figsize,)
  g = fig.add_gridspec(
    nrows=n_vars,
    ncols=len(idx_t)+1,
    width_ratios=([1]*len(idx_t) + [.1]),
    wspace=0.05,
    hspace=0.20,
  )
  # Add all axes
  axs = []
  for r in range(n_vars):
    row = []
    for c in range(len(idx_t)):
      row.append(fig.add_subplot(g[r, c]))
    axs.append(row)
  axs = np.array(axs)
  # Settings
  for ax in axs.flatten():
    ax: plt.Axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([domain[0][0], domain[1][0]])
    ax.set_ylim([domain[0][1], domain[1][1]])

  # Add hatch to the background
  print([np.min(x[..., 0]), np.max(x[..., 0])])
  for ax in axs.flatten():
    ax.fill_between(
      x=[domain[0][0], domain[1][0]], y1=domain[0][1], y2=domain[1][1],
      **HATCH_SETTINGS,
    )

  # Loop over variables
  for r in range(n_vars):
    # Set cmap and colorbar range
    if symmetric[r]:
      cmap = CMAP_BWR
      vmax = np.max(np.abs(u[idx_s, idx_t, ..., r]))
      vmin = -vmax
    else:
      cmap = CMAP_WRB
      vmax = np.max(u[idx_s, idx_t, ..., r])
      vmin = np.min(u[idx_s, idx_t, ..., r])

    # Loop over columns
    for icol in range(len(idx_t)):
      h = axs[r, icol].scatter(
        x=x[idx_s, idx_t[icol], ..., 0],
        y=x[idx_s, idx_t[icol], ..., 1],
        c=u[idx_s, idx_t[icol], ..., r],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **_SCATTER_SETTINGS,
      )
      if (r == 0) and (len(idx_t) > 1):
        axs[r, icol].set(title=f'$t=t_{{{idx_t[icol]}}}$')

    # Add colorbar
    ax_cb = fig.add_subplot(g[r, -1])
    cb = plt.colorbar(h, cax=ax_cb)
    cb.formatter.set_powerlimits((0, 0))
    ax_cb.yaxis.get_offset_text().set(size=8)
    ax_cb.yaxis.set_tick_params(labelsize=8)

  # Add ylabels
  for r in range(n_vars):
    label = ylabels[r] if ylabels else f'Variable {r:02d}'
    axs[r, 0].set(ylabel=label);

  return fig, axs

def plot_estimates(u_inp, u_gtr, u_prd, x_inp, x_out, symmetric=True, names=None, domain=([0, 0], [1, 1])):
  """Plot the model estimates along with ground-truth function."""

  _HEIGHT_PER_ROW = 1.9
  _HEIGHT_MARGIN = .2
  _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
  _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .4 * _HEIGHT_PER_ROW
  _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (x_inp.shape[0] ** .5)

  n_vars = u_gtr.shape[-1]
  if isinstance(symmetric, bool):
    symmetric = [symmetric] * n_vars

  # Create the figure and the gridspec
  figsize=(8.6, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN)
  fig = plt.figure(figsize=figsize)
  g_fig = fig.add_gridspec(
    nrows=n_vars,
    ncols=1,
    wspace=0,
    hspace=0,
  )

  figs = []
  for ivar in range(n_vars):
    figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
  # Add axes
  axs_inp = []
  axs_gtr = []
  axs_prd = []
  axs_err = []
  axs_cb_inp = []
  axs_cb_out = []
  axs_cb_err = []
  for ivar in range(n_vars):
    g = figs[ivar].add_gridspec(
      nrows=2,
      ncols=4,
      height_ratios=[1, .05],
      wspace=0.20,
      hspace=0.05,
    )
    axs_inp.append(figs[ivar].add_subplot(g[0, 0]))
    axs_gtr.append(figs[ivar].add_subplot(g[0, 1]))
    axs_prd.append(figs[ivar].add_subplot(g[0, 2]))
    axs_err.append(figs[ivar].add_subplot(g[0, 3]))
    axs_cb_inp.append(figs[ivar].add_subplot(g[1, 0]))
    axs_cb_out.append(figs[ivar].add_subplot(g[1, 1:3]))
    axs_cb_err.append(figs[ivar].add_subplot(g[1, 3]))
  # Settings
  for ax in [ax for axs in [axs_inp, axs_gtr, axs_prd, axs_err] for ax in axs]:
    ax: plt.Axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([domain[0][0], domain[1][0]])
    ax.set_ylim([domain[0][1], domain[1][1]])
    ax.fill_between(
      x=[domain[0][0], domain[1][0]], y1=domain[0][1], y2=domain[1][1],
      **HATCH_SETTINGS,
    )

  # Get prediction error
  u_err = (u_gtr - u_prd)

  # Loop over variables
  for ivar in range(n_vars):
    # Get ranges
    vmax_inp = np.max(u_inp[:, ivar])
    vmax_gtr = np.max(u_gtr[:, ivar])
    vmax_prd = np.max(u_prd[:, ivar])
    vmax_out = max(vmax_gtr, vmax_prd)
    vmin_inp = np.min(u_inp[:, ivar])
    vmin_gtr = np.min(u_gtr[:, ivar])
    vmin_prd = np.min(u_prd[:, ivar])
    vmin_out = min(vmin_gtr, vmin_prd)
    abs_vmax_inp = max(np.abs(vmax_inp), np.abs(vmin_inp))
    abs_vmax_out = max(np.abs(vmax_out), np.abs(vmin_out))

    # Plot input
    h = axs_inp[ivar].scatter(
      x=x_inp[:, 0],
      y=x_inp[:, 1],
      c=u_inp[:, ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(abs_vmax_inp if symmetric[ivar] else vmax_inp),
      vmin=(-abs_vmax_inp if symmetric[ivar] else vmin_inp),
      **_SCATTER_SETTINGS,
    )
    cb = plt.colorbar(h, cax=axs_cb_inp[ivar], orientation='horizontal')
    cb.formatter.set_powerlimits((0, 0))
    # Plot ground truth
    h = axs_gtr[ivar].scatter(
      x=x_out[:, 0],
      y=x_out[:, 1],
      c=u_gtr[:, ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
      vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
      **_SCATTER_SETTINGS,
    )
    # Plot estimate
    h = axs_prd[ivar].scatter(
      x=x_out[:, 0],
      y=x_out[:, 1],
      c=u_prd[:, ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
      vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
      **_SCATTER_SETTINGS,
    )
    cb = plt.colorbar(h, cax=axs_cb_out[ivar], orientation='horizontal')
    cb.formatter.set_powerlimits((0, 0))
    # Plot error
    h = axs_err[ivar].scatter(
      x=x_out[:, 0],
      y=x_out[:, 1],
      c=np.abs(u_err[:, ivar]),
      cmap=CMAP_WRB,
      vmin=0,
      vmax=np.max(np.abs(u_err[:, ivar])),
      **_SCATTER_SETTINGS,
    )
    cb = plt.colorbar(h, cax=axs_cb_err[ivar], orientation='horizontal')
    cb.formatter.set_powerlimits((0, 0))

  # Set titles
  axs_inp[0].set(title='Input');
  axs_gtr[0].set(title='Ground-truth');
  axs_prd[0].set(title='Model estimate');
  axs_err[0].set(title='Absolute error');

  # Set variable names
  for ivar in range(n_vars):
    label = names[ivar] if names else f'Variable {ivar:02d}'
    axs_inp[ivar].set(ylabel=label);

  # Rotate colorbar tick labels
  for ax in [ax for axs in [axs_cb_inp, axs_cb_out, axs_cb_err] for ax in axs]:
    ax: plt.Axes
    ax.xaxis.get_offset_text().set(size=8)
    ax.xaxis.set_tick_params(labelsize=8)

  return fig

def plot_ensemble(u_gtr, u_ens, x, idx_out: int, idx_s: int = 0, symmetric=True, names=None, domain=([0, 0], [1, 1])):
  """Plot the statistics of an ensemble of model estimates."""

  _HEIGHT_PER_ROW = 2.5
  _HEIGHT_MARGIN = .2
  _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
  _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .6 * _HEIGHT_PER_ROW
  _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (x.shape[0] ** .5)

  n_vars = u_gtr.shape[-1]
  if isinstance(symmetric, bool):
    symmetric = [symmetric] * n_vars

  # Create the figure and the gridspec
  figsize=(8.6, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN)
  fig = plt.figure(figsize=figsize)
  g_fig = fig.add_gridspec(
    nrows=n_vars,
    ncols=1,
    wspace=0,
    hspace=0,
  )

  figs = []
  for ivar in range(n_vars):
    figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
  # Add axes
  axs_avg = []
  axs_err = []
  axs_std = []
  axs_cb_avg = []
  axs_cb_err = []
  axs_cb_std = []
  for ivar in range(n_vars):
    g = figs[ivar].add_gridspec(
      nrows=2,
      ncols=3,
      height_ratios=[1, .05],
      wspace=0.20,
      hspace=0.05,
    )
    axs_avg.append(figs[ivar].add_subplot(g[0, 0]))
    axs_err.append(figs[ivar].add_subplot(g[0, 1]))
    axs_std.append(figs[ivar].add_subplot(g[0, 2]))
    axs_cb_avg.append(figs[ivar].add_subplot(g[1, 0]))
    axs_cb_err.append(figs[ivar].add_subplot(g[1, 1]))
    axs_cb_std.append(figs[ivar].add_subplot(g[1, 2]))
  # Settings
  for ax in [ax for axs in [axs_avg, axs_err, axs_std] for ax in axs]:
    ax: plt.Axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([domain[0][0], domain[1][0]])
    ax.set_ylim([domain[0][1], domain[1][1]])
    ax.fill_between(
      x=[domain[0][0], domain[1][0]], y1=domain[0][1], y2=domain[1][1],
      **HATCH_SETTINGS,
    )

  # Compute statistics and error
  u_ens_avg = np.mean(u_ens, axis=0)
  u_ens_std = np.std(u_ens, axis=0)
  u_err = (u_gtr - u_ens_avg)

  # Loop over variables
  for ivar in range(n_vars):
    vmax_gtr = np.max(np.abs(u_gtr[idx_s, idx_out, :, ivar]))
    # Plot mean
    h = axs_avg[ivar].scatter(
      x=x[:, 0],
      y=x[:, 1],
      c=u_ens_avg[idx_s, idx_out, :, ivar],
      cmap=(CMAP_BWR if symmetric[ivar] else CMAP_WRB),
      vmax=(vmax_gtr if symmetric[ivar] else None),
      vmin=(-vmax_gtr if symmetric[ivar] else None),
      **_SCATTER_SETTINGS,
    )
    cb = plt.colorbar(h, cax=axs_cb_avg[ivar], orientation='horizontal')
    cb.formatter.set_powerlimits((0, 0))
    # Plot error
    h = axs_err[ivar].scatter(
      x=x[:, 0],
      y=x[:, 1],
      c=np.abs(u_err[idx_s, idx_out, :, ivar]),
      cmap=CMAP_WRB,
      vmin=0,
      vmax=None,
      **_SCATTER_SETTINGS,
    )
    cb = plt.colorbar(h, cax=axs_cb_err[ivar], orientation='horizontal')
    cb.formatter.set_powerlimits((0, 0))
    # Plot std
    h = axs_std[ivar].scatter(
      x=x[:, 0],
      y=x[:, 1],
      c=u_ens_std[idx_s, idx_out, :, ivar],
      cmap=CMAP_WRB,
      vmin=0,
      vmax=None,
      **_SCATTER_SETTINGS,
    )
    cb = plt.colorbar(h, cax=axs_cb_std[ivar], orientation='horizontal')
    cb.formatter.set_powerlimits((0, 0))

  # Set titles
  axs_avg[0].set(title='Ensemble mean');
  axs_err[0].set(title='Absolute error');
  axs_std[0].set(title='Ensemble std');

  # Set variable names
  for ivar in range(n_vars):
    label = names[ivar] if names else f'Variable {ivar:02d}'
    axs_avg[ivar].set(ylabel=label);

  # Rotate colorbar tick labels
  for ax in [ax for axs in [axs_cb_avg, axs_cb_err, axs_cb_std] for ax in axs]:
    ax: plt.Axes
    ax.xaxis.get_offset_text().set(size=8)
    ax.xaxis.set_tick_params(labelsize=8)

  return fig

def plot_error_vs_time(df: pd.DataFrame, idx_fn: int, variable_title: str = 'variable', palette: str = None) -> sns.FacetGrid:
  g = sns.FacetGrid(
    data=(df[(df['error'] > 0.)]),
    hue='variable',
    palette=palette,
    height=4,
    aspect=.8,
  );
  g.set_titles(col_template='{col_name}');
  g.map(sns.scatterplot, 't', 'error', marker='o', s=30, alpha=1);
  g.map(sns.lineplot, 't', 'error', linewidth=2, alpha=.8);
  g.add_legend(title=variable_title);
  g.set_ylabels(label='Error (%)');

  sns.move_legend(g, loc='right', bbox_to_anchor=(1.02, .5))

  for ax in g.axes.flatten():
    ax.axvline(x=idx_fn, linestyle='--', color='black', linewidth=2, alpha=.5);
    ax.set_xticks([t for t in df['t'].unique() if (t%2 == 0)], minor=False);
    ax.set_xticks([t for t in df['t'].unique()], minor=True);
    ax.grid(which='major');

  return g
