import os, h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy as dcopy
from matplotlib.colors import Normalize as colorsNormalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

from . import grid as gd
from . import parameter_aliasing as aliasing
from . import parameter_label_formatting as label_formatting

PAR_LABELS = label_formatting.param_labels
PAR_UNITS = label_formatting.units
PARKEY_MAP = aliasing.PARKEY_MAP

DICTLIKE_TYPES = [dict, type(pd.DataFrame([{0: 0}])), type(pd.Series({0: 0}))]
def is_dict(check):
    return any([isinstance(check, dtype) for dtype in DICTLIKE_TYPES])

def label_from_key(key):
    return PAR_LABELS.get(aliasing.PARKEY_MAP.get(key, key), key)

def printarr(arr, prec=4, pre='', post='', sep='  ', form='f'):
    print(pre + np.array2string(np.asarray(arr), separator=sep,
                                max_line_width=np.inf, threshold=np.inf,
                                formatter={'float_kind':lambda x: f"%.{prec}{form}" % x}) + post)

def fmt(num, prec=4, form='f'):
    formstr = '{:.' + str(prec) + form + '}'
    return formstr.format(num)

########################################
#### CORNER PLOTTING FUNCTIONS
def corner_plot_samples(samps, pvkeys=['lnl', 'lnPrior'], title=None,
                        figsize=(9,7), scatter_points=None, weights=None,
                        grid_kws={}, fig=None, ax=None, return_grid=False, **corner_plot_kws):
    """make corner plots"""
    units, plabs = PAR_UNITS, PAR_LABELS
    for k in pvkeys:
        if k not in units.keys():
            units[k] = ''
        if k not in plabs.keys():
            plabs[k] = k
    sg = gd.Grid.from_samples(pvkeys, samps, weights=weights, pdf_key=None,
                              units=units, labels=plabs, **grid_kws)
    ff, aa = sg.corner_plot(pdf=None, title=title, figsize=figsize, set_legend=True,
                            scatter_points=scatter_points, fig=fig, ax=ax, **corner_plot_kws)
    if return_grid:
        return ff, aa, sg
    return ff, aa

def corner_plot_list(samps_list, samps_names, pvkeys=['lnl', 'lnPrior'], weight_key=None,
                     fig=None, ax=None, figsize=(9,7), scatter_points=None, grid_kws={},
                     multigrid_kws={}, return_grid=False, **corner_plot_kws):
    grids = []
    units, plabs = PAR_UNITS, PAR_LABELS
    for k in pvkeys:
        if k not in units.keys():
            units[k] = ''
        if k not in plabs.keys():
            plabs[k] = k
    for p, nm in zip(samps_list, samps_names):
        grids.append(gd.Grid.from_samples(pvkeys, p, units=units, labels=plabs, pdf_key=nm,
                                          weights=(None if weight_key is None
                                                   else p.samples[weight_key]), **grid_kws))
    multigrid = gd.MultiGrid(grids, **multigrid_kws)
    ff, aa = multigrid.corner_plot(set_legend=True, figsize=figsize, scatter_points=scatter_points,
                                   fig=fig, ax=ax, **corner_plot_kws)
    if return_grid:
        return ff, aa, multigrid
    return ff, aa

def get_dets_figure(detector_names, xlabel='Frequency (Hz)', ylabel='Amplitude', figsize=None):
    fig, ax = plt.subplots(len(detector_names), sharex=True, figsize=figsize)
    fig.text(.004, .54, ylabel, rotation=90, ha='left', va='center', size=10)
    ax[0].set_xlabel(xlabel)
    for a, det in zip(ax, detector_names):
        a.text(.02, .95, det, ha='left', va='top', transform=a.transAxes)
        a.tick_params(which='both', direction='in', right=True, top=True)
    return fig, ax

def plot_at_dets(xplot, dets_yplot, ax=None, fig=None, label=None,
                 xlabel='Frequency (Hz)', ylabel='Amplitude',
                 plot_type='loglog', xlim=None, ylim=None, title=None,
                 det_names=['D1', 'D2'], figsize=None, **plot_kws):
    if ax is None:
        fig, ax = get_dets_figure(det_names, xlabel=xlabel, ylabel=ylabel, figsize=figsize)
    if np.ndim(xplot) == 1:
        xplot = [xplot]*len(det_names)
    mask = slice(None)
    for j, a in enumerate(ax):
        if xlim is not None:
            mask = (xplot[j] >= xlim[0]) & (xplot[j] <= xlim[1])
        plotfunc = (a.loglog if plot_type in ['loglog', 'log'] else 
                    (a.semilogx if plot_type in ['semilogx', 'logx', 'xlog'] else 
                     (a.semilogy if plot_type in ['semilogy', 'logy', 'ylog'] else a.plot)))
        plotfunc(xplot[j][mask], dets_yplot[j][mask], label=label, **plot_kws)
        if label is not None:
            a.legend(title=det_names[j])
        a.set_ylim(ylim)
    if title is not None:
        plt.suptitle(title)
    return fig, ax
    
colorbar_kws_DEFAULTS = {'pad': 0.02, 'fraction': 0.1, 'aspect': 24, 'shrink': 0.5,
                         'ticks': 8, 'format': '%.2f'}
plot3d_kws_DEFAULTS = {'alpha': 0.1, 's': 0.05}

def scatter3d(x, y, z, cs=None, xlab='$\\alpha$', ylab='$\\beta$', zlab='$\\delta$',
              clab='ln$\\mathcal{L}$', colorsMap='jet', fig=None, ax=None,
              title=None, titlesize=20, figsize=(14, 14),
              xlim='auto', ylim='auto', zlim='auto', plot_kws=None, colorbar_kws=None):
    if cs is None:
        cs = np.ones(len(x))
    cm = plt.get_cmap(colorsMap)
    cNorm = colorsNormalize(vmin=min(cs), vmax=max(cs))
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    if (fig is None) or (ax is None):
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
    
    plot_kwargs = dcopy((plot3d_kws_DEFAULTS if plot_kws is None else plot_kws))
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), **plot_kwargs)
    scalarMap.set_array(cs)
    cbar_kws = dcopy((colorbar_kws_DEFAULTS if colorbar_kws is None else colorbar_kws))
    if isinstance(cbar_kws.get('ticks'), int):
        cbar_kws['ticks'] = np.linspace(np.min(cs), np.max(cs), cbar_kws['ticks'], endpoint=True)
    cbar = fig.colorbar(scalarMap, **cbar_kws)
    cbar.set_label(clab)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)
    if xlim == 'auto':
        xlim = (np.min(x), np.max(x))
    ax.set_xlim(xlim)
    if ylim == 'auto':
        ylim = (np.min(y), np.max(y))
    ax.set_ylim(ylim)
    if zlim == 'auto':
        zlim = (np.min(z), np.max(z))
    ax.set_zlim(zlim)
    if plot_kwargs.get('label') is not None:
        ax.legend(title=plot_kwargs.get('legend_title'))
    if title is not None:
        ax.set_title(title, size=titlesize)
    return fig, ax

def get_spin_plot_par(samples, key):
    if isinstance(key, list):
        return [get_spin_plot_par(samples, k) for k in key]
    elif len(key) == 3:
        return samples[key]
    else:
        if 'prime' in key:
            j = int(key[1])
            trigfunc = (np.cos if 'x' in key else np.sin)
            r = samples[f'cums{j}r_s{j}z']**.5
            if 'rescale' in key:
                return r * trigfunc(samples[f's{j}phi_hat'])
            else:
                return r * trigfunc(samples[f's{j}phi_hat']) * np.sqrt(1 - samples[f's{j}z']**2)
        else:
            assert 'sign' in key, "key must be 'sjx'+('' or '_newsign' or '_prime_rescale' or 'prime')"
            signcosiota = np.sign((samples['cosiota'] if 'cosiota' in samples else np.cos(samples['iota'])))
            return samples[key[:3]] * signcosiota

def plot_inplane_spin(pe_samples, color_key='q', use_V3=False, secondary_spin=False,
                      fractions=[.5, .95], plotstyle_color='r', scatter_alpha=.5,
                      figsize=None, title=None, tight=False, colorsMap='jet',
                      scatter_size=.8, scatter_nstep=1, get_contour=False, **colorbar_kws):
    plotstyle_2d = gd.PlotStyle2d(plotstyle_color, fractions=fractions,
                                  show_cl=True, clabel_fs=11)
    j = (2 if secondary_spin else 1)
    plotkeys = [f's{(2 if secondary_spin else 1)}{dct}_' +
                ('prime_rescale' if use_V3 else 'newsign')
                for dct in ['x', 'y']]
    plot_samples = pd.DataFrame({k: get_spin_plot_par(pe_samples, k) for k in plotkeys})
    x = np.linspace(0, 2*np.pi, 300)
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(color_key, str):
        plt.scatter(plot_samples[plotkeys[0]].to_numpy()[::scatter_nstep],
                    plot_samples[plotkeys[1]].to_numpy()[::scatter_nstep],
                    s=scatter_size, lw=0, c=pe_samples[color_key].to_numpy()[::scatter_nstep],
                    alpha=scatter_alpha, cmap=colorsMap)
        colorbar_kws['label'] = colorbar_kws.get('label', label_from_key(color_key))
        plt.colorbar(**colorbar_kws)
    plt.plot(np.cos(x), np.sin(x), lw=1, c='k')
    # Make grid
    g = gd.Grid.from_samples(plotkeys, plot_samples)
    # Make 2d plot
    contourout = g.grids_2d[plotkeys[0], plotkeys[1]].plot_pdf(
        'posterior', ax, style=plotstyle_2d, get_contour=get_contour)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.grid()
    ax.set_aspect('equal')
    plt.xlabel(label_from_key(plotkeys[0]))
    plt.ylabel(label_from_key(plotkeys[1]))
    if tight:
        plt.tight_layout()
    ax.set_title(title)
    if get_contour:
        return fig, ax, contourout
    return fig, ax

def plot_spin4d(samples, ckey='lnl', xkey='sx', ykey='sy', zkey='sz',
                nstep=1, title=None, xlab='auto', ylab='auto',
                zlab='auto', clab='auto', plotlim=[-1.1, 1.1],
                mask_keys_min={}, mask_keys_max={}, fig=None, ax=None, plot_kws=None,
                figsize=(14, 14), titlesize=20, colorbar_kws=None, colorsMap='jet',
                extra_point_dicts=[(0, 0, 0)], marker_if_not_dict='o',
                size_if_not_dict=20, color_if_not_dict='k'):
    """scatter3d but using a dataframe and keys instead of using x/y/z/color arrays directly"""

    x, y, z = [np.asarray(get_spin_plot_par(samples, k)) for k in [xkey, ykey, zkey]]
    # labels
    if (xlab is None) or (xlab == 'auto'):
        xlab = label_from_key(xkey)
    if (ylab is None) or (ylab == 'auto'):
        ylab = label_from_key(ykey)
    if (zlab is None) or (zlab == 'auto'):
        zlab = label_from_key(zkey)
    if (clab is None) or (clab == 'auto'):
        clab = label_from_key(ckey)
    # get colorbar array and mask based on mask_keys_min/max
    clr = np.asarray(samples[ckey])
    mask = np.ones(len(clr), dtype=bool)
    for k, v in mask_keys_min.items():
        mask *= samples[k].to_numpy() >= v
    for k, v in mask_keys_max.items():
        mask *= samples[k].to_numpy() <= v
    # plot using peplot.scatter3d
    if title == 'flat':
        title = 'Posterior Samples from PE with Flat $\\chi_{eff}$ Prior'
    elif title == 'iso':
        title = 'Posterior Samples from PE with Isotropic $\\vec{\\chi}_1, \\vec{\\chi}_2$ Priors'

    fig, ax = scatter3d(x[mask][::nstep], y[mask][::nstep], z[mask][::nstep], clr[mask][::nstep], title=title,
                      xlab=xlab, ylab=ylab, zlab=zlab, clab=clab, xlim=plotlim, ylim=plotlim, zlim=plotlim,
                      titlesize=titlesize, figsize=figsize, plot_kws=plot_kws, colorbar_kws=colorbar_kws,
                      fig=fig, ax=ax, colorsMap=colorsMap)
    # plot extra points
    for dic in extra_point_dicts:
        if is_dict(dic):
            xx, yy, zz = dic[xkey], dic[ykey], dic[zkey]
            ax.scatter(xx, yy, zz, marker=dic.get('marker', dic.get('m')), s=dic.get('size', dic.get('s')),
                       c=dic.get('color', dic.get('c')))
            if dic.get('text', None) is not None:
                ax.text(xx - 0.15, yy - 0.15, zz - 0.48, dic['text'], color=dic.get('textcolor'),
                        size=dic.get('textsize'))
        else:
            ax.scatter(dic[0], dic[1], dic[2], marker=marker_if_not_dict,
                       s=size_if_not_dict, c=color_if_not_dict)
    circang = np.linspace(0, 2*np.pi, 360)
    ax.plot(np.cos(circang), np.sin(circang), np.zeros(360), lw=2, c='k')
    if 'z' in zkey:
        ax.scatter(0, 0, 1.01, marker='^', c='k', s=25)
        ax.scatter(0, 0, -1, marker='s', c='k', s=20)
        ax.plot(np.zeros(100), np.zeros(100), np.linspace(-1, 1.02, 100), lw=1, c='k')
        ax.plot(np.cos(circang), np.zeros(360), np.sin(circang), lw=1, ls=':', c='k')
        ax.plot(np.zeros(360), np.cos(circang), np.sin(circang), lw=1, ls=':', c='k')
    print(f'Plotted {np.count_nonzero(mask) // nstep} of {len(samples)} samples')
    return fig, ax

########################################
#### GEOMETRIC CONVERSION

def xyz_from_rthetaphi(r, theta, phi):
    """DL in Mpc, ra in [0, 2pi], dec in [-pi/2, pi/2]"""
    return r * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

def rthetaphi_from_xyz(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return r, np.arccos(z / r), np.arctan2(y, x) % (2*np.pi)

#########################################
#### 4-DIMENSIONAL SAMPLE PLOTTING
def plot_loc3d(samples, title=None, xlim='auto', ylim='auto', zlim='auto', nstep=2,
               ckey='lnl', clab=None, mask_keys_min={}, mask_keys_max={},
               plot_kws=None, figsize=(14, 14), titlesize=20, colorbar_kws=None,
               units='km', extra_point_dicts=[], fig=None, ax=None, colorsMap='jet'):
    rkey = 'r'
    if rkey not in samples:
        rkey = PARKEY_MAP.get(rkey, rkey)
    x, y, z = xyz_from_rthetaphi(samples[rkey].to_numpy(), samples['theta'].to_numpy(),
                                 samples['phi'].to_numpy())
    clr = samples[ckey].to_numpy()
    Ns = len(clr)
    mask = np.ones(Ns, dtype=bool)
    for k, v in mask_keys_min.items():
        mask *= samples[k].to_numpy() >= v
    for k, v in mask_keys_max.items():
        mask *= samples[k].to_numpy() <= v

    if (clab is None) or (clab == 'auto'):
        clab = label_from_key(ckey)
    fig, ax = scatter3d(x[mask][::nstep], y[mask][::nstep], z[mask][::nstep], clr[mask][::nstep],
                        title=title, xlab=f'X ({units})', ylab=f'Y ({units})', zlab=f'Z ({units})',
                        clab=clab, xlim=xlim, ylim=ylim, zlim=zlim, titlesize=titlesize,
                        figsize=figsize, plot_kws=plot_kws, colorbar_kws=colorbar_kws,
                        fig=fig, ax=ax, colorsMap=colorsMap)
    # plot origin
    ax.scatter(0, 0, 0, marker='*', s=24, c='k')
    ax.text(-0.15, -0.15, -0.4, 'O', color='k', size=14)
    for dic in extra_point_dicts:
        xx, yy, zz = xyz_from_rthetaphi(dic[rkey], dic['theta'], dic['phi'])
        ax.scatter(xx, yy, zz, marker=dic.get('marker', dic.get('m')), s=dic.get('size', dic.get('s')),
                   c=dic.get('color', dic.get('c')))
        if dic.get('text', None) is not None:
            ax.text(xx - 0.15, yy - 0.15, zz - 0.48, dic['text'], color=dic.get('textcolor'),
                    size=dic.get('textsize'))
    print(f'Plotted {int(np.floor(np.count_nonzero(mask) / nstep))} of {Ns} samples')
    return fig, ax


def plot_samples4d(samples, xkey='x', ykey='y', zkey='z', ckey='lnL',
                   xlim='auto', ylim='auto', zlim='auto', nstep=2, title=None,
                   xlab='auto', ylab='auto', zlab='auto', clab='auto',
                   mask_keys_min={}, mask_keys_max={}, fig=None, ax=None,
                   plot_kws=None, figsize=(14, 14), titlesize=20, colorbar_kws=None,
                   extra_point_dicts=[], size_key=None, size_scale=1, colorsMap='jet'):
    """scatter3d but using a dataframe and keys instead of using x/y/z/color arrays directly"""
    if (xlab is None) or (xlab == 'auto'):
        xlab = label_from_key(xkey)
    if (ylab is None) or (ylab == 'auto'):
        ylab = label_from_key(ykey)
    if (zlab is None) or (zlab == 'auto'):
        zlab = label_from_key(zkey)
    if (clab is None) or (clab == 'auto'):
        clab = label_from_key(ckey)
    x, y, z, clr = [np.asarray(samples[key]) for key in [xkey, ykey, zkey, ckey]]
    Ns = len(clr)
    mask = np.ones(Ns, dtype=bool)
    for k, v in mask_keys_min.items():
        mask *= samples[k].to_numpy() >= v
    for k, v in mask_keys_max.items():
        mask *= samples[k].to_numpy() <= v
    
    if isinstance(size_key, str):
        size_arr = np.asarray(samples[size_key])[mask]
        plot_kws['s'] = size_arr[::nstep] * size_scale / np.max(size_arr)

    fig, ax = scatter3d(x[mask][::nstep], y[mask][::nstep], z[mask][::nstep], clr[mask][::nstep],
                        title=title, xlab=xlab, ylab=ylab, zlab=zlab, clab=clab, xlim=xlim,
                        ylim=ylim, zlim=zlim, titlesize=titlesize, figsize=figsize,
                        plot_kws=plot_kws, colorbar_kws=colorbar_kws, fig=fig, ax=ax,
                        colorsMap=colorsMap)
    # plot extra points
    for dic in extra_point_dicts:
        xx, yy, zz = dic[xkey], dic[ykey], dic[zkey]
        ax.scatter(xx, yy, zz, marker=dic.get('marker', dic.get('m')),
                   s=dic.get('size', dic.get('s')), c=dic.get('color', dic.get('c')))
        if dic.get('text', None) is not None:
            ax.text(xx - 0.15, yy - 0.15, zz - 0.48, dic['text'], color=dic.get('textcolor'),
                    size=dic.get('textsize'))
    print(f'Plotted {np.count_nonzero(mask)} of {Ns} samples')
    return fig, ax

def scatter2d_color(x, y, cs=None, xlab='$\\alpha$', ylab='$\\delta$',
                    clab='ln$\\mathcal{L}$', colorsMap='jet', fig=None, ax=None,
                    title=None, titlesize=20, figsize=(14, 14),
                    xlim='auto', ylim='auto', plot_kws=None, colorbar_kws=None):
    """make 2d scatter plot with colorbar for visualizing third dimension"""
    if cs is None:
        cs = np.ones(len(x))
    cm = plt.get_cmap(colorsMap)
    cNorm = colorsNormalize(vmin=min(cs), vmax=max(cs))
    scalarMap = ScalarMappable(norm=cNorm, cmap=cm)
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize=figsize)
    
    plot_kwargs = ({} if plot_kws is None else dcopy(plot_kws))
    ax.scatter(x, y, c=scalarMap.to_rgba(cs), **plot_kwargs)
    scalarMap.set_array(cs)
    cbar_kws = dcopy((colorbar_kws_DEFAULTS if colorbar_kws is None else colorbar_kws))
    if isinstance(cbar_kws.get('ticks'), int):
        cbar_kws['ticks'] = np.linspace(np.min(cs), np.max(cs), cbar_kws['ticks'], endpoint=True)
    cbar = fig.colorbar(scalarMap, **cbar_kws)
    cbar.set_label(clab)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if xlim == 'auto':
        xlim = (np.min(x), np.max(x))
    if ylim == 'auto':
        ylim = (np.min(y), np.max(y))
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    if plot_kwargs.get('label') is not None:
        ax.legend(title=plot_kwargs.get('legend_title'))
    if title is not None:
        ax.set_title(title, size=titlesize)
    return fig, ax

def plot_samples2d_color(samples_list, xkey='x', ykey='y', ckey='lnL', samples_per_posterior=None,
                         fig=None, ax=None, colorbar_kws=None, colorsMap='jet', figsize=(14, 14),
                         title=None, titlesize=20, xlim='auto', ylim='auto', clim=None,
                         size_key=None, size_scale=1, alpha_key=None, alpha_scale=1, **plot_kws):
    """2d scatter plotting with color key, calls scatter2d_color()"""
    plot_list = (samples_list if isinstance(samples_list, list) else [samples_list])
    if samples_per_posterior is None:
        xarr = np.concatenate([np.asarray(s[xkey]) for s in plot_list])
        yarr = np.concatenate([np.asarray(s[ykey]) for s in plot_list])
        carr = np.concatenate([np.asarray(s[ckey]) for s in plot_list])
        if isinstance(size_key, str):
            size_arr = np.concatenate([np.asarray(s[size_key]) for s in plot_list])
            size_arr *= size_scale / np.max(size_arr)
            plot_kws['s'] = size_arr.copy()
        if isinstance(alpha_key, str):
            alpha_arr = np.concatenate([np.asarray(s[alpha_key]) for s in plot_list])
            alpha_arr *= alpha_scale / np.max(alpha_arr)
            plot_kws['alpha'] = alpha_arr.copy()
    else:
        randinds = [np.random.choice(np.arange(len(s)), size=samples_per_posterior)
                    for s in plot_list]
        xarr = np.concatenate([np.asarray(s[xkey])[rinds]
                               for s, rinds in zip(plot_list, randinds)])
        yarr = np.concatenate([np.asarray(s[ykey])[rinds]
                               for s, rinds in zip(plot_list, randinds)])
        carr = np.concatenate([np.asarray(s[ckey])[rinds]
                               for s, rinds in zip(plot_list, randinds)])
        if isinstance(size_key, str):
            size_arr = np.concatenate([np.asarray(s[size_key])[rinds]
                                       for s, rinds in zip(plot_list, randinds)])
            size_arr *= size_scale / np.max(size_arr)
            plot_kws['s'] = size_arr.copy()
        if isinstance(alpha_key, str):
            alpha_arr = np.concatenate([np.asarray(s[alpha_key])[rinds]
                                       for s, rinds in zip(plot_list, randinds)])
            alpha_arr *= alpha_scale / np.max(alpha_arr)
            plot_kws['alpha'] = alpha_arr.copy()
    if clim is not None:
        mask = (carr > clim[0]) & (carr < clim[1])
        xarr = xarr[mask]
        yarr = yarr[mask]
        carr = carr[mask]
        if isinstance(size_key, str):
            size_arr = size_arr[mask]
            plot_kws['s'] = size_arr.copy()
        if isinstance(alpha_key, str):
            alpha_arr = alpha_arr[mask]
            plot_kws['alpha'] = alpha_arr.copy()
    xlab = label_from_key(xkey)
    ylab = label_from_key(ykey)
    clab = label_from_key(ckey)
    return scatter2d_color(xarr, yarr, cs=carr, colorsMap=colorsMap, fig=fig, ax=ax,
              title=title, titlesize=titlesize, figsize=figsize,
              xlim=xlim, ylim=ylim, xlab=xlab, ylab=ylab,
              clab=clab, colorbar_kws=colorbar_kws, plot_kws=plot_kws)


def combined_hist(samples_list, xkey='x', ykey=None, samples_per_posterior=None,
                  keys_min_means={}, keys_max_means={}, keys_min_medians={}, keys_max_medians={},
                  bins=100, cmap='rainbow', title=None, figsize=(10, 10), xlim=None, ylim=None,
                  fig=None, ax=None, **hist_kwargs):
    """
    make 1D or 2D histogram of combined posteriors
    :param samples_list: list with each element a DataFrame of samples
    :param xkey: key for parameter to go on x-axis
    :param ykey: (optional) key for parameter to go on y-axis
    :param keys_min_means: dict w/ keys & values = parameter keys & minimum mean values of those
      parameters, determining whether or not a given posterior is included (default: empty dict)
    :param keys_max_means: dict w/ keys & values = parameter keys & maximum mean values of those
      parameters, determining whether or not a given posterior is included (default: empty dict)
    :param keys_min_medians: dict w/ keys & values = parameter keys & minimum median values of those
      parameters, determining whether or not a given posterior is included (default: empty dict)
    :param keys_max_medians: dict w/ keys & values = parameter keys & maximum median values of those
      parameters, determining whether or not a given posterior is included (default: empty dict)
    :param bins: bins argument passed to hist or hist2d
    :param cmap: cmap argument passed to hist2d
    :param samples_per_posterior: number of samples to randomly draw from each set of samples
      (default is to use all samples)
    **remaining kwargs for hist or hist2d are for figure and label formatting**
    """
    # remove posteriors with means and medians outside the range
    plot_list = samples_list
    for k, v in keys_min_means.items():
        plot_list = [p for p in plot_list if np.mean(p[k]) > v]
    for k, v in keys_max_means.items():
        plot_list = [p for p in plot_list if np.mean(p[k]) < v]
    for k, v in keys_min_medians.items():
        plot_list = [p for p in plot_list if np.median(p[k]) > v]
    for k, v in keys_max_medians.items():
        plot_list = [p for p in plot_list if np.median(p[k]) < v]

    if samples_per_posterior is None:
        xarr = np.concatenate([np.asarray(s[xkey]) for s in plot_list])
    else:
        xarr = np.concatenate([np.asarray(s[xkey])[
                    np.random.choice(np.arange(len(s)), size=samples_per_posterior)]
                               for s in plot_list])
    # now make histogram (2d if ykey given, else 1d)
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    if isinstance(ykey, str):
        if samples_per_posterior is None:
            yarr = np.concatenate([np.asarray(s[ykey]) for s in plot_list])
        else:
            yarr = np.concatenate([np.asarray(s[ykey])[np.random.choice(np.arange(len(s)), \
                                                                        size=samples_per_posterior)] \
                                   for s in plot_list])
        hist_out = ax.hist2d(xarr, yarr, bins=bins, cmap=cmap, **hist_kwargs)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(label_from_key(ykey))
    else:
        hist_out = ax.hist(xarr, bins=bins, **hist_kwargs)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel(label_from_key(xkey))
    if title is not None:
        ax.set_title(title)
    return fig, ax, hist_out