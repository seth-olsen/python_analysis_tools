from . import standard_parameter_transformations as pxform

# Latex formatting
# ----------------
param_labels = {
    'log10rate': r'$\log_{10} (R \rm Gpc^3yr)$',
    't': r'$t$',
    'lnl': r'$\Delta \ln \mathcal{L}$',
    'lnL': r'$\Delta \ln \mathcal{L}$',
    'lnPrior': r'$\Delta \ln \Pi$',
    'lnPosterior': r'$\Delta \ln \mathcal{L} + \Delta \ln \Pi$',
    'lnLmax': r'$max_{\vec{\theta}} \Delta \ln \mathcal{L}(\vec{\theta} | d)$',
    'lnL_Lmax': r'$\ln( \mathcal{L} / \mathcal{L}_{\rm{max}} )$',
    'snr': r'$\rho$',
    'snr2': r'$\rho^2$',
}

# units associated to each parameter
units = {'t': 's'}
for k in param_labels.keys():
    if units.get(k, None) is None:
        units[k] = ''

# names for each parameter that are recognized by labelling system
param_names = {
    'log10rate': 'Log Base 10 Rate',
    't': 'Time',
    'lnl': 'Log Likelihood',
    'lnL': 'Log Likelihood',
    'lnPrior': 'Log Prior',
    'lnPosterior': 'Log Posterior',
    'lnLmax': 'Maximized Log Likelihood',
    'snr': 'Signal-to-Noise Ratio',
    'snr2': 'Squared Signal-to-Noise Ratio',
}


def fmt_num(num, prec_override=None):
    if prec_override is not None:
        return ('{:.'+str(prec_override)+'f}').format(num)
    if abs(num) > 10000:
        return f'{num:.1e}'
    if abs(num) > 10:
        return f'{num:.0f}'
    if abs(num) > 1:
        return f'{num:.1f}'
    return f'{num:.2f}'

def label_from_pdic(pdic, keys=['lnl'], pre='', post='',
                    sep=', ', connector=' = ', prec_override=None,
                    add_units=False):
    pstr = ''
    get_sep = lambda k: sep
    if add_units:
        get_sep = lambda k: ' (' + units[k] + ')' + sep
    pdic_use = dict(pdic)
    if any([(pdic_use.get(k) is None) for k in keys]):
        pxform.complete_par_dic(pdic_use)
    get_num = lambda k: (r'$' + fmt_num(pdic_use.get(k), prec_override)
                         + r'$' + get_sep(k))
    for k in keys:
        pstr += param_labels.get(k, k) + connector + get_num(k)
    return pre + pstr[:-len(sep)] + post


