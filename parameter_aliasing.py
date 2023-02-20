from copy import deepcopy as dcopy
#     Parameter Aliases
# ----------------------------

ALL_PARKEYS = {
    'lnl': ['lnl', 'lnL', 'deltalogL', 'deltalogl', 'loglr', 'log_likelihood',
            'lnlike', 'loglike', 'loglikelihood'],
    'lnPrior': ['lnPrior', 'ln_prior', 'log_prior', 'lnprior', 'logprior'],
    'lnPosterior': ['lnPosterior', 'lnPost', 'log_posterior', 'log_post',
                    'lnL+lnPrior', 'lnPrior+lnL'],
    'prior': ['prior'],
    'snr': ['snr', 'SNR', 'signal_to_noise', 'signal-to-noise',
            'matched_filter_snr'],
    'snr2': ['snr2', 'SNR2', 'snr_squared', 'SNR_squared', 'snr_sq', 'SNR_sq',
             'snrsq', 'SNRsq', 'signal_to_noise_squared', 'signal-to-noise-squared'],
    'lnLmax': ['lnLmax', 'lnL_max', 'maximum_log_likelihood', 'max_log_likelihood',
               'maxl_loglr', 'max_lnL', 'max_lnlike', 'lnlike_max'],
    'lnLmarg': ['lnLmarg', 'lnL_marg', 'log_likelihood_marginalized'],
    'lnL_Lmax': ['lnL_Lmax', 'log_max_likelihood_ratio', 'delta_log_max_likelihood'],
    'approximant': ['approximant', 'approx'],
    'score': ['score'],
    'index': ['index'],
    'rank': ['rank'],
}

# map taking every element of ALL_PARKEYS[k] back to k
PARKEY_MAP = {}
for k, alt_keys in ALL_PARKEYS.items():
    PARKEY_MAP.update({k_alt: k for k_alt in alt_keys})

def get_key(dic, key, map_to_all_keys=PARKEY_MAP):
    if key in dic:
        return key
    return map_to_all_keys.get(key, None)

def get_from_dic(dic, key, alt_val=None, map_to_all_keys=PARKEY_MAP):
    return dic.get(get_key(dic, key, map_to_all_keys=map_to_all_keys), alt_val)