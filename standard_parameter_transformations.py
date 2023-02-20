from . import parameter_aliasing as aliasing

def complete_par_dic(par_dic):
    lnlkey = aliasing.get_key(par_dic, 'lnl')
    priorkey = aliasing.get_key(par_dic, 'lnPrior')
    postkey = aliasing.get_key(par_dic, 'lnPosterior')
    if lnlkey in par_dic:
        if (priorkey in par_dic) and (postkey not in par_dic):
            par_dic[postkey] = par_dic[lnlkey] + par_dic[priorkey]
        elif (priorkey not in par_dic) and (postkey in par_dic):
            par_dic[priorkey] = par_dic[postkey] - par_dic[lnlkey]
    elif (priorkey in par_dic) and (postkey in par_dic):
        par_dic[lnlkey] = par_dic[postkey] - par_dic[priorkey]