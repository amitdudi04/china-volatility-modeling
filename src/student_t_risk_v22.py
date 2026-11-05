"""Student-t GARCH VaR/ES and VaR hit-dependence diagnostics (v2.2).

Requires per-origin fitted GARCH parameters. Does not infer Student-t df
from a fixed arbitrary constant or from a full-sample fit.

The reported ES is a PARAMETRIC MODEL FORECAST; realized ES backtesting
requires a separate jointly consistent evaluation and is not claimed here.
"""
from __future__ import annotations
import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2,t as student_t
from src.risk_economic_v22 import kupiec_unconditional_coverage

def standardized_t_var_es(mean_daily, sigma_daily, nu, alpha=.05):
    """Return daily left-tail loss cutoffs in signed log-return units.

    arch Student-t innovations are standardized to conditional variance 1,
    while scipy.stats.t is scaled with variance nu/(nu-2).
    """
    if not (np.isfinite(mean_daily) and np.isfinite(sigma_daily)
            and np.isfinite(nu) and sigma_daily>0 and nu>2 and 0<alpha<.5):
        raise ValueError("Invalid Student-t distribution parameters")
    z=float(student_t.ppf(alpha,nu))
    scale=float(np.sqrt((nu-2)/nu))
    cutoff=float(mean_daily+sigma_daily*scale*z)
    # E[t_nu | t_nu <= z] = -pdf(z) * (nu+z^2)/(alpha*(nu-1)).
    tail=float(mean_daily-sigma_daily*scale*
               ((nu+z*z)*student_t.pdf(z,nu))/(alpha*(nu-1)))
    if not (np.isfinite(tail) and tail<cutoff<mean_daily):
        raise ValueError("Invalid Student-t tail conditional mean")
    return cutoff,tail

def christoffersen_tests(breaches,alpha=.05)->dict:
    """Independence LR(1), conditional-coverage LR(2) from binary VaR hits."""
    hits=np.asarray(breaches,dtype=bool)
    if hits.ndim!=1 or len(hits)<3:
        raise ValueError("At least three observed hits are necessary")
    n00=int(np.sum(~hits[:-1]&~hits[1:]))
    n01=int(np.sum(~hits[:-1]&hits[1:]))
    n10=int(np.sum(hits[:-1]&~hits[1:]))
    n11=int(np.sum(hits[:-1]&hits[1:]))
    n0=n00+n01;n1=n10+n11
    pi0=(n01/n0) if n0 else 0.
    pi1=(n11/n1) if n1 else 0.
    pi=(n01+n11)/(len(hits)-1)
    null=float(xlogy(n01+n11,pi)+xlogy(n00+n10,1-pi))
    alternate=float(
        xlogy(n01,pi0)+xlogy(n00,1-pi0)
        +xlogy(n11,pi1)+xlogy(n10,1-pi1))
    lr_ind=max(0.,2*(alternate-null))
    unconditional=kupiec_unconditional_coverage(hits,alpha=alpha)
    lr_cc=lr_ind+unconditional["kupiec_lr"]
    return {
        **unconditional,"n00":n00,"n01":n01,"n10":n10,"n11":n11,
        "christoffersen_ind_lr":float(lr_ind),
        "christoffersen_ind_pvalue":float(chi2.sf(lr_ind,1)),
        "christoffersen_cc_lr":float(lr_cc),
        "christoffersen_cc_pvalue":float(chi2.sf(lr_cc,2)),
    }
