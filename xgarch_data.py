import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, Normal, StudentsT
from scipy import optimize

K_PI = math.pi
HUGE_PENALTY = 1e300


def sigmoid(x: float) -> float:
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def logit(p: float) -> float:
    if p <= 0:
        return -HUGE_PENALTY
    if p >= 1:
        return HUGE_PENALTY
    return math.log(p / (1.0 - p))


def theta_to_dof(theta_val: float) -> float:
    return 2.0 + math.exp(theta_val)


def theta_to_nagarch_params(theta: Sequence[float]) -> Dict[str, float]:
    mu = theta[0]
    omega = math.exp(theta[1])
    u = sigmoid(theta[2])
    v = sigmoid(theta[3])
    gamma = theta[4]
    beta = u * v
    alpha = u * (1.0 - v) / (1.0 + gamma * gamma)
    return {"mu": mu, "omega": omega, "alpha": alpha, "beta": beta, "gamma": gamma}


def theta_to_igarch_params(theta: Sequence[float]) -> Dict[str, float]:
    mu = theta[0]
    alpha_raw = sigmoid(theta[1])
    eps = 1e-6
    alpha = eps + (1.0 - 2.0 * eps) * alpha_raw
    beta = 1.0 - alpha
    return {"mu": mu, "alpha": alpha, "beta": beta}


def theta_to_st_params(theta: Sequence[float]) -> Dict[str, float]:
    mu = theta[0]
    omega = math.exp(theta[1])
    u = sigmoid(theta[2])
    v = sigmoid(theta[3])
    w = sigmoid(theta[4])
    remainder = u * (1.0 - v)
    beta = u * v
    gamma = 2.0 * remainder * w
    alpha = remainder * (1.0 - w)
    shift = theta[5]
    return {
        "mu": mu,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "shift": shift,
    }


def log_t_pdf(x: float, dof: float) -> float:
    a = math.lgamma(0.5 * (dof + 1.0)) - math.lgamma(0.5 * dof)
    b = -0.5 * (math.log(dof) + math.log(K_PI))
    c = -0.5 * (dof + 1.0) * math.log(1.0 + (x * x) / dof)
    return a + b + c


def standardized_t_scale(dof: float) -> float:
    return math.sqrt(dof / (dof - 2.0))


def initial_variance(r: np.ndarray, mu: float) -> float:
    var = float(np.mean((r - mu) ** 2))
    if not (var > 1e-12):
        var = 1e-12
    return var


def nagarch_neg_loglik(r: np.ndarray, params: Dict[str, float], dist: str, dof: float) -> float:
    if r.size == 0:
        return HUGE_PENALTY
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    mu = params["mu"]
    if not (omega > 0.0 and alpha >= 0.0 and beta >= 0.0):
        return HUGE_PENALTY
    u = alpha * (1.0 + gamma * gamma) + beta
    if not (u < 1.0):
        return HUGE_PENALTY
    if dist == "student-t" and not (dof > 2.0001):
        return HUGE_PENALTY
    if dist not in {"normal", "student-t"}:
        return HUGE_PENALTY
    denom = 1.0 - u
    h = omega / denom
    if not (h > 0.0 and math.isfinite(h)):
        return HUGE_PENALTY
    nll = 0.0
    log2pi = math.log(2.0 * K_PI)
    for value in r:
        eps = value - mu
        if not (h > 0.0 and math.isfinite(h)):
            return HUGE_PENALTY
        sd = math.sqrt(h)
        z = eps / sd
        if dist == "normal":
            nll += 0.5 * (log2pi + math.log(h) + (eps * eps) / h)
        else:
            scale = standardized_t_scale(dof)
            x = scale * z
            log_fz = log_t_pdf(x, dof) + math.log(scale)
            log_feps = log_fz - 0.5 * math.log(h)
            nll += -log_feps
        shock = eps - gamma * sd
        h = omega + alpha * (shock * shock) + beta * h
    if not math.isfinite(nll):
        return HUGE_PENALTY
    return nll

def igarch_neg_loglik(r: np.ndarray, params: Dict[str, float], dist: str, dof: float) -> float:
    if r.size == 0:
        return HUGE_PENALTY
    alpha = params["alpha"]
    beta = params["beta"]
    mu = params["mu"]
    if not (alpha > 0.0 and beta >= 0.0):
        return HUGE_PENALTY
    if abs(alpha + beta - 1.0) > 1e-4:
        return HUGE_PENALTY
    if dist == "student-t" and not (dof > 2.0001):
        return HUGE_PENALTY
    if dist not in {"normal", "student-t"}:
        return HUGE_PENALTY
    h = initial_variance(r, mu)
    nll = 0.0
    log2pi = math.log(2.0 * K_PI)
    for value in r:
        eps = value - mu
        if not (h > 0.0 and math.isfinite(h)):
            return HUGE_PENALTY
        if dist == "normal":
            nll += 0.5 * (log2pi + math.log(h) + (eps * eps) / h)
        else:
            scale = standardized_t_scale(dof)
            z = eps / math.sqrt(h)
            x = scale * z
            log_fz = log_t_pdf(x, dof) + math.log(scale)
            log_feps = log_fz - 0.5 * math.log(h)
            nll += -log_feps
        h = alpha * (eps * eps) + beta * h
    if not math.isfinite(nll):
        return HUGE_PENALTY
    return nll


def nagarch_cond_sd(r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    mu = params["mu"]
    u = alpha * (1.0 + gamma * gamma) + beta
    denom = 1.0 - u
    h = omega / denom if denom > 1e-12 else omega
    if not (h > 0.0):
        h = 1e-6
    out = np.zeros_like(r)
    for i, value in enumerate(r):
        if not (h > 0.0):
            h = 1e-6
        sd = math.sqrt(max(h, 1e-12))
        out[i] = sd
        eps = value - mu
        shock = eps - gamma * sd
        h_next = omega + alpha * (shock * shock) + beta * h
        if not (h_next > 0.0) or not math.isfinite(h_next):
            h_next = 1e-8
        h = h_next
    return out


def st_neg_loglik(r: np.ndarray, params: Dict[str, float], dist: str, dof: float) -> float:
    if r.size == 0:
        return HUGE_PENALTY
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    mu = params["mu"]
    shift = params["shift"]
    if not (omega > 0.0 and alpha >= 0.0 and beta >= 0.0 and gamma >= 0.0):
        return HUGE_PENALTY
    u = alpha + 0.5 * gamma + beta
    if not (u < 1.0):
        return HUGE_PENALTY
    if dist == "student-t" and not (dof > 2.0001):
        return HUGE_PENALTY
    if dist not in {"normal", "student-t"}:
        return HUGE_PENALTY
    denom = 1.0 - u
    h = omega / denom
    if not (h > 0.0 and math.isfinite(h)):
        return HUGE_PENALTY
    nll = 0.0
    log2pi = math.log(2.0 * K_PI)
    for value in r:
        eps = value - mu
        if not (h > 0.0 and math.isfinite(h)):
            return HUGE_PENALTY
        sd = math.sqrt(h)
        z = eps / sd
        if dist == "normal":
            nll += 0.5 * (log2pi + math.log(h) + (eps * eps) / h)
        else:
            scale = standardized_t_scale(dof)
            x = scale * z
            log_fz = log_t_pdf(x, dof) + math.log(scale)
            log_feps = log_fz - 0.5 * math.log(h)
            nll += -log_feps
        indicator = 1.0 if eps < 0.0 else 0.0
        shock = eps - shift * sd
        var_scale = alpha + gamma * indicator
        h = omega + var_scale * (shock * shock) + beta * h
    if not math.isfinite(nll):
        return HUGE_PENALTY
    return nll


def igarch_cond_sd(r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    mu = params["mu"]
    alpha = params["alpha"]
    beta = params["beta"]
    h = initial_variance(r, mu)
    cond = np.empty_like(r)
    for idx, value in enumerate(r):
        cond[idx] = math.sqrt(max(h, 1e-12))
        eps = value - mu
        h = alpha * (eps * eps) + beta * h
    return cond


def st_cond_sd(r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    omega = params["omega"]
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]
    mu = params["mu"]
    shift = params["shift"]
    u = alpha + 0.5 * gamma + beta
    denom = 1.0 - u
    h = omega / denom if denom > 1e-12 else omega
    if not (h > 0.0):
        h = 1e-6
    out = np.zeros_like(r)
    for i, value in enumerate(r):
        if not (h > 0.0):
            h = 1e-6
        sd = math.sqrt(max(h, 1e-12))
        out[i] = sd
        eps = value - mu
        indicator = 1.0 if eps < 0.0 else 0.0
        shock = eps - shift * sd
        var_scale = alpha + gamma * indicator
        h_next = omega + var_scale * (shock * shock) + beta * h
        if not (h_next > 0.0) or not math.isfinite(h_next):
            h_next = 1e-8
        h = h_next
    return out


def minimize_with_simplex(func, theta0: np.ndarray, steps: np.ndarray, maxiter: int) -> np.ndarray:
    steps = np.where(steps > 0, steps, 0.1)
    simplex = [theta0]
    for i in range(theta0.size):
        vertex = theta0.copy()
        vertex[i] += steps[i]
        simplex.append(vertex)
    result = optimize.minimize(
        func,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "fatol": 1e-8, "xatol": 1e-8, "initial_simplex": np.array(simplex)},
    )
    return result.x if result.x.size else theta0


def fit_nagarch(r: np.ndarray, dist: str, dof0: float = 6.0, warm: Optional[np.ndarray] = None):
    if r.size <= 5:
        raise ValueError("fit_nagarch requires more data")
    mean = float(r.mean())
    var = float(np.mean((r - mean) ** 2))
    if not (var > 0.0):
        var = 1e-8
    u0 = 0.97
    v0 = 0.50
    g0 = 0.0
    omega0 = var * (1.0 - u0)
    if not (omega0 > 0.0):
        omega0 = 1e-10
    sd = math.sqrt(var)
    if dist == "normal":
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.20])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), g0])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: nagarch_neg_loglik(r, theta_to_nagarch_params(th), dist, 0.0)
        th_star = minimize_with_simplex(func, theta0, steps, 3000)
        params = theta_to_nagarch_params(th_star)
        return params, 0.0, func(th_star)
    if dist == "student-t":
        if not (dof0 > 2.0001):
            dof0 = 6.0
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.20, 0.30])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), g0, math.log(dof0 - 2.0)])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: nagarch_neg_loglik(r, theta_to_nagarch_params(th[:-1]), dist, theta_to_dof(th[-1]))
        th_star = minimize_with_simplex(func, theta0, steps, 4000)
        params = theta_to_nagarch_params(th_star[:-1])
        dof = theta_to_dof(th_star[-1])
        return params, dof, func(th_star)
    raise ValueError("invalid dist")


def fit_igarch(r: np.ndarray, dist: str, dof0: float = 6.0, warm: Optional[np.ndarray] = None):
    if r.size <= 5:
        raise ValueError("fit_igarch requires more data")
    mean = float(r.mean())
    var = float(np.mean((r - mean) ** 2))
    if not (var > 0.0):
        var = 1e-8
    sd = math.sqrt(var)
    alpha0 = 0.1
    if dist == "normal":
        steps = np.array([0.10 * sd, 0.50])
        theta0 = np.array([mean, logit(alpha0)])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: igarch_neg_loglik(r, theta_to_igarch_params(th), dist, 0.0)
        th_star = minimize_with_simplex(func, theta0, steps, 2000)
        params = theta_to_igarch_params(th_star)
        return params, 0.0, func(th_star)
    if dist == "student-t":
        if not (dof0 > 2.0001):
            dof0 = 6.0
        steps = np.array([0.10 * sd, 0.50, 0.30])
        theta0 = np.array([mean, logit(alpha0), math.log(dof0 - 2.0)])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: igarch_neg_loglik(
            r, theta_to_igarch_params(th[:-1]), dist, theta_to_dof(th[-1])
        )
        th_star = minimize_with_simplex(func, theta0, steps, 2500)
        params = theta_to_igarch_params(th_star[:-1])
        dof = theta_to_dof(th_star[-1])
        return params, dof, func(th_star)
    raise ValueError("invalid dist")


def fit_st(r: np.ndarray, dist: str, dof0: float = 6.0, warm: Optional[np.ndarray] = None):
    if r.size <= 5:
        raise ValueError("fit_st requires more data")
    mean = float(r.mean())
    var = float(np.mean((r - mean) ** 2))
    if not (var > 0.0):
        var = 1e-8
    u0 = 0.97
    v0 = 0.50
    w0 = 0.50
    shift0 = 0.0
    omega0 = var * (1.0 - u0)
    if not (omega0 > 0.0):
        omega0 = 1e-10
    sd = math.sqrt(var)
    if dist == "normal":
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.50, 0.10])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), logit(w0), shift0])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: st_neg_loglik(r, theta_to_st_params(th), dist, 0.0)
        th_star = minimize_with_simplex(func, theta0, steps, 4000)
        params = theta_to_st_params(th_star)
        return params, 0.0, func(th_star)
    if dist == "student-t":
        if not (dof0 > 2.0001):
            dof0 = 6.0
        steps = np.array([0.10 * sd, 0.50, 0.50, 0.50, 0.50, 0.10, 0.30])
        theta0 = np.array([mean, math.log(omega0), logit(u0), logit(v0), logit(w0), shift0, math.log(dof0 - 2.0)])
        if warm is not None and warm.size == theta0.size:
            theta0 = warm
        func = lambda th: st_neg_loglik(r, theta_to_st_params(th[:-1]), dist, theta_to_dof(th[-1]))
        th_star = minimize_with_simplex(func, theta0, steps, 4500)
        params = theta_to_st_params(th_star[:-1])
        dof = theta_to_dof(th_star[-1])
        return params, dof, func(th_star)
    raise ValueError("invalid dist")


@dataclass
class ModelRow:
    label: str
    mu: float = math.nan
    omega: float = math.nan
    alpha: float = math.nan
    beta: float = math.nan
    gamma: float = math.nan
    shift: float = math.nan
    dof: float = math.nan
    uncond_sd: float = math.nan
    loglik: float = math.nan
    n_params: int = 0
    cond_sd: Optional[np.ndarray] = None
    std_resid: Optional[np.ndarray] = None


@dataclass
class CriteriaSummary:
    aicc_order: List[str] = field(default_factory=list)
    bic_order: List[str] = field(default_factory=list)


@dataclass
class ColumnSummary:
    name: str
    criteria: CriteriaSummary

def summary_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values)
    n = arr.size
    if n == 0:
        return {k: math.nan for k in ["n", "mean", "sd", "skew", "exkurt", "min", "max"]}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=0))
    centered = arr - mean
    m2 = float(np.mean(centered ** 2))
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))
    skew = math.nan
    exkurt = math.nan
    if m2 > 0:
        skew = m3 / (m2 ** 1.5)
        exkurt = m4 / (m2 * m2) - 3.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "skew": skew,
        "exkurt": exkurt,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def autocorrelations(values: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(values)
    n = arr.size
    if k <= 0 or n <= 1:
        return np.array([])
    k = min(k, n - 1)
    mean = arr.mean()
    centered = arr - mean
    denom = np.dot(centered, centered)
    if not (denom > 0):
        return np.full(k, math.nan)
    acf = []
    for lag in range(1, k + 1):
        num = np.dot(centered[lag:], centered[:-lag])
        acf.append(num / denom)
    return np.array(acf)


def print_summary_table(stats_dict: Dict[str, float], width: int = 15, precision: int = 4) -> None:
    headers = ["n", "mean", "sd", "skew", "ex_kurtosis", "min", "max"]
    print("         n" + "".join(f"{h:>{width}}" for h in headers[1:]))
    values = [stats_dict.get("n", math.nan), stats_dict.get("mean", math.nan), stats_dict.get("sd", math.nan),
              stats_dict.get("skew", math.nan), stats_dict.get("exkurt", math.nan), stats_dict.get("min", math.nan),
              stats_dict.get("max", math.nan)]
    line = f"{int(values[0]):10d}"
    for val in values[1:]:
        if math.isfinite(val):
            line += f"{val:{width}.{precision}f}"
        else:
            line += f"{'NA':>{width}}"
    print(line)


def print_autocorr_table(returns: np.ndarray, lags: int, width: int = 12, precision: int = 3) -> None:
    print(f"autocorrelations (lag 1-{lags})")
    headers = ["lag", "returns", "|returns|", "returns^2"]
    print(f"{headers[0]:>6} {headers[1]:>{width}} {headers[2]:>{width}} {headers[3]:>{width}}")
    abs_vals = np.abs(returns)
    sq_vals = returns ** 2
    ac_return = autocorrelations(returns, lags)
    ac_abs = autocorrelations(abs_vals, lags)
    ac_sq = autocorrelations(sq_vals, lags)
    for lag in range(1, lags + 1):
        def fmt(value: float) -> str:
            return f"{value:{width}.{precision}f}" if math.isfinite(value) else f"{'NA':>{width}}"
        print(f"{lag:6d} {fmt(ac_return[lag - 1] if ac_return.size >= lag else math.nan)}"
              f" {fmt(ac_abs[lag - 1] if ac_abs.size >= lag else math.nan)}"
              f" {fmt(ac_sq[lag - 1] if ac_sq.size >= lag else math.nan)}")


def compute_aicc(loglik: float, k: int, n: int) -> float:
    if n <= 0:
        return math.nan
    denom = n - k - 1
    if denom <= 0:
        return math.nan
    aic = 2.0 * k - 2.0 * loglik
    return aic + (2.0 * k * (k + 1)) / denom


def compute_bic(loglik: float, k: int, n: int) -> float:
    if n <= 0:
        return math.nan
    return math.log(n) * k - 2.0 * loglik


def print_model_table(rows: List[ModelRow], aicc_vals: List[float], bic_vals: List[float],
                      loglik_ranks: List[int], aicc_ranks: List[int], bic_ranks: List[int], precision: int = 6) -> None:
    headers = ["uncond_sd", "mu", "omega", "alpha", "beta", "gamma", "shift", "dof",
               "loglik", "n_params", "AICC", "BIC", "loglik_rank", "AICC_rank", "BIC_rank"]
    print("model" + "".join(f"{h:>20}" for h in headers))
    for idx, row in enumerate(rows):
        values = [row.uncond_sd, row.mu, row.omega, row.alpha, row.beta, row.gamma, row.shift, row.dof,
                  row.loglik]
        formatted = []
        for val in values:
            if math.isfinite(val):
                formatted.append(f"{val:20.{precision}f}")
            else:
                formatted.append(f"{'NA':>20}")
        n_params_str = f"{row.n_params:20d}"
        aicc_val = aicc_vals[idx]
        bic_val = bic_vals[idx]
        formatted_aicc = f"{aicc_val:20.{precision}f}" if math.isfinite(aicc_val) else f"{'NA':>20}"
        formatted_bic = f"{bic_val:20.{precision}f}" if math.isfinite(bic_val) else f"{'NA':>20}"
        print(f"{row.label:<20}" + "".join(formatted) + n_params_str + formatted_aicc + formatted_bic +
              f"{loglik_ranks[idx]:20d}{aicc_ranks[idx]:20d}{bic_ranks[idx]:20d}")


def sorted_indices(values: List[float], ascending: bool = True) -> List[int]:
    finite = [math.isfinite(v) for v in values]
    order = list(range(len(values)))
    def key(idx: int):
        if not finite[idx]:
            return (1, idx)
        return (0, values[idx]) if ascending else (0, -values[idx])
    order.sort(key=key)
    return order


def compute_ranks(values: List[float], ascending: bool = True) -> List[int]:
    order = sorted_indices(values, ascending)
    ranks = [len(values)] * len(values)
    for pos, idx in enumerate(order):
        ranks[idx] = pos + 1
    return ranks


def print_selects(label: str, names: List[str], values: List[float]) -> None:
    order = sorted_indices(values, ascending=True)
    print(f"\n{label} selects")
    if not order:
        print("  (no models)")
        return
    base = values[order[0]]
    for pos, idx in enumerate(order):
        name = names[idx]
        val = values[idx]
        diff = val - base if math.isfinite(val) and math.isfinite(base) else math.nan
        val_str = f"{val:18.6f}" if math.isfinite(val) else f"{'NA':>18}"
        diff_str = f"{diff:18.6f}" if math.isfinite(diff) else f"{'NA':>18}"
        suffix = "  best" if pos == 0 else ""
        print(f"{name:<18} {val_str} {diff_str}{suffix}")

def residual_summary_table(entries: List[Tuple[str, np.ndarray]], width: int = 15, precision: int = 4) -> None:
    print("\nstandardized residual stats")
    header = ["model", "n", "mean", "sd", "skew", "ex_kurtosis", "min", "max"]
    print(f"{header[0]:<18} {header[1]:>10}" + "".join(f"{h:>{width}}" for h in header[2:]))
    for name, resid in entries:
        stats = summary_stats(resid)
        line = f"{name:<18} {int(stats['n']):>10d}"
        for key in ["mean", "sd", "skew", "exkurt", "min", "max"]:
            val = stats[key]
            if math.isfinite(val):
                line += f"{val:{width}.{precision}f}"
            else:
                line += f"{'NA':>{width}}"
        print(line)


def cond_sd_summary_table(entries: List[Tuple[str, np.ndarray]], width: int = 15, precision: int = 4) -> None:
    print("\nconditional sd stats")
    header = ["model", "n", "mean", "sd", "skew", "ex_kurtosis", "min", "max"]
    print(f"{header[0]:<18} {header[1]:>10}" + "".join(f"{h:>{width}}" for h in header[2:]))
    for name, cond_sd in entries:
        stats = summary_stats(cond_sd)
        line = f"{name:<18} {int(stats['n']):>10d}"
        for key in ["mean", "sd", "skew", "exkurt", "min", "max"]:
            val = stats[key]
            if math.isfinite(val):
                line += f"{val:{width}.{precision}f}"
            else:
                line += f"{'NA':>{width}}"
        print(line)


def fit_constant_vol(returns: np.ndarray) -> ModelRow:
    mu = 0.0
    omega = float(np.mean(returns ** 2))
    if not (omega > 1e-12):
        omega = 1e-12
    sd = math.sqrt(omega)
    n = returns.size
    if n == 0:
        loglik = math.nan
    else:
        loglik = -0.5 * (n * math.log(2.0 * K_PI * omega) + np.sum((returns - mu) ** 2) / omega)
    cond_sd = np.full(n, sd)
    std_resid = (returns - mu) / cond_sd
    return ModelRow(label="constant_vol", mu=mu, omega=omega, alpha=math.nan, beta=math.nan,
                    gamma=math.nan, shift=math.nan, dof=math.nan, uncond_sd=sd, loglik=loglik,
                    n_params=1, cond_sd=cond_sd, std_resid=std_resid)


def fit_arch_model(returns: np.ndarray, model: str, dist: str) -> ModelRow:
    cm = ConstantMean(returns)
    if model == "garch":
        cm.volatility = GARCH(p=1, o=0, q=1)
    elif model == "gjr":
        cm.volatility = GARCH(p=1, o=1, q=1)
    else:
        raise ValueError("unknown model type")
    if dist == "normal":
        cm.distribution = Normal()
    elif dist == "student-t":
        cm.distribution = StudentsT()
    else:
        raise ValueError("unknown dist")
    res = cm.fit(disp="off")
    params = res.params
    mu = float(params.get("mu", 0.0))
    omega = float(params.get("omega", math.nan))
    alpha = float(params.get("alpha[1]", math.nan))
    beta = float(params.get("beta[1]", math.nan))
    gamma = float(params.get("gamma[1]", math.nan)) if model == "gjr" else math.nan
    dof = float(params.get("nu", math.nan)) if dist == "student-t" else math.nan
    if model == "garch":
        denom = 1.0 - (alpha + beta)
    else:
        gamma_val = gamma if math.isfinite(gamma) else 0.0
        denom = 1.0 - (alpha + 0.5 * gamma_val + beta)
    uncond_sd = math.sqrt(omega / denom) if denom > 0 and omega > 0 else math.nan
    n_params = 4 if model == "garch" else 5
    if dist == "student-t":
        n_params += 1
    cond_sd = np.asarray(getattr(res, "conditional_volatility"))
    std_resid = np.asarray(getattr(res, "std_resid"))
    return ModelRow(label=f"{model}_{'student_t' if dist == 'student-t' else 'normal'}",
                    mu=mu, omega=omega, alpha=alpha, beta=beta, gamma=gamma,
                    shift=math.nan, dof=dof, uncond_sd=uncond_sd, loglik=float(res.loglikelihood),
                    n_params=n_params, cond_sd=cond_sd,
                    std_resid=std_resid)


def fit_nagarch_model(returns: np.ndarray, dist: str) -> ModelRow:
    params, dof, nll = fit_nagarch(returns, dist)
    gamma = params["gamma"]
    persistence = params["alpha"] * (1.0 + gamma * gamma) + params["beta"]
    uncond = math.sqrt(params["omega"] / (1.0 - persistence)) if (1.0 - persistence) > 0 else math.nan
    cond_sd = nagarch_cond_sd(returns, params)
    std_resid = (returns - params["mu"]) / cond_sd
    n_params = 5 if dist == "normal" else 6
    label = f"nagarch_{'student_t' if dist == 'student-t' else 'normal'}"
    dof_val = dof if dist == "student-t" else math.nan
    return ModelRow(label=label, mu=params["mu"], omega=params["omega"], alpha=params["alpha"],
                    beta=params["beta"], gamma=gamma, shift=math.nan, dof=dof_val,
                    uncond_sd=uncond, loglik=-nll, n_params=n_params, cond_sd=cond_sd,
                    std_resid=std_resid)


def fit_igarch_model(returns: np.ndarray, dist: str) -> ModelRow:
    params, dof, nll = fit_igarch(returns, dist)
    cond_sd = igarch_cond_sd(returns, params)
    std_resid = (returns - params["mu"]) / cond_sd
    n_params = 2 if dist == "normal" else 3
    label = f"igarch_{'student_t' if dist == 'student-t' else 'normal'}"
    dof_val = dof if dist == "student-t" else math.nan
    return ModelRow(label=label,
                    mu=params["mu"],
                    omega=math.nan,
                    alpha=params["alpha"],
                    beta=params["beta"],
                    gamma=math.nan,
                    shift=math.nan,
                    dof=dof_val,
                    uncond_sd=math.nan,
                    loglik=-nll,
                    n_params=n_params,
                    cond_sd=cond_sd,
                    std_resid=std_resid)


def fit_st_model(returns: np.ndarray, dist: str) -> ModelRow:
    params, dof, nll = fit_st(returns, dist)
    gamma = params["gamma"]
    shift = params["shift"]
    persistence = params["alpha"] + 0.5 * gamma + params["beta"]
    denom = 1.0 - persistence
    uncond = math.sqrt(params["omega"] / denom) if denom > 0 and params["omega"] > 0 else math.nan
    cond_sd = st_cond_sd(returns, params)
    std_resid = (returns - params["mu"]) / cond_sd
    n_params = 6 if dist == "normal" else 7
    label = f"st_{'student_t' if dist == 'student-t' else 'normal'}"
    dof_val = dof if dist == "student-t" else math.nan
    return ModelRow(
        label=label,
        mu=params["mu"],
        omega=params["omega"],
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=gamma,
        shift=shift,
        dof=dof_val,
        uncond_sd=uncond,
        loglik=-nll,
        n_params=n_params,
        cond_sd=cond_sd,
        std_resid=std_resid,
    )


def compute_loglik_select_values(rows: List[ModelRow]) -> List[float]:
    out = []
    for row in rows:
        out.append(-row.loglik if math.isfinite(row.loglik) else math.inf)
    return out


def compute_metric_order(rows: List[ModelRow], metric_values: List[float]) -> List[str]:
    indices = sorted_indices(metric_values, ascending=True)
    return [rows[idx].label for idx in indices]


ALLOWED_MODELS = {
    "nagarch_normal",
    "nagarch_student_t",
    "garch_normal",
    "garch_student_t",
    "gjr_normal",
    "gjr_student_t",
    "igarch_normal",
    "igarch_student_t",
    "st_normal",
    "st_student_t",
    "constant_vol",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit GARCH-family models using arch")
    parser.add_argument("--file", default="prices.csv")
    parser.add_argument("--columns", default="")
    parser.add_argument(
        "--models",
        default=(
            "nagarch_normal,nagarch_student_t,garch_student_t,garch_normal,"
            "gjr_student_t,gjr_normal,igarch_student_t,igarch_normal,"
            "constant_vol" # st_student_t,st_normal,
        ),
    )
    parser.add_argument("--max-columns", type=int, default=-1)
    parser.add_argument("--min-rows", type=int, default=250)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--no-demean", action="store_true")
    parser.add_argument("--no-resid-stats", action="store_true")
    parser.add_argument("--cond-sd-stats", action="store_true", default=True)
    parser.add_argument("--autocorr-lags", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scale <= 0:
        print("--scale must be positive", file=sys.stderr)
        return 1
    try:
        raw = pd.read_csv(args.file)
    except Exception as exc:
        print(f"failed to read {args.file}: {exc}", file=sys.stderr)
        return 1
    if raw.shape[1] < 2:
        print("expecting at least one price column", file=sys.stderr)
        return 1
    date_col = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    prices = raw.iloc[:, 1:]
    prices.columns = [col.strip() for col in prices.columns]
    n_rows, n_cols = prices.shape
    print(f"loaded price data from {args.file} with {n_rows} rows and {n_cols} columns")
    if date_col.notna().any():
        first = date_col.dropna().iloc[0]
        last = date_col.dropna().iloc[-1]
        first_str = first.strftime("%Y-%m-%d") if isinstance(first, pd.Timestamp) else str(first)
        last_str = last.strftime("%Y-%m-%d") if isinstance(last, pd.Timestamp) else str(last)
        print(f"date range: {first_str} to {last_str}")
    print(f"return scaling factor: {args.scale}")
    print(f"demean returns: {'no' if args.no_demean else 'yes'}")
    requested = [name.strip() for name in args.models.split(',') if name.strip()]
    for name in requested:
        if name not in ALLOWED_MODELS:
            print(f"unknown model: {name}", file=sys.stderr)
            return 1
    columns = prices.columns.tolist()
    if args.columns.strip():
        subset = []
        for name in args.columns.split(','):
            name = name.strip()
            if name:
                if name not in columns:
                    print(f"unknown column {name}", file=sys.stderr)
                    return 1
                if name not in subset:
                    subset.append(name)
        columns = subset
    if args.max_columns > 0 and len(columns) > args.max_columns:
        columns = columns[: args.max_columns]
        print(f"limiting to first {args.max_columns} column(s)")
    summaries: List[ColumnSummary] = []
    for column in columns:
        series = prices[column].dropna()
        price_obs = series.size
        returns = np.diff(np.log(series.values))
        returns = returns * args.scale
        stats = summary_stats(returns)
        print(f"\n==== column: {column} ====")
        print(f"price observations: {price_obs}, log returns used: {returns.size}")
        print_summary_table(stats)
        print()
        print_autocorr_table(returns, args.autocorr_lags)
        adjusted_returns = returns if args.no_demean else returns - returns.mean()
        if adjusted_returns.size < args.min_rows:
            print(f"not enough data (need {args.min_rows})")
            continue
        model_rows: List[ModelRow] = []
        residual_entries: List[Tuple[str, np.ndarray]] = []
        cond_sd_entries: List[Tuple[str, np.ndarray]] = []
        for model in requested:
            try:
                if model == "constant_vol":
                    row = fit_constant_vol(adjusted_returns)
                elif model.startswith("garch"):
                    dist = "student-t" if model.endswith("student_t") else "normal"
                    row = fit_arch_model(adjusted_returns, "garch", dist)
                elif model.startswith("gjr"):
                    dist = "student-t" if model.endswith("student_t") else "normal"
                    row = fit_arch_model(adjusted_returns, "gjr", dist)
                elif model.startswith("igarch"):
                    dist = "student-t" if model.endswith("student_t") else "normal"
                    row = fit_igarch_model(adjusted_returns, dist)
                elif model.startswith("nagarch"):
                    dist = "student-t" if model.endswith("student_t") else "normal"
                    row = fit_nagarch_model(adjusted_returns, dist)
                elif model.startswith("st"):
                    dist = "student-t" if model.endswith("student_t") else "normal"
                    row = fit_st_model(adjusted_returns, dist)
                else:
                    continue
            except Exception as exc:
                print(f"failed to fit {model}: {exc}")
                continue
            model_rows.append(row)
            residual_entries.append((row.label, row.std_resid))
            if row.cond_sd is not None:
                cond_sd_entries.append((row.label, row.cond_sd))
        if not model_rows:
            print("no models fitted")
            continue
        n_obs = adjusted_returns.size
        aicc_vals = [compute_aicc(row.loglik, row.n_params, n_obs) for row in model_rows]
        bic_vals = [compute_bic(row.loglik, row.n_params, n_obs) for row in model_rows]
        loglik_vals = [row.loglik for row in model_rows]
        loglik_ranks = compute_ranks(loglik_vals, ascending=False)
        aicc_ranks = compute_ranks(aicc_vals)
        bic_ranks = compute_ranks(bic_vals)
        print()
        print_model_table(model_rows, aicc_vals, bic_vals, loglik_ranks, aicc_ranks, bic_ranks)
        names = [row.label for row in model_rows]
        loglik_select_vals = compute_loglik_select_values(model_rows)
        print_selects("loglik", names, loglik_select_vals)
        print_selects("AICC", names, aicc_vals)
        print_selects("BIC", names, bic_vals)
        if not args.no_resid_stats:
            residual_summary_table(residual_entries)
        if args.cond_sd_stats and cond_sd_entries:
            cond_sd_summary_table(cond_sd_entries)
        summaries.append(ColumnSummary(name=column,
                                       criteria=CriteriaSummary(
                                           aicc_order=compute_metric_order(model_rows, aicc_vals),
                                           bic_order=compute_metric_order(model_rows, bic_vals))))
    print("\n==== summary ====")
    if not summaries:
        print("no fitted models to summarize")
    else:
        for entry in summaries:
            print(f"{entry.name} AICC: {', '.join(entry.criteria.aicc_order)}")
            print(f"{entry.name} BIC: {', '.join(entry.criteria.bic_order)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
