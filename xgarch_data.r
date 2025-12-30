suppressMessages({
  if (!requireNamespace("rugarch", quietly = TRUE)) {
    stop("The rugarch package is required. Please install it before running this script.")
  }
})

sigmoid <- function(x) {
  ifelse(x >= 0, 1 / (1 + exp(-x)), exp(x) / (1 + exp(x)))
}

logit <- function(p) {
  if (p <= 0) return(-1e300)
  if (p >= 1) return(1e300)
  log(p / (1 - p))
}

theta_to_dof <- function(val) 2 + exp(val)

theta_to_nagarch_params <- function(th) {
  u <- sigmoid(th[3])
  v <- sigmoid(th[4])
  gamma <- th[5]
  beta <- u * v
  alpha <- u * (1 - v) / (1 + gamma * gamma)
  list(mu = th[1], omega = exp(th[2]), alpha = alpha, beta = beta, gamma = gamma)
}

theta_to_igarch_params <- function(th) {
  eps <- 1e-6
  alpha_raw <- sigmoid(th[2])
  alpha <- eps + (1 - 2 * eps) * alpha_raw
  beta <- 1 - alpha
  list(mu = th[1], alpha = alpha, beta = beta)
}

theta_to_st_params <- function(th) {
  u <- sigmoid(th[3])
  v <- sigmoid(th[4])
  w <- sigmoid(th[5])
  remainder <- u * (1 - v)
  beta <- u * v
  gamma <- 2 * remainder * w
  alpha <- remainder * (1 - w)
  list(mu = th[1], omega = exp(th[2]), alpha = alpha, beta = beta, gamma = gamma, shift = th[6])
}

log_t_pdf <- function(x, dof) {
  lgamma(0.5 * (dof + 1)) - lgamma(0.5 * dof) - 0.5 * (log(dof) + log(pi)) - 0.5 * (dof + 1) * log1p((x * x) / dof)
}

standardized_t_scale <- function(dof) sqrt(dof / (dof - 2))

initial_variance <- function(r, mu) {
  var <- mean((r - mu)^2)
  if (!is.finite(var) || var <= 1e-12) var <- 1e-12
  var
}

nagarch_neg_loglik <- function(r, params, dist, dof) {
  n <- length(r)
  if (n == 0) return(1e300)
  omega <- params$omega; alpha <- params$alpha; beta <- params$beta; gamma <- params$gamma; mu <- params$mu
  if (!(omega > 0 && alpha >= 0 && beta >= 0)) return(1e300)
  u <- alpha * (1 + gamma * gamma) + beta
  if (!(u < 1)) return(1e300)
  if (dist == "student-t" && !(dof > 2.0001)) return(1e300)
  if (!(dist %in% c("normal", "student-t"))) return(1e300)
  h <- omega / (1 - u)
  if (!(h > 0) || !is.finite(h)) return(1e300)
  nll <- 0
  log2pi <- log(2 * pi)
  for (value in r) {
    eps <- value - mu
    if (!(h > 0) || !is.finite(h)) return(1e300)
    if (dist == "normal") {
      nll <- nll + 0.5 * (log2pi + log(h) + (eps * eps) / h)
    } else {
      scale <- standardized_t_scale(dof)
      z <- eps / sqrt(h)
      x <- scale * z
      log_fz <- log_t_pdf(x, dof) + log(scale)
      nll <- nll - (log_fz - 0.5 * log(h))
    }
    shock <- eps - gamma * sqrt(h)
    h <- omega + alpha * shock * shock + beta * h
  }
  if (!is.finite(nll)) return(1e300)
  nll
}

st_neg_loglik <- function(r, params, dist, dof) {
  n <- length(r)
  if (n == 0) return(1e300)
  omega <- params$omega; alpha <- params$alpha; beta <- params$beta; gamma <- params$gamma; mu <- params$mu; shift <- params$shift
  if (!(omega > 0 && alpha >= 0 && beta >= 0 && gamma >= 0)) return(1e300)
  u <- alpha + 0.5 * gamma + beta
  if (!(u < 1)) return(1e300)
  if (dist == "student-t" && !(dof > 2.0001)) return(1e300)
  if (!(dist %in% c("normal", "student-t"))) return(1e300)
  h <- omega / (1 - u)
  if (!(h > 0) || !is.finite(h)) return(1e300)
  nll <- 0
  log2pi <- log(2 * pi)
  for (value in r) {
    eps <- value - mu
    if (!(h > 0) || !is.finite(h)) return(1e300)
    if (dist == "normal") {
      nll <- nll + 0.5 * (log2pi + log(h) + (eps * eps) / h)
    } else {
      scale <- standardized_t_scale(dof)
      z <- eps / sqrt(h)
      x <- scale * z
      log_fz <- log_t_pdf(x, dof) + log(scale)
      nll <- nll - (log_fz - 0.5 * log(h))
    }
    shock <- eps - shift * sqrt(h)
    indicator <- ifelse(eps < 0, 1, 0)
    scaled <- (alpha + gamma * indicator) * (shock * shock)
    h <- omega + scaled + beta * h
  }
  if (!is.finite(nll)) return(1e300)
  nll
}
igarch_neg_loglik <- function(r, params, dist, dof) {
  n <- length(r)
  if (n == 0) return(1e300)
  alpha <- params$alpha; beta <- params$beta; mu <- params$mu
  if (!(alpha > 0 && beta >= 0)) return(1e300)
  if (abs(alpha + beta - 1) > 1e-4) return(1e300)
  if (dist == "student-t" && !(dof > 2.0001)) return(1e300)
  if (!(dist %in% c("normal", "student-t"))) return(1e300)
  h <- initial_variance(r, mu)
  nll <- 0
  log2pi <- log(2 * pi)
  for (value in r) {
    eps <- value - mu
    if (!(h > 0) || !is.finite(h)) return(1e300)
    if (dist == "normal") {
      nll <- nll + 0.5 * (log2pi + log(h) + (eps * eps) / h)
    } else {
      scale <- standardized_t_scale(dof)
      z <- eps / sqrt(h)
      x <- scale * z
      log_fz <- log_t_pdf(x, dof) + log(scale)
      nll <- nll - (log_fz - 0.5 * log(h))
    }
    h <- alpha * eps * eps + beta * h
  }
  if (!is.finite(nll)) return(1e300)
  nll
}

nagarch_cond_sd <- function(r, params) {
  omega <- params$omega; alpha <- params$alpha; beta <- params$beta; gamma <- params$gamma; mu <- params$mu
  u <- alpha * (1 + gamma * gamma) + beta
  h <- omega / (1 - u)
  cond <- numeric(length(r))
  for (i in seq_along(r)) {
    cond[i] <- sqrt(max(h, 1e-12))
    eps <- r[i] - mu
    shock <- eps - gamma * sqrt(h)
    h <- omega + alpha * shock * shock + beta * h
  }
  cond
}

st_cond_sd <- function(r, params) {
  omega <- params$omega; alpha <- params$alpha; beta <- params$beta; gamma <- params$gamma; mu <- params$mu; shift <- params$shift
  u <- alpha + 0.5 * gamma + beta
  h <- omega / (1 - u)
  cond <- numeric(length(r))
  for (i in seq_along(r)) {
    cond[i] <- sqrt(max(h, 1e-12))
    eps <- r[i] - mu
    indicator <- ifelse(eps < 0, 1, 0)
    scaled <- (alpha + gamma * indicator)
    shock <- eps - shift * sqrt(h)
    h <- omega + scaled * shock * shock + beta * h
  }
  cond
}

igarch_cond_sd <- function(r, params) {
  mu <- params$mu; alpha <- params$alpha; beta <- params$beta
  h <- initial_variance(r, mu)
  cond <- numeric(length(r))
  for (i in seq_along(r)) {
    cond[i] <- sqrt(max(h, 1e-12))
    eps <- r[i] - mu
    h <- alpha * eps * eps + beta * h
  }
  cond
}

fit_nagarch <- function(r, dist, dof0 = 6) {
  if (length(r) <= 5) stop("fit_nagarch requires more data")
  mean_r <- mean(r)
  var_r <- mean((r - mean_r)^2)
  if (!(var_r > 0)) var_r <- 1e-8
  u0 <- 0.97; v0 <- 0.50; g0 <- 0
  omega0 <- var_r * (1 - u0)
  if (!(omega0 > 0)) omega0 <- 1e-10
  sd_r <- sqrt(var_r)
  if (dist == "normal") {
    theta0 <- c(mean_r, log(omega0), logit(u0), logit(v0), g0)
    obj <- function(th) nagarch_neg_loglik(r, theta_to_nagarch_params(th), dist, 0)
    opt <- optim(theta0, obj, method = "Nelder-Mead", control = list(maxit = 3000, reltol = 1e-8))
    params <- theta_to_nagarch_params(opt$par)
    return(list(params = params, dof = NA_real_, nll = opt$value))
  }
  if (dist == "student-t") {
    if (!(dof0 > 2.0001)) dof0 <- 6
    theta0 <- c(mean_r, log(omega0), logit(u0), logit(v0), g0, log(dof0 - 2))
    obj <- function(th) nagarch_neg_loglik(r, theta_to_nagarch_params(th[1:5]), dist, theta_to_dof(th[6]))
    opt <- optim(theta0, obj, method = "Nelder-Mead", control = list(maxit = 4000, reltol = 1e-8))
    params <- theta_to_nagarch_params(opt$par[1:5])
    return(list(params = params, dof = theta_to_dof(opt$par[6]), nll = opt$value))
  }
  stop("invalid dist")
}

fit_st <- function(r, dist, dof0 = 6) {
  if (length(r) <= 5) stop("fit_st requires more data")
  mean_r <- mean(r)
  var_r <- mean((r - mean_r)^2)
  if (!(var_r > 0)) var_r <- 1e-8
  u0 <- 0.97; v0 <- 0.50; w0 <- 0.50; shift0 <- 0
  omega0 <- var_r * (1 - u0)
  if (!(omega0 > 0)) omega0 <- 1e-10
  sd_r <- sqrt(var_r)
  if (dist == "normal") {
    theta0 <- c(mean_r, log(omega0), logit(u0), logit(v0), logit(w0), shift0)
    obj <- function(th) st_neg_loglik(r, theta_to_st_params(th), dist, 0)
    opt <- optim(theta0, obj, method = "Nelder-Mead", control = list(maxit = 4000, reltol = 1e-8))
    params <- theta_to_st_params(opt$par)
    return(list(params = params, dof = NA_real_, nll = opt$value))
  }
  if (dist == "student-t") {
    if (!(dof0 > 2.0001)) dof0 <- 6
    theta0 <- c(mean_r, log(omega0), logit(u0), logit(v0), logit(w0), shift0, log(dof0 - 2))
    obj <- function(th) st_neg_loglik(r, theta_to_st_params(th[1:6]), dist, theta_to_dof(th[7]))
    opt <- optim(theta0, obj, method = "Nelder-Mead", control = list(maxit = 4500, reltol = 1e-8))
    params <- theta_to_st_params(opt$par[1:6])
    return(list(params = params, dof = theta_to_dof(opt$par[7]), nll = opt$value))
  }
  stop("invalid dist")
}

fit_igarch_custom <- function(r, dist, dof0 = 6) {
  if (length(r) <= 5) stop("fit_igarch requires more data")
  mean_r <- mean(r)
  var_r <- mean((r - mean_r)^2)
  if (!(var_r > 0)) var_r <- 1e-8
  sd_r <- sqrt(var_r)
  alpha0 <- 0.1
  if (dist == "normal") {
    theta0 <- c(mean_r, logit(alpha0))
    obj <- function(th) igarch_neg_loglik(r, theta_to_igarch_params(th), dist, 0)
    opt <- optim(theta0, obj, method = "Nelder-Mead", control = list(maxit = 2000, reltol = 1e-8))
    params <- theta_to_igarch_params(opt$par)
    return(list(params = params, dof = NA_real_, nll = opt$value))
  }
  if (dist == "student-t") {
    if (!(dof0 > 2.0001)) dof0 <- 6
    theta0 <- c(mean_r, logit(alpha0), log(dof0 - 2))
    obj <- function(th) igarch_neg_loglik(r, theta_to_igarch_params(th[1:2]), dist, theta_to_dof(th[3]))
    opt <- optim(theta0, obj, method = "Nelder-Mead", control = list(maxit = 2500, reltol = 1e-8))
    params <- theta_to_igarch_params(opt$par[1:2])
    return(list(params = params, dof = theta_to_dof(opt$par[3]), nll = opt$value))
  }
  stop("invalid dist")
}

fit_constant_vol <- function(returns) {
  mu <- 0
  omega <- mean(returns^2)
  if (!(omega > 1e-12)) omega <- 1e-12
  sd_val <- sqrt(omega)
  n <- length(returns)
  loglik <- if (n == 0) NA_real_ else -0.5 * (n * log(2 * pi * omega) + sum((returns - mu)^2) / omega)
  cond_sd <- rep(sd_val, n)
  list(label = "constant_vol", mu = mu, omega = omega, alpha = NA_real_, beta = NA_real_,
       gamma = NA_real_, shift = NA_real_, dof = NA_real_, uncond_sd = sd_val,
       loglik = loglik, n_params = 1L, cond_sd = cond_sd, std_resid = (returns - mu) / cond_sd)
}

fit_rugarch_model <- function(returns, model, dist) {
  library(rugarch)
  dist_model <- ifelse(dist == "student-t", "std", "norm")
  if (model == "garch") {
    variance.model <- list(model = "sGARCH", garchOrder = c(1, 1))
  } else if (model == "gjr") {
    variance.model <- list(model = "gjrGARCH", garchOrder = c(1, 1))
  } else if (model == "igarch") {
    variance.model <- list(model = "iGARCH", garchOrder = c(1, 1))
  } else {
    stop("unknown rugarch model")
  }
  spec <- rugarch::ugarchspec(mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
                              variance.model = variance.model,
                              distribution.model = dist_model)
  fit <- rugarch::ugarchfit(spec, returns, solver = "hybrid", solver.control = list(trace = 0))
  params <- as.list(coef(fit))
  mu <- as.numeric(params$mu)
  omega <- if (!is.null(params$omega)) as.numeric(params$omega) else NA_real_
  alpha <- if (!is.null(params$alpha1)) as.numeric(params$alpha1) else NA_real_
  beta <- if (!is.null(params$beta1)) as.numeric(params$beta1) else NA_real_
  gamma <- if (!is.null(params$gamma1)) as.numeric(params$gamma1) else NA_real_
  dof <- if (dist == "student-t" && !is.null(params$shape)) as.numeric(params$shape) else NA_real_
  if (model == "garch") {
    denom <- 1 - (alpha + beta)
  } else if (model == "gjr") {
    gamma_val <- ifelse(is.finite(gamma), gamma, 0)
    denom <- 1 - (alpha + 0.5 * gamma_val + beta)
  } else {
    denom <- NA_real_
  }
  uncond_sd <- if (is.finite(omega) && is.finite(denom) && denom > 0) sqrt(omega / denom) else NA_real_
  n_params <- length(coef(fit))
  cond_sd <- as.numeric(rugarch::sigma(fit))
  std_resid <- as.numeric(residuals(fit, standardize = TRUE))
  list(label = sprintf("%s_%s", model, ifelse(dist == "student-t", "student_t", "normal")),
       mu = mu, omega = omega, alpha = alpha, beta = beta, gamma = gamma, shift = NA_real_,
       dof = dof, uncond_sd = uncond_sd, loglik = rugarch::likelihood(fit),
       n_params = n_params, cond_sd = cond_sd, std_resid = std_resid)
}

fit_nagarch_model <- function(returns, dist) {
  fit <- fit_nagarch(returns, dist)
  params <- fit$params; gamma <- params$gamma
  persistence <- params$alpha * (1 + gamma * gamma) + params$beta
  uncond <- if ((1 - persistence) > 0) sqrt(params$omega / (1 - persistence)) else NA_real_
  cond_sd <- nagarch_cond_sd(returns, params)
  std_resid <- (returns - params$mu) / cond_sd
  n_params <- ifelse(dist == "normal", 5L, 6L)
  list(label = sprintf("nagarch_%s", ifelse(dist == "student-t", "student_t", "normal")),
       mu = params$mu, omega = params$omega, alpha = params$alpha, beta = params$beta,
       gamma = gamma, shift = NA_real_, dof = ifelse(dist == "student-t", fit$dof, NA_real_),
       uncond_sd = uncond, loglik = -fit$nll, n_params = n_params,
       cond_sd = cond_sd, std_resid = std_resid)
}

fit_st_model <- function(returns, dist) {
  fit <- fit_st(returns, dist)
  params <- fit$params; gamma <- params$gamma; shift <- params$shift
  persistence <- params$alpha + 0.5 * gamma + params$beta
  denom <- 1 - persistence
  uncond <- if (denom > 0 && params$omega > 0) sqrt(params$omega / denom) else NA_real_
  cond_sd <- st_cond_sd(returns, params)
  std_resid <- (returns - params$mu) / cond_sd
  n_params <- ifelse(dist == "normal", 6L, 7L)
  list(label = sprintf("st_%s", ifelse(dist == "student-t", "student_t", "normal")),
       mu = params$mu, omega = params$omega, alpha = params$alpha, beta = params$beta,
       gamma = gamma, shift = shift, dof = ifelse(dist == "student-t", fit$dof, NA_real_),
       uncond_sd = uncond, loglik = -fit$nll, n_params = n_params,
       cond_sd = cond_sd, std_resid = std_resid)
}

fit_igarch_model <- function(returns, dist) {
  fit <- fit_igarch_custom(returns, dist)
  params <- fit$params
  cond_sd <- igarch_cond_sd(returns, params)
  std_resid <- (returns - params$mu) / cond_sd
  n_params <- ifelse(dist == "normal", 2L, 3L)
  list(label = sprintf("igarch_%s", ifelse(dist == "student-t", "student_t", "normal")),
       mu = params$mu, omega = NA_real_, alpha = params$alpha, beta = params$beta,
       gamma = NA_real_, shift = NA_real_, dof = ifelse(dist == "student-t", fit$dof, NA_real_),
       uncond_sd = NA_real_, loglik = -fit$nll, n_params = n_params,
       cond_sd = cond_sd, std_resid = std_resid)
}

summary_stats <- function(values) {
  n <- length(values)
  if (n == 0) return(list(n = 0, mean = NA_real_, sd = NA_real_, skew = NA_real_, exkurt = NA_real_, min = NA_real_, max = NA_real_))
  mean_val <- mean(values)
  sd_val <- sd(values)
  centered <- values - mean_val
  m2 <- mean(centered^2)
  m3 <- mean(centered^3)
  m4 <- mean(centered^4)
  skew <- ifelse(m2 > 0, m3 / (m2^(3/2)), NA_real_)
  exkurt <- ifelse(m2 > 0, m4 / (m2^2) - 3, NA_real_)
  list(n = n, mean = mean_val, sd = sd_val, skew = skew, exkurt = exkurt, min = min(values), max = max(values))
}

print_summary_table <- function(stats, precision = 4) {
  fmt <- function(x) ifelse(is.finite(x), sprintf(paste0("%.", precision, "f"), x), " NA    ")
  cat("         n           mean             sd             skew      ex_kurtosis              min              max\n")
  cat(sprintf("%10d %15s %15s %15s %15s %15s %15s\n",
              stats$n, fmt(stats$mean), fmt(stats$sd), fmt(stats$skew), fmt(stats$exkurt), fmt(stats$min), fmt(stats$max)))
}

autocorrelations_vec <- function(x, lags) {
  if (length(x) == 0) return(rep(NA_real_, lags))
  ac <- acf(x, plot = FALSE, lag.max = lags)$acf
  as.numeric(ac[2:(lags + 1)])
}

print_autocorr_table <- function(returns, lags, precision = 3) {
  cat(sprintf("autocorrelations (lag 1-%d)\n", lags))
  cat(sprintf("%6s %12s %12s %12s\n", "lag", "returns", "|returns|", "returns^2"))
  ac_ret <- autocorrelations_vec(returns, lags)
  ac_abs <- autocorrelations_vec(abs(returns), lags)
  ac_sq <- autocorrelations_vec(returns^2, lags)
  fmt <- function(v) ifelse(is.finite(v), sprintf("%12.3f", v), sprintf("%12s", "NA"))
  for (lag in seq_len(lags)) {
    cat(sprintf("%6d%s%s%s\n", lag, fmt(ac_ret[lag]), fmt(ac_abs[lag]), fmt(ac_sq[lag])))
  }
}

compute_aicc <- function(loglik, k, n) {
  if (n <= 0 || n - k - 1 <= 0) return(NA_real_)
  aic <- 2 * k - 2 * loglik
  aic + (2 * k * (k + 1)) / (n - k - 1)
}

compute_bic <- function(loglik, k, n) {
  if (n <= 0) return(NA_real_)
  log(n) * k - 2 * loglik
}

sorted_indices <- function(values, ascending = TRUE) {
  order(ifelse(is.finite(values), values, ifelse(ascending, Inf, -Inf)), decreasing = !ascending)
}

compute_ranks <- function(values, ascending = TRUE) {
  ord <- sorted_indices(values, ascending)
  ranks <- integer(length(values))
  ranks[ord] <- seq_along(values)
  ranks
}

print_model_table <- function(rows, aicc_vals, bic_vals, loglik_ranks, aicc_ranks, bic_ranks, precision = 6) {
  headers <- c("uncond_sd", "mu", "omega", "alpha", "beta", "gamma", "shift", "dof", "loglik", "n_params", "AICC", "BIC", "loglik_rank", "AICC_rank", "BIC_rank")
  cat("model", paste(sprintf("%20s", headers), collapse = ""), "\n", sep = "")
  fmt <- function(v) ifelse(is.finite(v), sprintf(paste0("%20.", precision, "f"), v), sprintf("%20s", "NA"))
  for (i in seq_along(rows)) {
    row <- rows[[i]]
    vals <- c(row$uncond_sd, row$mu, row$omega, row$alpha, row$beta, row$gamma, row$shift, row$dof, row$loglik)
    formatted <- paste(vapply(vals, fmt, character(1)), collapse = "")
    n_params_str <- ifelse(is.finite(row$n_params), sprintf("%20d", as.integer(row$n_params)), sprintf("%20s", "NA"))
    aicc_str <- ifelse(is.finite(aicc_vals[i]), sprintf(paste0("%20.", precision, "f"), aicc_vals[i]), sprintf("%20s", "NA"))
    bic_str <- ifelse(is.finite(bic_vals[i]), sprintf(paste0("%20.", precision, "f"), bic_vals[i]), sprintf("%20s", "NA"))
        cat(sprintf("%-20s%s%s%s%s%20d%20d%20d\n", row$label, formatted, n_params_str, aicc_str, bic_str, loglik_ranks[i], aicc_ranks[i], bic_ranks[i]))
  }
}

print_selects <- function(label, names, values) {
  ord <- order(ifelse(is.finite(values), values, Inf))
  cat(sprintf("\n%s selects\n", label))
  if (!length(ord)) {
    cat("  (no models)\n")
    return()
  }
  base <- values[ord[1]]
  for (idx in ord) {
    val <- values[idx]
    diff <- val - base
    val_str <- ifelse(is.finite(val), sprintf("%18.6f", val), sprintf("%18s", "NA"))
    diff_str <- ifelse(is.finite(diff), sprintf("%18.6f", diff), sprintf("%18s", "NA"))
    suffix <- ifelse(idx == ord[1], "  best", "")
    cat(sprintf("%-18s %s %s%s\n", names[idx], val_str, diff_str, suffix))
  }
}

residual_summary_table <- function(entries, width = 15, precision = 4) {
  cat("\nstandardized residual stats\n")
  cat(sprintf("%-18s %10s%15s%15s%15s%15s%15s%15s\n", "model", "n", "mean", "sd", "skew", "ex_kurtosis", "min", "max"))
  for (entry in entries) {
    stats <- summary_stats(entry$values)
    fmt <- function(v) ifelse(is.finite(v), sprintf(paste0("%", width, ".", precision, "f"), v), sprintf(paste0("%", width, "s"), "NA"))
    cat(sprintf("%-18s %10d%s%s%s%s%s%s\n", entry$name, stats$n, fmt(stats$mean), fmt(stats$sd), fmt(stats$skew), fmt(stats$exkurt), fmt(stats$min), fmt(stats$max)))
  }
}

cond_sd_summary_table <- function(entries, width = 15, precision = 4) {
  cat("\nconditional sd stats\n")
  cat(sprintf("%-18s %10s%15s%15s%15s%15s%15s%15s\n", "model", "n", "mean", "sd", "skew", "ex_kurtosis", "min", "max"))
  for (entry in entries) {
    stats <- summary_stats(entry$values)
    fmt <- function(v) ifelse(is.finite(v), sprintf(paste0("%", width, ".", precision, "f"), v), sprintf(paste0("%", width, "s"), "NA"))
    cat(sprintf("%-18s %10d%s%s%s%s%s%s\n", entry$name, stats$n, fmt(stats$mean), fmt(stats$sd), fmt(stats$skew), fmt(stats$exkurt), fmt(stats$min), fmt(stats$max)))
  }
}

compute_loglik_select_values <- function(rows) {
  vapply(rows, function(row) ifelse(is.finite(row$loglik), -row$loglik, Inf), numeric(1))
}

compute_metric_order <- function(rows, metric_vals) {
  ord <- order(metric_vals, na.last = TRUE)
  vapply(ord, function(idx) rows[[idx]]$label, character(1))
}

parse_args <- function() {
  defaults <- list(file = "prices.csv",
                   columns = "",
                   models = "nagarch_normal,nagarch_student_t,garch_student_t,garch_normal,gjr_student_t,gjr_normal,igarch_student_t,igarch_normal,st_student_t,st_normal,constant_vol",
                   max_columns = -1L,
                   min_rows = 250L,
                   scale = 100,
                   no_demean = FALSE,
                   no_resid_stats = FALSE,
                   cond_sd_stats = TRUE,
                   autocorr_lags = 5L)
  args <- commandArgs(trailingOnly = TRUE)
  for (arg in args) {
    if (grepl("=", arg)) {
      parts <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
      key <- parts[1]; value <- parts[2]
      if (key %in% names(defaults)) {
        defaults[[key]] <- type.convert(value, as.is = TRUE)
      }
    } else {
      key <- sub("^--", "", arg)
      if (key == "no-demean") defaults$no_demean <- TRUE else
        if (key == "no-resid-stats") defaults$no_resid_stats <- TRUE else
          if (key == "no-cond-sd-stats") defaults$cond_sd_stats <- FALSE else
            if (key == "cond-sd-stats") defaults$cond_sd_stats <- TRUE
    }
  }
  defaults
}

main <- function() {
  args <- parse_args()
  if (args$scale <= 0) stop("--scale must be positive")
  raw <- tryCatch(read.csv(args$file, stringsAsFactors = FALSE), error = function(e) stop(sprintf("failed to read %s: %s", args$file, e$message)))
  if (ncol(raw) < 2) stop("expecting at least one price column")
  date_col <- suppressWarnings(as.POSIXct(raw[[1]], tz = "UTC"))
  prices <- raw[-1]
  colnames(prices) <- trimws(colnames(prices))
  n_rows <- nrow(prices); n_cols <- ncol(prices)
  cat(sprintf("loaded price data from %s with %d rows and %d columns\n", args$file, n_rows, n_cols))
  if (any(!is.na(date_col))) {
    non_na <- date_col[!is.na(date_col)]
    if (length(non_na) > 0) {
      cat(sprintf("date range: %s to %s\n", format(non_na[1], "%Y-%m-%d"), format(non_na[length(non_na)], "%Y-%m-%d")))
    }
  }
  cat(sprintf("return scaling factor: %.1f\n", args$scale))
  cat(sprintf("demean returns: %s\n", ifelse(args$no_demean, "no", "yes")))
  requested <- trimws(unlist(strsplit(args$models, ",")))
  requested <- requested[nzchar(requested)]
  allowed <- c("nagarch_normal","nagarch_student_t","garch_normal","garch_student_t","gjr_normal","gjr_student_t","igarch_normal","igarch_student_t","st_normal","st_student_t","constant_vol")
  for (name in requested) {
    if (!(name %in% allowed)) stop(sprintf("unknown model: %s", name))
  }
  columns <- colnames(prices)
  if (nzchar(args$columns)) {
    subset <- trimws(unlist(strsplit(args$columns, ",")))
    subset <- subset[nzchar(subset)]
    for (name in subset) {
      if (!(name %in% columns)) stop(sprintf("unknown column %s", name))
    }
    columns <- unique(subset)
  }
  if (args$max_columns > 0 && length(columns) > args$max_columns) {
    columns <- columns[seq_len(args$max_columns)]
    cat(sprintf("limiting to first %d column(s)\n", args$max_columns))
  }
  summaries <- list()
  for (column in columns) {
    series <- prices[[column]]
    series <- series[!is.na(series)]
    price_obs <- length(series)
    returns <- diff(log(series)) * args$scale
    stats <- summary_stats(returns)
    cat(sprintf("\n==== column: %s ====\n", column))
    cat(sprintf("price observations: %d, log returns used: %d\n", price_obs, length(returns)))
    print_summary_table(stats)
    cat("\n")
    print_autocorr_table(returns, args$autocorr_lags)
    adj_returns <- if (args$no_demean) returns else returns - mean(returns)
    if (length(adj_returns) < args$min_rows) {
      cat(sprintf("not enough data (need %d)\n", args$min_rows))
      next
    }
    model_rows <- list()
    resid_entries <- list()
    cond_sd_entries <- list()
    for (model in requested) {
      dist <- if (grepl("student_t$", model)) "student-t" else "normal"
      if (grepl("^st", model)) {
        cat(sprintf("skipping %s (ST-GARCH not supported in R script)\n", model))
        next
      }
      row <- tryCatch({
        if (model == "constant_vol") {
          fit_constant_vol(adj_returns)
        } else if (grepl("^garch", model)) {
          fit_rugarch_model(adj_returns, "garch", dist)
        } else if (grepl("^gjr", model)) {
          fit_rugarch_model(adj_returns, "gjr", dist)
        } else if (grepl("^igarch", model)) {
          fit_igarch_model(adj_returns, dist)
        } else if (grepl("^nagarch", model)) {
          fit_nagarch_model(adj_returns, dist)
        } else {
          stop(sprintf("unknown model: %s", model))
        }
      }, error = function(e) {
        cat(sprintf("\nfailed to fit %s: %s\n", model, conditionMessage(e)))
        NULL
      })
      if (is.null(row)) {
        next
      }
      model_rows[[length(model_rows) + 1]] <- row
      resid_entries[[length(resid_entries) + 1]] <- list(name = row$label, values = row$std_resid)
      if (!is.null(row$cond_sd)) cond_sd_entries[[length(cond_sd_entries) + 1]] <- list(name = row$label, values = row$cond_sd)
    }
    if (!length(model_rows)) {
      cat("no models fitted\n")
      next
    }
    n_obs <- length(adj_returns)
    loglik_vals <- sapply(model_rows, function(row) row$loglik)
    aicc_vals <- vapply(seq_along(model_rows), function(i) compute_aicc(model_rows[[i]]$loglik, model_rows[[i]]$n_params, n_obs), numeric(1))
    bic_vals <- vapply(seq_along(model_rows), function(i) compute_bic(model_rows[[i]]$loglik, model_rows[[i]]$n_params, n_obs), numeric(1))
    loglik_ranks <- compute_ranks(loglik_vals, ascending = FALSE)
    aicc_ranks <- compute_ranks(aicc_vals, ascending = TRUE)
    bic_ranks <- compute_ranks(bic_vals, ascending = TRUE)
    cat("\n")
    print_model_table(model_rows, aicc_vals, bic_vals, loglik_ranks, aicc_ranks, bic_ranks)
    names_vec <- vapply(model_rows, function(row) row$label, character(1))
    loglik_select <- compute_loglik_select_values(model_rows)
    print_selects("loglik", names_vec, loglik_select)
    print_selects("AICC", names_vec, aicc_vals)
    print_selects("BIC", names_vec, bic_vals)
    if (!args$no_resid_stats) {
      residual_summary_table(resid_entries)
    }
    if (args$cond_sd_stats && length(cond_sd_entries)) {
      cond_sd_summary_table(cond_sd_entries)
    }
    summaries[[length(summaries) + 1]] <- list(name = column,
                                               aicc_order = compute_metric_order(model_rows, aicc_vals),
                                               bic_order = compute_metric_order(model_rows, bic_vals))
  }
  cat("\n==== summary ====\n")
  if (!length(summaries)) {
    cat("no fitted models to summarize\n")
  } else {
    for (entry in summaries) {
      cat(sprintf("%s AICC: %s\n", entry$name, paste(entry$aicc_order, collapse = ", ")))
      cat(sprintf("%s BIC: %s\n", entry$name, paste(entry$bic_order, collapse = ", ")))
    }
  }
}

main()
