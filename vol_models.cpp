// vol_models.cpp
#include "vol_models.h"

#include "constants.h"
#include "dist.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vol {

static const double kPi = 3.14159265358979323846;

namespace {

double initial_variance(const std::vector<double>& r, double mu) {
	if (r.empty()) return 1e-6;
	double sum = 0.0;
	for (double value : r) {
		const double eps = value - mu;
		sum += eps * eps;
	}
	double var = sum / static_cast<double>(r.size());
	if (!(var > 1e-12)) var = 1e-6;
	return var;
}
struct LikelihoodPartials {
	double d_eps;
	double d_h;
	double d_dof;
};

double digamma(double x) {
	double result = 0.0;
	while (x < 6.0) {
		result -= 1.0 / x;
		x += 1.0;
	}
	const double inv = 1.0 / x;
	const double inv2 = inv * inv;
	const double inv4 = inv2 * inv2;
	const double inv6 = inv4 * inv2;
	result += std::log(x) - 0.5 * inv - (1.0 / 12.0) * inv2 + (1.0 / 120.0) * inv4 - (1.0 / 252.0) * inv6;
	return result;
}

LikelihoodPartials normal_partials(double eps, double h) {
	const double inv_h = 1.0 / h;
	LikelihoodPartials p{};
	p.d_eps = eps * inv_h;
	p.d_h = 0.5 * (inv_h - (eps * eps) * (inv_h * inv_h));
	p.d_dof = 0.0;
	return p;
}

LikelihoodPartials student_partials(double eps, double h, double dof) {
	LikelihoodPartials p{};
	const double sqrt_h = std::sqrt(h);
	const double z = eps / sqrt_h;
	const double scale = standardized_t_scale(dof);
	const double x = scale * z;
	const double dlog_pdf_dx = -(dof + 1.0) * x / (dof + x * x);
	p.d_eps = -dlog_pdf_dx * scale / sqrt_h;
	p.d_h = 0.5 / h + dlog_pdf_dx * scale * eps / (2.0 * h * sqrt_h);
	const double log_term = std::log(1.0 + (x * x) / dof);
	const double dlog_pdf_dnu = 0.5 * digamma(0.5 * (dof + 1.0)) - 0.5 * digamma(0.5 * dof) - 0.5 / dof - 0.5 * log_term + 0.5 * (dof + 1.0) * (x * x) / (dof * (dof + x * x));
	const double dlog_scale_dnu = -1.0 / (dof * (dof - 2.0));
	const double dlog_fz_dnu = dlog_pdf_dnu + dlog_pdf_dx * x * dlog_scale_dnu + dlog_scale_dnu;
	p.d_dof = -dlog_fz_dnu;
	return p;
}

}  // namespace

double neg_loglik_nagarch(const std::vector<double>& r,
			  const NagarchParams& p,
			  const std::string& dist,
			  double dof) {
	const long long n = (long long)r.size();
	if (n <= 0) return kHugePenalty;
	if (!(p.omega > 0.0)) return kHugePenalty;
	if (p.alpha < 0.0) return kHugePenalty;
	if (p.beta < 0.0) return kHugePenalty;

	const double u = p.alpha * (1.0 + p.gamma * p.gamma) + p.beta;
	if (!(u < 1.0)) return kHugePenalty;

	if (dist == "student-t") {
		if (!(dof > 2.0001)) return kHugePenalty;
	} else if (dist == "normal") {
    // ok
	} else {
		return kHugePenalty;
	}

	double h = p.omega / (1.0 - u);
	if (!(h > 0.0) || !std::isfinite(h)) return kHugePenalty;

	double nll = 0.0;
	const double log2pi = std::log(2.0 * kPi);

	for (long long t = 0; t < n; ++t) {
		const double eps = r[t] - p.mu;
		if (!(h > 0.0) || !std::isfinite(h)) return kHugePenalty;

		const double z = eps / std::sqrt(h);

		if (dist == "normal") {
			nll += 0.5 * (log2pi + std::log(h) + (eps * eps) / h);
		} else {
			const double nu = dof;
			const double scale = standardized_t_scale(nu);
			const double x = scale * z;
			const double log_fz = log_t_pdf(x, nu) + std::log(scale);
			const double log_feps = log_fz - 0.5 * std::log(h);
			nll += -log_feps;
		}

		const double shock = eps - p.gamma * std::sqrt(h);
		const double h_next = p.omega + p.alpha * (shock * shock) + p.beta * h;
		h = h_next;
	}

	if (!std::isfinite(nll)) return kHugePenalty;
	return nll;
}

double neg_loglik_nagarch_grad(const std::vector<double>& r,
		const NagarchParams& p,
		const std::string& dist,
		double dof,
		std::vector<double>* grad) {
	const double base = neg_loglik_nagarch(r, p, dist, dof);
	if (!grad) return base;
	const bool has_dof = (dist == "student-t");
	grad->assign(has_dof ? 6 : 5, 0.0);
	if (!std::isfinite(base) || base >= 0.5 * kHugePenalty) {
		return base;
	}
	const long long n = static_cast<long long>(r.size());
	if (n <= 0) return base;

	const double gamma_sq = p.gamma * p.gamma;
	const double u = p.alpha * (1.0 + gamma_sq) + p.beta;
	const double denom = 1.0 - u;
	double h = p.omega / denom;
	if (!(h > 0.0)) return base;

	double dh_mu = 0.0;
	double dh_omega = 1.0 / denom;
	double dh_alpha = p.omega * (1.0 + gamma_sq) / (denom * denom);
	double dh_beta = p.omega / (denom * denom);
	double dh_gamma = p.omega * (2.0 * p.alpha * p.gamma) / (denom * denom);

	double g_mu = 0.0;
	double g_omega = 0.0;
	double g_alpha = 0.0;
	double g_beta = 0.0;
	double g_gamma = 0.0;
	double g_dof = 0.0;

	for (long long t = 0; t < n; ++t) {
		const double eps = r[t] - p.mu;
		if (!(h > 0.0) || !std::isfinite(h)) return base;

		const LikelihoodPartials partials =
				(dist == "normal") ? normal_partials(eps, h)
									: student_partials(eps, h, dof);
		g_mu += partials.d_eps * (-1.0) + partials.d_h * dh_mu;
		g_omega += partials.d_h * dh_omega;
		g_alpha += partials.d_h * dh_alpha;
		g_beta += partials.d_h * dh_beta;
		g_gamma += partials.d_h * dh_gamma;
		if (has_dof) g_dof += partials.d_dof;

		const double sqrt_h_cur = std::sqrt(h);
		const double shock = eps - p.gamma * sqrt_h_cur;
		const double shock_sq = shock * shock;

		const auto advance = [&](double& deriv,
								 double domega,
								 double dalpha,
								 double dbeta,
								 double dgamma,
								 double deps) {
			const double prev = deriv;
			const double dshock =
					deps - sqrt_h_cur * dgamma - p.gamma * (0.5 / sqrt_h_cur) * prev;
			deriv = domega + dalpha * shock_sq +
					p.alpha * 2.0 * shock * dshock +
					dbeta * h + p.beta * prev;
		};

		advance(dh_mu, 0.0, 0.0, 0.0, 0.0, -1.0);
		advance(dh_omega, 1.0, 0.0, 0.0, 0.0, 0.0);
		advance(dh_alpha, 0.0, 1.0, 0.0, 0.0, 0.0);
		advance(dh_beta, 0.0, 0.0, 1.0, 0.0, 0.0);
		advance(dh_gamma, 0.0, 0.0, 0.0, 1.0, 0.0);

		const double h_next = p.omega + p.alpha * shock_sq + p.beta * h;
		h = h_next;
	}

	(*grad)[0] = g_mu;
	(*grad)[1] = g_omega;
	(*grad)[2] = g_alpha;
	(*grad)[3] = g_beta;
	(*grad)[4] = g_gamma;
	if (has_dof) (*grad)[5] = g_dof;
	return base;
}
double neg_loglik_garch(const std::vector<double>& r,
			const GarchParams& p,
			const std::string& dist,
			double dof) {
	const long long n = (long long)r.size();
	if (n <= 0) return kHugePenalty;
	if (!(p.omega > 0.0)) return kHugePenalty;
	if (p.alpha < 0.0) return kHugePenalty;
	if (p.beta < 0.0) return kHugePenalty;

	const double u = p.alpha + p.beta;
	if (!(u < 1.0)) return kHugePenalty;

	if (dist == "student-t") {
		if (!(dof > 2.0001)) return kHugePenalty;
	} else if (dist == "normal") {
    // ok
	} else {
		return kHugePenalty;
	}

	double h = p.omega / (1.0 - u);
	if (!(h > 0.0) || !std::isfinite(h)) return kHugePenalty;

	double nll = 0.0;
	const double log2pi = std::log(2.0 * kPi);

	for (long long t = 0; t < n; ++t) {
		const double eps = r[t] - p.mu;
		if (!(h > 0.0) || !std::isfinite(h)) return kHugePenalty;

		const double z = eps / std::sqrt(h);

		if (dist == "normal") {
			nll += 0.5 * (log2pi + std::log(h) + (eps * eps) / h);
		} else {
			const double nu = dof;
			const double scale = standardized_t_scale(nu);
			const double x = scale * z;
			const double log_fz = log_t_pdf(x, nu) + std::log(scale);
			const double log_feps = log_fz - 0.5 * std::log(h);
			nll += -log_feps;
		}

		const double h_next = p.omega + p.alpha * (eps * eps) + p.beta * h;
		h = h_next;
	}

	if (!std::isfinite(nll)) return kHugePenalty;
	return nll;
}

double neg_loglik_garch_grad(const std::vector<double>& r,
		const GarchParams& p,
		const std::string& dist,
		double dof,
		std::vector<double>* grad) {
	const double base = neg_loglik_garch(r, p, dist, dof);
	if (!grad) return base;
	const bool has_dof = (dist == "student-t");
	grad->assign(has_dof ? 5 : 4, 0.0);
	if (!std::isfinite(base) || base >= 0.5 * kHugePenalty) {
		return base;
	}
	const long long n = static_cast<long long>(r.size());
	if (n <= 0) return base;

	const double u = p.alpha + p.beta;
	const double denom = 1.0 - u;
	double h = p.omega / denom;
	if (!(h > 0.0)) return base;

	double dh_mu = 0.0;
	double dh_omega = 1.0 / denom;
	double dh_alpha = p.omega / (denom * denom);
	double dh_beta = dh_alpha;

	double g_mu = 0.0;
	double g_omega = 0.0;
	double g_alpha = 0.0;
	double g_beta = 0.0;
	double g_dof = 0.0;

	for (long long t = 0; t < n; ++t) {
		const double eps = r[t] - p.mu;
		if (!(h > 0.0) || !std::isfinite(h)) return base;

		const LikelihoodPartials partials =
				(dist == "normal") ? normal_partials(eps, h)
									: student_partials(eps, h, dof);
		g_mu += partials.d_eps * (-1.0) + partials.d_h * dh_mu;
		g_omega += partials.d_h * dh_omega;
		g_alpha += partials.d_h * dh_alpha;
		g_beta += partials.d_h * dh_beta;
		if (has_dof) g_dof += partials.d_dof;

		const double h_prev = h;
		const double eps_sq = eps * eps;

		const double dh_mu_next = p.alpha * 2.0 * eps * (-1.0) + p.beta * dh_mu;
		const double dh_omega_next = 1.0 + p.beta * dh_omega;
		const double dh_alpha_next = eps_sq + p.beta * dh_alpha;
		const double dh_beta_next = h_prev + p.beta * dh_beta;

		h = p.omega + p.alpha * eps_sq + p.beta * h_prev;
		dh_mu = dh_mu_next;
		dh_omega = dh_omega_next;
		dh_alpha = dh_alpha_next;
		dh_beta = dh_beta_next;
	}

	(*grad)[0] = g_mu;
	(*grad)[1] = g_omega;
	(*grad)[2] = g_alpha;
	(*grad)[3] = g_beta;
	if (has_dof) (*grad)[4] = g_dof;
	return base;
}
double neg_loglik_igarch(const std::vector<double>& r,
                           const IgarchParams& p,
                           const std::string& dist,
                           double dof) {
    const long long n = (long long)r.size();
    if (n <= 0) return kHugePenalty;
    if (!(p.alpha > 0.0)) return kHugePenalty;
    if (!(p.beta >= 0.0)) return kHugePenalty;
    const double sum = p.alpha + p.beta;
    if (std::abs(sum - 1.0) > 1e-6) return kHugePenalty;

    if (dist == "student-t") {
        if (!(dof > 2.0001)) return kHugePenalty;
    } else if (dist == "normal") {
        // ok
    } else {
        return kHugePenalty;
    }

    double h = initial_variance(r, p.mu);
    double nll = 0.0;
    const double log2pi = std::log(2.0 * kPi);

    for (long long t = 0; t < n; ++t) {
        const double eps = r[t] - p.mu;
        if (!(h > 0.0) || !std::isfinite(h)) return kHugePenalty;

        const double z = eps / std::sqrt(h);

        if (dist == "normal") {
            nll += 0.5 * (log2pi + std::log(h) + (eps * eps) / h);
        } else {
            const double nu = dof;
            const double scale = standardized_t_scale(nu);
            const double x = scale * z;
            const double log_fz = log_t_pdf(x, nu) + std::log(scale);
            const double log_feps = log_fz - 0.5 * std::log(h);
            nll += -log_feps;
        }

        const double h_next = p.alpha * (eps * eps) + p.beta * h;
        h = h_next;
    }

    if (!std::isfinite(nll)) return kHugePenalty;
    return nll;
}


double neg_loglik_igarch_grad(const std::vector<double>& r,
		const IgarchParams& p,
		const std::string& dist,
		double dof,
		std::vector<double>* grad) {
	const double base = neg_loglik_igarch(r, p, dist, dof);
	if (!grad) return base;
	const bool has_dof = (dist == "student-t");
	grad->assign(has_dof ? 3 : 2, 0.0);
	if (!std::isfinite(base) || base >= 0.5 * kHugePenalty) {
		return base;
	}
	const long long n = static_cast<long long>(r.size());
	if (n <= 0) return base;

	double h = initial_variance(r, p.mu);
	if (!(h > 0.0)) return base;

	double sum_eps = 0.0;
	for (double value : r) sum_eps += value - p.mu;
	double dh_mu = (-2.0 / static_cast<double>(n)) * sum_eps;
	double dh_alpha = 0.0;

	double g_mu = 0.0;
	double g_alpha = 0.0;
	double g_dof = 0.0;

	for (long long t = 0; t < n; ++t) {
		const double eps = r[t] - p.mu;
		if (!(h > 0.0) || !std::isfinite(h)) return base;

		const LikelihoodPartials partials =
				(dist == "normal") ? normal_partials(eps, h)
									: student_partials(eps, h, dof);
		g_mu += partials.d_eps * (-1.0) + partials.d_h * dh_mu;
		g_alpha += partials.d_h * dh_alpha;
		if (has_dof) g_dof += partials.d_dof;

		const double h_prev = h;
		const double eps_sq = eps * eps;

		const double dh_mu_next = p.alpha * 2.0 * eps * (-1.0) + p.beta * dh_mu;
		const double dh_alpha_next = eps_sq - h_prev + p.beta * dh_alpha;

		h = p.alpha * eps_sq + p.beta * h_prev;
		dh_mu = dh_mu_next;
		dh_alpha = dh_alpha_next;
	}

	(*grad)[0] = g_mu;
	(*grad)[1] = g_alpha;
	if (has_dof) (*grad)[2] = g_dof;
	return base;
}
double neg_loglik_gjr(const std::vector<double>& r,
		       const GjrParams& p,
		       const std::string& dist,
		       double dof) {
	const long long n = static_cast<long long>(r.size());
	if (n <= 0) return kHugePenalty;
	if (!(p.omega > 0.0)) return kHugePenalty;
	if (p.alpha < 0.0) return kHugePenalty;
	if (p.gamma < 0.0) return kHugePenalty;
	if (p.beta < 0.0) return kHugePenalty;

	const double u = p.alpha + 0.5 * p.gamma + p.beta;
	if (!(u < 1.0)) return kHugePenalty;

	if (dist == "student-t") {
		if (!(dof > 2.0001)) return kHugePenalty;
	} else if (dist == "normal") {
    // ok
	} else {
		return kHugePenalty;
	}

	double h = p.omega / (1.0 - u);
	if (!(h > 0.0) || !std::isfinite(h)) return kHugePenalty;

	double nll = 0.0;
	const double log2pi = std::log(2.0 * kPi);

	for (long long t = 0; t < n; ++t) {
		const double eps = r[t] - p.mu;
		if (!(h > 0.0) || !std::isfinite(h)) return kHugePenalty;

		const double z = eps / std::sqrt(h);

		if (dist == "normal") {
			nll += 0.5 * (log2pi + std::log(h) + (eps * eps) / h);
		} else {
			const double nu = dof;
			const double scale = standardized_t_scale(nu);
			const double x = scale * z;
			const double log_fz = log_t_pdf(x, nu) + std::log(scale);
			const double log_feps = log_fz - 0.5 * std::log(h);
			nll += -log_feps;
		}

		const double indicator = (eps < 0.0) ? 1.0 : 0.0;
		const double h_next = p.omega + (p.alpha + p.gamma * indicator) * (eps * eps) + p.beta * h;
		h = h_next;
	}

	if (!std::isfinite(nll)) return kHugePenalty;
	return nll;
}

double neg_loglik_gjr_grad(const std::vector<double>& r,
		const GjrParams& p,
		const std::string& dist,
		double dof,
		std::vector<double>* grad) {
	const double base = neg_loglik_gjr(r, p, dist, dof);
	if (!grad) return base;
	const bool has_dof = (dist == "student-t");
	grad->assign(has_dof ? 6 : 5, 0.0);
	if (!std::isfinite(base) || base >= 0.5 * kHugePenalty) {
		return base;
	}
	const long long n = static_cast<long long>(r.size());
	if (n <= 0) return base;

	const double u = p.alpha + 0.5 * p.gamma + p.beta;
	const double denom = 1.0 - u;
	double h = p.omega / denom;
	if (!(h > 0.0)) return base;

	double dh_mu = 0.0;
	double dh_omega = 1.0 / denom;
	double dh_alpha = p.omega / (denom * denom);
	double dh_beta = p.omega / (denom * denom);
	double dh_gamma = p.omega * 0.5 / (denom * denom);

	double g_mu = 0.0;
	double g_omega = 0.0;
	double g_alpha = 0.0;
	double g_beta = 0.0;
	double g_gamma = 0.0;
	double g_dof = 0.0;

	for (long long t = 0; t < n; ++t) {
		const double eps = r[t] - p.mu;
		if (!(h > 0.0) || !std::isfinite(h)) return base;

		const double indicator = (eps < 0.0) ? 1.0 : 0.0;
		const LikelihoodPartials partials =
				(dist == "normal") ? normal_partials(eps, h)
									: student_partials(eps, h, dof);
		g_mu += partials.d_eps * (-1.0) + partials.d_h * dh_mu;
		g_omega += partials.d_h * dh_omega;
		g_alpha += partials.d_h * dh_alpha;
		g_beta += partials.d_h * dh_beta;
		g_gamma += partials.d_h * dh_gamma;
		if (has_dof) g_dof += partials.d_dof;

		const double h_prev = h;
		const double eps_sq = eps * eps;
		const double coeff = p.alpha + p.gamma * indicator;

		const double dh_mu_next = coeff * 2.0 * eps * (-1.0) + p.beta * dh_mu;
		const double dh_omega_next = 1.0 + p.beta * dh_omega;
		const double dh_alpha_next = eps_sq + p.beta * dh_alpha;
		const double dh_beta_next = h_prev + p.beta * dh_beta;
		const double dh_gamma_next = indicator * eps_sq + p.beta * dh_gamma;

		h = p.omega + coeff * eps_sq + p.beta * h_prev;
		dh_mu = dh_mu_next;
		dh_omega = dh_omega_next;
		dh_alpha = dh_alpha_next;
		dh_beta = dh_beta_next;
		dh_gamma = dh_gamma_next;
	}

	(*grad)[0] = g_mu;
	(*grad)[1] = g_omega;
	(*grad)[2] = g_alpha;
	(*grad)[3] = g_beta;
	(*grad)[4] = g_gamma;
	if (has_dof) (*grad)[5] = g_dof;
	return base;
}
double neg_loglik_st(const std::vector<double>& r,
            const StParams& p,
            const std::string& dist,
            double dof) {
	const long long n = static_cast<long long>(r.size());
	if (n <= 0) return kHugePenalty;
	if (!(p.omega > 0.0)) return kHugePenalty;
	if (p.alpha < 0.0) return kHugePenalty;
	if (p.gamma < 0.0) return kHugePenalty;
	if (p.beta < 0.0) return kHugePenalty;

	const double u = p.alpha + 0.5 * p.gamma + p.beta;
	if (!(u < 1.0)) return kHugePenalty;

	if (dist == "student-t") {
		if (!(dof > 2.0001)) return kHugePenalty;
	} else if (dist == "normal") {
    // ok
	} else {
		return kHugePenalty;
	}

	double h = p.omega / (1.0 - u);
	if (!(h > 0.0) || !std::isfinite(h)) return kHugePenalty;

	double nll = 0.0;
	const double log2pi = std::log(2.0 * kPi);

	for (long long t = 0; t < n; ++t) {
		const double eps = r[t] - p.mu;
		if (!(h > 0.0) || !std::isfinite(h)) return kHugePenalty;

		const double z = eps / std::sqrt(h);

		if (dist == "normal") {
			nll += 0.5 * (log2pi + std::log(h) + (eps * eps) / h);
		} else {
			const double nu = dof;
			const double scale = standardized_t_scale(nu);
			const double x = scale * z;
			const double log_fz = log_t_pdf(x, nu) + std::log(scale);
			const double log_feps = log_fz - 0.5 * std::log(h);
			nll += -log_feps;
		}

		const double shock = eps - p.shift * std::sqrt(h);
		const double indicator = (eps < 0.0) ? 1.0 : 0.0;
		const double scaled = (p.alpha + p.gamma * indicator) * (shock * shock);
		const double h_next = p.omega + scaled + p.beta * h;
		h = h_next;
	}

	if (!std::isfinite(nll)) return kHugePenalty;
	return nll;
}

double neg_loglik_st_grad(const std::vector<double>& r,
		const StParams& p,
		const std::string& dist,
		double dof,
		std::vector<double>* grad) {
	const double base = neg_loglik_st(r, p, dist, dof);
	if (!grad) return base;
	const bool has_dof = (dist == "student-t");
	grad->assign(has_dof ? 7 : 6, 0.0);
	if (!std::isfinite(base) || base >= 0.5 * kHugePenalty) {
		return base;
	}
	const long long n = static_cast<long long>(r.size());
	if (n <= 0) return base;

	const double u = p.alpha + 0.5 * p.gamma + p.beta;
	const double denom = 1.0 - u;
	double h = p.omega / denom;
	if (!(h > 0.0)) return base;

	double dh_mu = 0.0;
	double dh_omega = 1.0 / denom;
	double dh_alpha = p.omega / (denom * denom);
	double dh_beta = p.omega / (denom * denom);
	double dh_gamma = p.omega * 0.5 / (denom * denom);
	double dh_shift = 0.0;

	double g_mu = 0.0;
	double g_omega = 0.0;
	double g_alpha = 0.0;
	double g_beta = 0.0;
	double g_gamma = 0.0;
	double g_shift = 0.0;
	double g_dof = 0.0;

	for (long long t = 0; t < n; ++t) {
		const double eps = r[t] - p.mu;
		if (!(h > 0.0) || !std::isfinite(h)) return base;

		const double indicator = (eps < 0.0) ? 1.0 : 0.0;
		const LikelihoodPartials partials =
				(dist == "normal") ? normal_partials(eps, h)
									: student_partials(eps, h, dof);
		g_mu += partials.d_eps * (-1.0) + partials.d_h * dh_mu;
		g_omega += partials.d_h * dh_omega;
		g_alpha += partials.d_h * dh_alpha;
		g_beta += partials.d_h * dh_beta;
		g_gamma += partials.d_h * dh_gamma;
		g_shift += partials.d_h * dh_shift;
		if (has_dof) g_dof += partials.d_dof;

		const double sqrt_h_cur = std::sqrt(h);
		const double shock = eps - p.shift * sqrt_h_cur;
		const double shock_sq = shock * shock;
		const double scaled = p.alpha + p.gamma * indicator;

		const auto advance = [&](double& deriv,
								 double domega,
								 double dalpha,
								 double dbeta,
								 double dgamma,
								 double dshift,
								 double deps) {
			const double prev = deriv;
			const double dscale = dalpha + indicator * dgamma;
			const double dshock =
					deps - sqrt_h_cur * dshift - p.shift * (0.5 / sqrt_h_cur) * prev;
			const double dscaled = dscale * shock_sq + scaled * 2.0 * shock * dshock;
			deriv = domega + dscaled + dbeta * h + p.beta * prev;
		};

		advance(dh_mu, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0);
		advance(dh_omega, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		advance(dh_alpha, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
		advance(dh_beta, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
		advance(dh_gamma, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
		advance(dh_shift, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

		const double h_next = p.omega + scaled * shock_sq + p.beta * h;
		h = h_next;
	}

	(*grad)[0] = g_mu;
	(*grad)[1] = g_omega;
	(*grad)[2] = g_alpha;
	(*grad)[3] = g_beta;
	(*grad)[4] = g_gamma;
	(*grad)[5] = g_shift;
	if (has_dof) (*grad)[6] = g_dof;
	return base;
}
double neg_loglik_constant(const std::vector<double>& r,
                           double mu,
                           double omega) {
    const long long n = static_cast<long long>(r.size());
    if (n <= 0) return kHugePenalty;
    if (!(omega > 0.0)) return kHugePenalty;
    double nll = 0.0;
    const double log_term = std::log(2.0 * kPi) + std::log(omega);
    for (double value : r) {
        const double eps = value - mu;
        nll += 0.5 * (log_term + (eps * eps) / omega);
    }
    if (!std::isfinite(nll)) return kHugePenalty;
    return nll;
}

double neg_loglik_constant_grad(const std::vector<double>& r,
		double mu,
		double omega,
		std::vector<double>* grad) {
	const double base = neg_loglik_constant(r, mu, omega);
	if (!grad) return base;
	grad->assign(2, 0.0);
	if (!std::isfinite(base) || base >= 0.5 * kHugePenalty) {
		return base;
	}
	const long long n = static_cast<long long>(r.size());
	if (n <= 0 || !(omega > 0.0)) return base;

	double sum_eps = 0.0;
	double sum_sq = 0.0;
	for (double value : r) {
		const double eps = value - mu;
		sum_eps += eps;
		sum_sq += eps * eps;
	}

	(*grad)[0] = -sum_eps / omega;
	(*grad)[1] = 0.5 * static_cast<double>(n) / omega - 0.5 * sum_sq / (omega * omega);
	return base;
}
}  // namespace vol

