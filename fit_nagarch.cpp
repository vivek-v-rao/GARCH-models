// fit_nagarch.cpp
#include "fit_nagarch.h"

#include "nelder_mead.h"
#include "constants.h"
#include "param_maps.h"
#include "util.h"
#include "vol_models.h"

#include <algorithm>
#include <cmath>

namespace vol {

static double obj_theta_nagarch(const std::vector<double>& th,
				const std::vector<double>& r,
				const std::string& dist) {
	if (dist == "normal") {
		if ((int)th.size() != 5) return kHugePenalty;
		for (int i = 0; i < 5; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
		NagarchParams p = theta_to_nagarch_params(th);
		return neg_loglik_nagarch(r, p, dist, 0.0);
	} else if (dist == "student-t") {
		if ((int)th.size() != 6) return kHugePenalty;
		for (int i = 0; i < 6; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
		NagarchParams p = theta_to_nagarch_params(th);
		const double dof = theta_to_dof(th[5]);
		return neg_loglik_nagarch(r, p, dist, dof);
	}
    return kHugePenalty;
}

static bool warm_start_theta(const NagarchParams& p,
			   double dof,
			   const std::string& dist,
			   std::vector<double>* th_out) {
	if (!(p.omega > 0.0)) return false;
	const double gamma_sq = p.gamma * p.gamma;
	double u = p.alpha * (1.0 + gamma_sq) + p.beta;
	const double eps = 1e-6;
	if (!(u > eps && u < 1.0 - eps)) return false;
	const double denom = u;
	double v = (denom > eps) ? (p.beta / denom) : 0.5;
	u = std::clamp(u, eps, 1.0 - eps);
	v = std::clamp(v, eps, 1.0 - eps);
	const double omega_log = std::log(std::max(p.omega, 1e-12));

	if (dist == "student-t") {
		if (!(dof > 2.0001)) return false;
		th_out->assign(6, 0.0);
		(*th_out)[5] = std::log(std::max(dof - 2.0, 1e-8));
	} else {
		th_out->assign(5, 0.0);
	}
	(*th_out)[0] = p.mu;
	(*th_out)[1] = omega_log;
	(*th_out)[2] = logit(u);
	(*th_out)[3] = logit(v);
	(*th_out)[4] = p.gamma;
	return true;
}

NagarchFit fit_nagarch(const std::vector<double>& r,
		       const std::string& dist,
		       double dof0,
		       const NagarchFit* warm_start) {
	const long long n = (long long)r.size();
	if (n <= 5) die("fit_nagarch: need more data");

  // mean/var
	double mean = 0.0;
	for (long long i = 0; i < n; ++i) mean += r[i];
	mean /= (double)n;

	double var = 0.0;
	for (long long i = 0; i < n; ++i) {
		double d = r[i] - mean;
		var += d * d;
	}
	var /= (double)n;
	if (!(var > 0.0)) var = 1e-8;

  // initial guesses
	const double u0 = 0.97;
	const double v0 = 0.50;
	const double g0 = 0.00;
	double omega0 = var * (1.0 - u0);
	if (!(omega0 > 0.0)) omega0 = 1e-10;

	const double sd = std::sqrt(var);

	if (dist == "normal") {
		std::vector<double> th0;
		std::vector<double> step(5, 0.0);
		step[0] = 0.10 * sd;
		step[1] = 0.50;
		step[2] = 0.50;
		step[3] = 0.50;
		step[4] = 0.20;

		std::vector<double> warm_theta;
		if (warm_start && warm_start_theta(warm_start->p, warm_start->dof, dist, &warm_theta)) {
			th0 = warm_theta;
		} else {
			th0.assign(5, 0.0);
		th0[0] = mean;
		th0[1] = std::log(omega0);
		th0[2] = logit(u0);
		th0[3] = logit(v0);
		th0[4] = g0;
		}

		auto f = [&](const std::vector<double>& th) -> double { return obj_theta_nagarch(th, r, dist); };
		std::vector<double> th = nelder_mead_minimize(th0, step, f, 3000, 1e-8);

		NagarchFit fit;
		fit.p = theta_to_nagarch_params(th);
		fit.dof = 0.0;
		return fit;
	}

	if (dist == "student-t") {
		if (!(dof0 > 2.0001)) dof0 = 6.0;

		std::vector<double> th0;
		std::vector<double> step(6, 0.0);
		step[0] = 0.10 * sd;
		step[1] = 0.50;
		step[2] = 0.50;
		step[3] = 0.50;
		step[4] = 0.20;
		step[5] = 0.30;

		std::vector<double> warm_theta;
		if (warm_start && warm_start_theta(warm_start->p, warm_start->dof, dist, &warm_theta)) {
			th0 = warm_theta;
		} else {
			th0.assign(6, 0.0);
		th0[0] = mean;
		th0[1] = std::log(omega0);
		th0[2] = logit(u0);
		th0[3] = logit(v0);
		th0[4] = g0;
		th0[5] = std::log(dof0 - 2.0);
		}

		auto f = [&](const std::vector<double>& th) -> double { return obj_theta_nagarch(th, r, dist); };
		std::vector<double> th = nelder_mead_minimize(th0, step, f, 4000, 1e-8);

		NagarchFit fit;
		fit.p = theta_to_nagarch_params(th);
		fit.dof = theta_to_dof(th[5]);
		return fit;
	}

	die("fit_nagarch: invalid dist");
	NagarchFit fit;
	fit.p.mu = 0.0;
	fit.p.omega = 1.0;
	fit.p.alpha = 0.0;
	fit.p.beta = 0.0;
	fit.p.gamma = 0.0;
	fit.dof = 0.0;
	return fit;
}

}  // namespace vol
