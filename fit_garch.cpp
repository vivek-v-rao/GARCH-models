// fit_garch.cpp
#include "fit_garch.h"

#include "nelder_mead.h"
#include "constants.h"
#include "param_maps.h"
#include "util.h"
#include "vol_models.h"

#include <algorithm>
#include <cmath>

namespace vol {

static double obj_theta_garch(const std::vector<double>& th,
			      const std::vector<double>& r,
			      const std::string& dist) {
	if (dist == "normal") {
		if ((int)th.size() != 4) return kHugePenalty;
		for (int i = 0; i < 4; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
		GarchParams p = theta_to_garch_params(th);
		return neg_loglik_garch(r, p, dist, 0.0);
	} else if (dist == "student-t") {
		if ((int)th.size() != 5) return kHugePenalty;
		for (int i = 0; i < 5; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
		GarchParams p = theta_to_garch_params(th);
		const double dof = theta_to_dof(th[4]);
		return neg_loglik_garch(r, p, dist, dof);
	}
	return kHugePenalty;
}


static double obj_theta_igarch(const std::vector<double>& th,
                               const std::vector<double>& r,
                               const std::string& dist) {
    if (dist == "normal") {
        if ((int)th.size() != 2) return kHugePenalty;
        for (int i = 0; i < 2; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
        IgarchParams p = theta_to_igarch_params(th);
        return neg_loglik_igarch(r, p, dist, 0.0);
    } else if (dist == "student-t") {
        if ((int)th.size() != 3) return kHugePenalty;
        for (int i = 0; i < 3; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
        IgarchParams p = theta_to_igarch_params(th);
        const double dof = theta_to_dof(th[2]);
        return neg_loglik_igarch(r, p, dist, dof);
    }
    return kHugePenalty;
}

static double obj_theta_gjr(const std::vector<double>& th,
			    const std::vector<double>& r,
			    const std::string& dist) {
	if (dist == "normal") {
		if ((int)th.size() != 5) return kHugePenalty;
		for (int i = 0; i < 5; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
		GjrParams p = theta_to_gjr_params(th);
		return neg_loglik_gjr(r, p, dist, 0.0);
	} else if (dist == "student-t") {
		if ((int)th.size() != 6) return kHugePenalty;
		for (int i = 0; i < 6; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
		GjrParams p = theta_to_gjr_params(th);
		const double dof = theta_to_dof(th[5]);
		return neg_loglik_gjr(r, p, dist, dof);
	}
	return kHugePenalty;
}

static double obj_theta_st(const std::vector<double>& th,
			  const std::vector<double>& r,
			  const std::string& dist) {
	if (dist == "normal") {
		if ((int)th.size() != 6) return kHugePenalty;
		for (int i = 0; i < 6; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
		StParams p = theta_to_st_params(th);
		return neg_loglik_st(r, p, dist, 0.0);
	} else if (dist == "student-t") {
		if ((int)th.size() != 7) return kHugePenalty;
		for (int i = 0; i < 7; ++i) if (!std::isfinite(th[i])) return kHugePenalty;
		StParams p = theta_to_st_params(th);
		const double dof = theta_to_dof(th[6]);
		return neg_loglik_st(r, p, dist, dof);
	}
	return kHugePenalty;
}

static bool warm_start_theta_st(const StFit& warm,
			     const std::string& dist,
			     std::vector<double>* th_out) {
	if (!(warm.p.omega > 0.0)) return false;
	double u = warm.p.alpha + warm.p.gamma * 0.5 + warm.p.beta;
	const double eps = 1e-6;
	u = std::clamp(u, eps, 1.0 - eps);
	const double beta_share = warm.p.beta / u;
	double v = std::clamp(beta_share, eps, 1.0 - eps);
	const double w = std::clamp(warm.p.gamma / (2.0 * u * (1.0 - v) + eps), eps, 1.0 - eps);
	const double omega_log = std::log(std::max(warm.p.omega, 1e-12));

	if (dist == "student-t") {
		if (!(warm.dof > 2.0001)) return false;
		th_out->assign(7, 0.0);
		(*th_out)[6] = std::log(std::max(warm.dof - 2.0, 1e-8));
	} else {
		th_out->assign(6, 0.0);
	}
	(*th_out)[0] = warm.p.mu;
	(*th_out)[1] = omega_log;
	(*th_out)[2] = logit(u);
	(*th_out)[3] = logit(v);
	(*th_out)[4] = logit(w);
	(*th_out)[5] = warm.p.shift;
	return true;
}

GarchFit fit_garch(const std::vector<double>& r,
		   const std::string& dist,
		   double dof0) {
	const long long n = (long long)r.size();
	if (n <= 5) die("fit_garch: need more data");

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
	double omega0 = var * (1.0 - u0);
	if (!(omega0 > 0.0)) omega0 = 1e-10;

	const double sd = std::sqrt(var);

	if (dist == "normal") {
		std::vector<double> th0(4, 0.0);
		th0[0] = mean;
		th0[1] = std::log(omega0);
		th0[2] = logit(u0);
		th0[3] = logit(v0);

		std::vector<double> step(4, 0.0);
		step[0] = 0.10 * sd;
		step[1] = 0.50;
		step[2] = 0.50;
		step[3] = 0.50;

		auto f = [&](const std::vector<double>& th) -> double { return obj_theta_garch(th, r, dist); };
		std::vector<double> th = nelder_mead_minimize(th0, step, f, 3000, 1e-8);

		GarchFit fit;
		fit.p = theta_to_garch_params(th);
		fit.dof = 0.0;
		return fit;
	}

	if (dist == "student-t") {
		if (!(dof0 > 2.0001)) dof0 = 6.0;

		std::vector<double> th0(5, 0.0);
		th0[0] = mean;
		th0[1] = std::log(omega0);
		th0[2] = logit(u0);
		th0[3] = logit(v0);
		th0[4] = std::log(dof0 - 2.0);

		std::vector<double> step(5, 0.0);
		step[0] = 0.10 * sd;
		step[1] = 0.50;
		step[2] = 0.50;
		step[3] = 0.50;
		step[4] = 0.30;

		auto f = [&](const std::vector<double>& th) -> double { return obj_theta_garch(th, r, dist); };
		std::vector<double> th = nelder_mead_minimize(th0, step, f, 3500, 1e-8);

		GarchFit fit;
		fit.p = theta_to_garch_params(th);
		fit.dof = theta_to_dof(th[4]);
		return fit;
	}

	die("fit_garch: invalid dist");
	GarchFit fit;
	fit.p.mu = 0.0;
	fit.p.omega = 1.0;
	fit.p.alpha = 0.0;
	fit.p.beta = 0.0;
	fit.dof = 0.0;
	return fit;
}

GjrFit fit_gjr(const std::vector<double>& r,
	     const std::string& dist,
	     double dof0) {
	const long long n = static_cast<long long>(r.size());
	if (n <= 5) die("fit_gjr: need more data");

	double mean = 0.0;
	for (long long i = 0; i < n; ++i) mean += r[i];
	mean /= static_cast<double>(n);

	double var = 0.0;
	for (long long i = 0; i < n; ++i) {
		double d = r[i] - mean;
		var += d * d;
	}
	var /= static_cast<double>(n);
	if (!(var > 0.0)) var = 1e-8;

	const double u0 = 0.97;
	const double v0 = 0.50;
	const double w0 = 0.50;
	double omega0 = var * (1.0 - u0);
	if (!(omega0 > 0.0)) omega0 = 1e-10;

	const double sd = std::sqrt(var);

	if (dist == "normal") {
		std::vector<double> th0(5, 0.0);
		th0[0] = mean;
		th0[1] = std::log(omega0);
		th0[2] = logit(u0);
		th0[3] = logit(v0);
		th0[4] = logit(w0);

		std::vector<double> step(5, 0.0);
		step[0] = 0.10 * sd;
		step[1] = 0.50;
		step[2] = 0.50;
		step[3] = 0.50;
		step[4] = 0.50;

		auto f = [&](const std::vector<double>& th) -> double { return obj_theta_gjr(th, r, dist); };
		std::vector<double> th = nelder_mead_minimize(th0, step, f, 3500, 1e-8);

		GjrFit fit;
		fit.p = theta_to_gjr_params(th);
		fit.dof = 0.0;
		return fit;
	}

	if (dist == "student-t") {
		if (!(dof0 > 2.0001)) dof0 = 6.0;

		std::vector<double> th0(6, 0.0);
		th0[0] = mean;
		th0[1] = std::log(omega0);
		th0[2] = logit(u0);
		th0[3] = logit(v0);
		th0[4] = logit(w0);
		th0[5] = std::log(dof0 - 2.0);

		std::vector<double> step(6, 0.0);
		step[0] = 0.10 * sd;
		step[1] = 0.50;
		step[2] = 0.50;
		step[3] = 0.50;
		step[4] = 0.50;
		step[5] = 0.30;

		auto f = [&](const std::vector<double>& th) -> double { return obj_theta_gjr(th, r, dist); };
		std::vector<double> th = nelder_mead_minimize(th0, step, f, 4000, 1e-8);

		GjrFit fit;
		fit.p = theta_to_gjr_params(th);
		fit.dof = theta_to_dof(th[5]);
		return fit;
	}

	die("fit_gjr: invalid dist");
	GjrFit fit;
	fit.p.mu = 0.0;
	fit.p.omega = 1.0;
	fit.p.alpha = 0.0;
	fit.p.gamma = 0.0;
	fit.p.beta = 0.0;
	fit.dof = 0.0;
	return fit;
}

IgarchFit fit_igarch(const std::vector<double>& r,
         const std::string& dist,
         double dof0) {
    const long long n = static_cast<long long>(r.size());
    if (n <= 5) die("fit_igarch: need more data");

    double mean = 0.0;
    for (long long i = 0; i < n; ++i) mean += r[i];
    mean /= static_cast<double>(n);

    double var = 0.0;
    for (long long i = 0; i < n; ++i) {
        double d = r[i] - mean;
        var += d * d;
    }
    var /= static_cast<double>(n);
    if (!(var > 0.0)) var = 1e-8;
    const double sd = std::sqrt(var);
    const double alpha0 = 0.05;

    if (dist == "normal") {
        std::vector<double> th0(2, 0.0);
        std::vector<double> step(2, 0.0);
        step[0] = 0.10 * sd;
        step[1] = 0.50;
        th0[0] = mean;
        th0[1] = logit(alpha0);
        auto f = [&](const std::vector<double>& th) -> double { return obj_theta_igarch(th, r, dist); };
        std::vector<double> th = nelder_mead_minimize(th0, step, f, 3500, 1e-8);
        IgarchFit fit;
        fit.p = theta_to_igarch_params(th);
        fit.dof = 0.0;
        return fit;
    }

    if (dist == "student-t") {
        if (!(dof0 > 2.0001)) dof0 = 6.0;
        std::vector<double> th0(3, 0.0);
        std::vector<double> step(3, 0.0);
        step[0] = 0.10 * sd;
        step[1] = 0.50;
        step[2] = 0.30;
        th0[0] = mean;
        th0[1] = logit(alpha0);
        th0[2] = std::log(dof0 - 2.0);
        auto f = [&](const std::vector<double>& th) -> double { return obj_theta_igarch(th, r, dist); };
        std::vector<double> th = nelder_mead_minimize(th0, step, f, 4000, 1e-8);
        IgarchFit fit;
        fit.p = theta_to_igarch_params(th);
        fit.dof = theta_to_dof(th[2]);
        return fit;
    }

    die("fit_igarch: invalid dist");
    IgarchFit fit;
    fit.p.mu = 0.0;
    fit.p.alpha = 0.5;
    fit.p.beta = 0.5;
    fit.dof = 0.0;
    return fit;
}


StFit fit_st(const std::vector<double>& r,
   const std::string& dist,
   double dof0,
   const StFit* warm_start) {
	const long long n = static_cast<long long>(r.size());
	if (n <= 5) die("fit_st: need more data");

	double mean = 0.0;
	for (long long i = 0; i < n; ++i) mean += r[i];
	mean /= static_cast<double>(n);

	double var = 0.0;
	for (long long i = 0; i < n; ++i) {
		double d = r[i] - mean;
		var += d * d;
	}
	var /= static_cast<double>(n);
	if (!(var > 0.0)) var = 1e-8;

	const double u0 = 0.97;
	const double v0 = 0.50;
	const double w0 = 0.50;
	double omega0 = var * (1.0 - u0);
	if (!(omega0 > 0.0)) omega0 = 1e-10;

	const double sd = std::sqrt(var);
	const double shift0 = 0.0;

	if (dist == "normal") {
		std::vector<double> th0;
		std::vector<double> step(6, 0.0);
		step[0] = 0.10 * sd;
		step[1] = 0.50;
		step[2] = 0.50;
		step[3] = 0.50;
		step[4] = 0.50;
		step[5] = 0.10;

		std::vector<double> warm_theta;
		if (warm_start && warm_start_theta_st(*warm_start, dist, &warm_theta)) {
			th0 = warm_theta;
		} else {
			th0.assign(6, 0.0);
		th0[0] = mean;
		th0[1] = std::log(omega0);
		th0[2] = logit(u0);
		th0[3] = logit(v0);
		th0[4] = logit(w0);
		th0[5] = shift0;
		}

		auto f = [&](const std::vector<double>& th) -> double { return obj_theta_st(th, r, dist); };
		std::vector<double> th = nelder_mead_minimize(th0, step, f, 4000, 1e-8);

		StFit fit;
		fit.p = theta_to_st_params(th);
		fit.dof = 0.0;
		return fit;
	}

	if (dist == "student-t") {
		if (!(dof0 > 2.0001)) dof0 = 6.0;

		std::vector<double> th0;
		std::vector<double> step(7, 0.0);
		step[0] = 0.10 * sd;
		step[1] = 0.50;
		step[2] = 0.50;
		step[3] = 0.50;
		step[4] = 0.50;
		step[5] = 0.10;
		step[6] = 0.30;

		std::vector<double> warm_theta;
		if (warm_start && warm_start_theta_st(*warm_start, dist, &warm_theta)) {
			th0 = warm_theta;
		} else {
			th0.assign(7, 0.0);
		th0[0] = mean;
		th0[1] = std::log(omega0);
		th0[2] = logit(u0);
		th0[3] = logit(v0);
		th0[4] = logit(w0);
		th0[5] = shift0;
		th0[6] = std::log(dof0 - 2.0);
		}

		auto f = [&](const std::vector<double>& th) -> double { return obj_theta_st(th, r, dist); };
		std::vector<double> th = nelder_mead_minimize(th0, step, f, 4500, 1e-8);

		StFit fit;
		fit.p = theta_to_st_params(th);
		fit.dof = theta_to_dof(th[6]);
		return fit;
	}

	die("fit_st: invalid dist");
	StFit fit;
	fit.p.mu = 0.0;
	fit.p.omega = 1.0;
	fit.p.alpha = 0.0;
	fit.p.gamma = 0.0;
	fit.p.beta = 0.0;
	fit.p.shift = 0.0;
	fit.dof = 0.0;
	return fit;
}

ConstantVolFit fit_constant_vol(const std::vector<double>& r) {
    ConstantVolFit fit;
    const long long n = static_cast<long long>(r.size());
    if (n <= 0) {
        fit.omega = 1e-6;
        return fit;
    }
    double sum_sq = 0.0;
    for (double v : r) sum_sq += v * v;
    double omega = sum_sq / static_cast<double>(n);
    if (!(omega > 1e-12)) omega = 1e-6;
    fit.omega = omega;
    return fit;
}

}  // namespace vol
