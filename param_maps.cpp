// param_maps.cpp
#include "param_maps.h"
#include "constants.h"

#include <cmath>

namespace vol {

double sigmoid(double x) {
	if (x >= 0.0) {
		double e = std::exp(-x);
		return 1.0 / (1.0 + e);
	} else {
		double e = std::exp(x);
		return e / (1.0 + e);
	}
}

double logit(double p) {
	if (p <= 0.0) return -kHugePenalty;
	if (p >= 1.0) return  kHugePenalty;
	return std::log(p / (1.0 - p));
}

NagarchParams theta_to_nagarch_params(const std::vector<double>& th) {
  // th = [mu, logomega, logit_u, logit_v, gamma]  or + [log(dof-2)]
	NagarchParams p;
	p.mu = th[0];
	p.omega = std::exp(th[1]);

	const double u = sigmoid(th[2]);  // total persistence in (0,1)
	const double v = sigmoid(th[3]);  // split in (0,1)
	p.gamma = th[4];

	p.beta = u * v;
	p.alpha = u * (1.0 - v) / (1.0 + p.gamma * p.gamma);

	return p;
}

IgarchParams theta_to_igarch_params(const std::vector<double>& th) {
  // th = [mu, logit_alpha] or + [log(dof-2)]
	IgarchParams p;
	p.mu = th[0];
	const double alpha_raw = sigmoid(th[1]);
	const double eps = 1e-6;
	p.alpha = eps + (1.0 - 2.0 * eps) * alpha_raw;
	p.beta = 1.0 - p.alpha;
	return p;
}


GarchParams theta_to_garch_params(const std::vector<double>& th) {
  // th = [mu, logomega, logit_u, logit_v] or + [log(dof-2)]
	GarchParams p;
	p.mu = th[0];
	p.omega = std::exp(th[1]);

	const double u = sigmoid(th[2]);
	const double v = sigmoid(th[3]);

	p.beta = u * v;
	p.alpha = u * (1.0 - v);

	return p;
}

GjrParams theta_to_gjr_params(const std::vector<double>& th) {
  // th = [mu, logomega, logit_u, logit_v, logit_w] (+ optional dof)
	GjrParams p;
	p.mu = th[0];
	p.omega = std::exp(th[1]);

	const double u = sigmoid(th[2]);
	const double v = sigmoid(th[3]);
	const double w = sigmoid(th[4]);

	const double beta_share = u * v;
	const double remainder = u * (1.0 - v);
	const double gamma_half = remainder * w;

	p.beta = beta_share;
	p.alpha = remainder * (1.0 - w);
	p.gamma = 2.0 * gamma_half;

	return p;
}

StParams theta_to_st_params(const std::vector<double>& th) {
  // th = [mu, logomega, logit_u, logit_v, logit_w, shift] (+ optional dof)
	StParams p;
	p.mu = th[0];
	p.omega = std::exp(th[1]);

	const double u = sigmoid(th[2]);
	const double v = sigmoid(th[3]);
	const double w = sigmoid(th[4]);

	const double remainder = u * (1.0 - v);
	const double beta_share = u * v;
	const double gamma_half = remainder * w;

	p.beta = beta_share;
	p.alpha = remainder * (1.0 - w);
	p.gamma = 2.0 * gamma_half;
	p.shift = th[5];

	return p;
}

double theta_to_dof(double th_dof) {
	return 2.0 + std::exp(th_dof);
}

}  // namespace vol
