// param_maps.h
#ifndef PARAM_MAPS_H
#define PARAM_MAPS_H

#include <vector>

namespace vol {

struct NagarchParams {
	double mu;
	double omega;
	double alpha;
	double beta;
	double gamma;
};

struct GarchParams {
	double mu;
	double omega;
	double alpha;
	double beta;
};


struct IgarchParams {
	double mu;
	double alpha;
	double beta;
};

struct GjrParams {
	double mu;
	double omega;
	double alpha;
	double gamma;
	double beta;
};

struct StParams {
	double mu;
	double omega;
	double alpha;
	double gamma;
	double beta;
	double shift;
};

// doc: numerically-stable logistic sigmoid mapping R -> (0,1).
double sigmoid(double x);

// doc: inverse sigmoid mapping (0,1) -> R, with saturation at endpoints.
double logit(double p);

// doc: map unconstrained optimizer vector to valid NAGARCH parameters (enforces stationarity/positivity).
NagarchParams theta_to_nagarch_params(const std::vector<double>& th);

// doc: map unconstrained optimizer vector to valid symmetric GARCH(1,1) parameters.
GarchParams theta_to_garch_params(const std::vector<double>& th);
IgarchParams theta_to_igarch_params(const std::vector<double>& th);

// doc: map unconstrained optimizer vector to valid GJR-GARCH(1,1) parameters.
GjrParams theta_to_gjr_params(const std::vector<double>& th);

// doc: map unconstrained optimizer vector to valid Shift-Twist GARCH(1,1) parameters.
StParams theta_to_st_params(const std::vector<double>& th);

// doc: map unconstrained dof parameter to degrees of freedom with constraint dof > 2.
double theta_to_dof(double th_dof);

}  // namespace vol

#endif
