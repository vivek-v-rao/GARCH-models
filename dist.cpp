// dist.cpp
#include "dist.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace vol {

static const double kPi = 3.14159265358979323846;

double log_t_pdf(double x, double dof) {
  // log pdf of standard Student-t with df=dof (location 0, scale 1)
  // f(x) = Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2)) * (1 + x^2/nu)^(-(nu+1)/2)
	const double nu = dof;
	const double a = std::lgamma(0.5 * (nu + 1.0)) - std::lgamma(0.5 * nu);
	const double b = -0.5 * (std::log(nu) + std::log(kPi));
	const double c = -0.5 * (nu + 1.0) * std::log(1.0 + (x * x) / nu);
	return a + b + c;
}

double standardized_t_scale(double dof) {
  // Var(t_nu) = nu/(nu-2), so scale = sqrt(nu/(nu-2))
	return std::sqrt(dof / (dof - 2.0));
}

double student_t_excess_kurtosis(double dof) {
	if (!(dof > 0.0)) {
		throw std::runtime_error("student_t_excess_kurtosis: dof must be positive");
	}
	if (dof <= 4.0) {
		return std::numeric_limits<double>::infinity();
	}
	return 6.0 / (dof - 4.0);
}

double innovations_excess_kurtosis(const std::string& dist, double dof) {
	if (dist == "normal") return 0.0;

	if (dist == "student-t") {
		// excess kurtosis for Student-t with dof nu is 6/(nu-4) for nu>4; infinite otherwise.
		if (!(dof > 4.0001)) return std::numeric_limits<double>::infinity();
		return 6.0 / (dof - 4.0);
	}

	return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace vol
