// nelder_mead.h
#ifndef NELDER_MEAD_H
#define NELDER_MEAD_H

#include <functional>
#include <vector>

namespace vol {

// doc: Nelder-Mead simplex minimizer for a user-supplied objective f(x).
std::vector<double> nelder_mead_minimize(const std::vector<double>& x0,
					 const std::vector<double>& step,
					 const std::function<double(const std::vector<double>&)>& f,
					 int max_iter,
					 double tol_f);

}  // namespace vol

#endif
