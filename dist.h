// dist.h
#include <string>
#ifndef DIST_H
#define DIST_H

namespace vol {

// doc: log pdf of standard Student-t(df=dof) at x (location 0, scale 1).
double log_t_pdf(double x, double dof);

// doc: scale so that if x ~ t_nu then z = x/scale has Var(z)=1 (requires nu > 2).
double standardized_t_scale(double dof);
double student_t_excess_kurtosis(double dof);
// doc: theoretical excess kurtosis of innovations for dist in {"normal","student-t"}.
//      for student-t, returns +inf if dof <= 4.
double innovations_excess_kurtosis(const std::string& dist, double dof);
}  // namespace vol

#endif
