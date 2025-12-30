// fit_garch.h
#ifndef FIT_GARCH_H
#define FIT_GARCH_H

#include "param_maps.h"

#include <string>
#include <vector>

namespace vol {

struct GarchFit {
	GarchParams p;
	double dof;
};

struct GjrFit {
	GjrParams p;
	double dof;
};

struct StFit {
	StParams p;
	double dof;
};

struct IgarchFit {
	IgarchParams p;
	double dof;
};

struct ConstantVolFit {
	double omega;
};

// doc: fit symmetric GARCH(1,1) by MLE under normal or Student-t innovations (optionally fitting dof).
GarchFit fit_garch(const std::vector<double>& r,
		   const std::string& dist,
		   double dof0);

// doc: fit GJR-GARCH(1,1) by MLE under normal or Student-t innovations (optionally fitting dof).
GjrFit fit_gjr(const std::vector<double>& r,
	     const std::string& dist,
	     double dof0);

StFit fit_st(const std::vector<double>& r,
   const std::string& dist,
   double dof0,
   const StFit* warm_start = nullptr);

IgarchFit fit_igarch(const std::vector<double>& r,
         const std::string& dist,
         double dof0);

ConstantVolFit fit_constant_vol(const std::vector<double>& r);

}  // namespace vol

#endif
