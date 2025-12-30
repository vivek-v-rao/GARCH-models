// fit_nagarch.h
#ifndef FIT_NAGARCH_H
#define FIT_NAGARCH_H

#include "param_maps.h"

#include <string>
#include <vector>

namespace vol {

struct NagarchFit {
	NagarchParams p;
	double dof;
};

// doc: fit NAGARCH(1,1) by MLE under normal or Student-t innovations (optionally fitting dof).
NagarchFit fit_nagarch(const std::vector<double>& r,
	       const std::string& dist,
	       double dof0,
	       const NagarchFit* warm_start = nullptr);

}  // namespace vol

#endif
