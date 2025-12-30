// vol_models.h
#ifndef VOL_MODELS_H
#define VOL_MODELS_H

#include "param_maps.h"

#include <string>
#include <vector>

namespace vol {

// doc: negative log-likelihood for NAGARCH(1,1) under normal or standardized Student-t errors.
double neg_loglik_nagarch(const std::vector<double>& r,
			  const NagarchParams& p,
			  const std::string& dist,
			  double dof);
double neg_loglik_nagarch_grad(const std::vector<double>& r,
                               const NagarchParams& p,
                               const std::string& dist,
                               double dof,
                               std::vector<double>* grad);

// doc: negative log-likelihood for symmetric GARCH(1,1) under normal or standardized Student-t errors.
double neg_loglik_garch(const std::vector<double>& r,
			const GarchParams& p,
			const std::string& dist,
			double dof);
double neg_loglik_garch_grad(const std::vector<double>& r,
                              const GarchParams& p,
                              const std::string& dist,
                              double dof,
                              std::vector<double>* grad);

double neg_loglik_igarch(const std::vector<double>& r,
                           const IgarchParams& p,
                           const std::string& dist,
                           double dof);

double neg_loglik_igarch_grad(const std::vector<double>& r,
                                 const IgarchParams& p,
                                 const std::string& dist,
                                 double dof,
                                 std::vector<double>* grad);

// doc: negative log-likelihood for GJR-GARCH(1,1) under normal or standardized Student-t errors.
double neg_loglik_gjr(const std::vector<double>& r,
		       const GjrParams& p,
		       const std::string& dist,
		       double dof);
double neg_loglik_gjr_grad(const std::vector<double>& r,
                           const GjrParams& p,
                           const std::string& dist,
                           double dof,
                           std::vector<double>* grad);

double neg_loglik_st(const std::vector<double>& r,
            const StParams& p,
            const std::string& dist,
            double dof);
double neg_loglik_st_grad(const std::vector<double>& r,
                          const StParams& p,
                          const std::string& dist,
                          double dof,
                          std::vector<double>* grad);

double neg_loglik_constant(const std::vector<double>& r,
                           double mu,
                           double omega);
double neg_loglik_constant_grad(const std::vector<double>& r,
                                double mu,
                                double omega,
                                std::vector<double>* grad);

}  // namespace vol

#endif
