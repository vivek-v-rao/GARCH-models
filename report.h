#ifndef REPORT_H
#define REPORT_H

#include <limits>
#include <string>
#include <vector>

namespace vol {

struct ReportModel {
    std::string label;
    double uncond_sd = std::numeric_limits<double>::quiet_NaN();
    double mu;
    double omega;
    double alpha;
    double beta;
    double gamma;
    double shift = std::numeric_limits<double>::quiet_NaN();
    double dof;
    double loglik;
    int n_params;
};

struct CriteriaSummary {
    std::vector<std::string> aicc_order;
    std::vector<std::string> bic_order;
};

// doc: print selected model summaries (parameter-major or model-major) plus IC rankings.
void print_all_models(const std::vector<ReportModel>& models,
                      long long n_obs,
                      bool model_rows,
                      int precision);

// doc: return model label orderings (best to worst) for AICC/BIC given fitted models.
CriteriaSummary summarize_criteria(const std::vector<ReportModel>& models,
                                   long long n_obs);

}  // namespace vol

#endif
