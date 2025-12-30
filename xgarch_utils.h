#ifndef XGARCH_UTILS_H
#define XGARCH_UTILS_H

#include "dataframe.h"
#include "fit_garch.h"
#include "fit_nagarch.h"
#include "report.h"
#include "stats.h"
#include "vol_models.h"

#include <ostream>
#include <string>
#include <vector>

namespace xgarch {

using DataFrame = df::DataFrame<int>;

struct ColumnSummary {
    std::string name;
    vol::CriteriaSummary criteria;
};

struct ResidualSummary {
    std::string label;
    stats::SummaryStats stats;
};


struct SeriesSummary {
    std::string label;
    stats::SummaryStats stats;
};
std::string trim(const std::string& s);
std::string to_lower(std::string s);
std::vector<std::string> split_csv_list(const std::string& csv);
DataFrame load_price_dataframe(const std::string& path);
std::vector<std::string> resolve_columns(const DataFrame& df, const std::string& csv_columns);
std::vector<double> compute_log_returns(const std::vector<double>& prices);

std::vector<double> compute_cond_sd_nagarch(const std::vector<double>& returns,
                                            const vol::NagarchParams& p);
std::vector<double> compute_cond_sd_garch(const std::vector<double>& returns,
                                          const vol::GarchParams& p);
std::vector<double> compute_cond_sd_igarch(const std::vector<double>& returns,
                                           const vol::IgarchParams& p);
std::vector<double> compute_cond_sd_gjr(const std::vector<double>& returns,
                                        const vol::GjrParams& p);
std::vector<double> compute_cond_sd_st(const std::vector<double>& returns,
                                       const vol::StParams& p);
std::vector<double> compute_cond_sd_constant(const std::vector<double>& returns,
                                             double omega);

void print_residual_summary_table(const std::vector<ResidualSummary>& rows,
                                  int width,
                                  int precision,
                                  std::ostream& os);
void print_series_summary_table(const std::vector<SeriesSummary>& rows,
                                int width,
                                int precision,
                                std::ostream& os);

vol::NagarchFit st_to_nagarch_warm(const vol::StFit& st);
vol::StFit nagarch_to_st_warm(const vol::NagarchFit& nag);
double compute_nagarch_uncond(const vol::NagarchParams& p);
double compute_nagarch_uncond(const vol::NagarchFit& fit);
double compute_garch_uncond(const vol::GarchFit& fit);
double compute_gjr_uncond(const vol::GjrFit& fit);

}  // namespace xgarch

#endif
