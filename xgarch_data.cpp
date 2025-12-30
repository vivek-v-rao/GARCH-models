// xgarch_data.cpp
//
// Read price data via dataframe utilities, compute log returns by column, and fit GARCH-family models.

#include "dataframe.h"
#include "date_utils.h"

#include "cli.h"
#include "fit_garch.h"
#include "fit_nagarch.h"
#include "report.h"
#include "stats.h"
#include "util.h"
#include "vol_models.h"
#include "xgarch_utils.h"

#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace xu = xgarch;
using xu::ColumnSummary;
using xu::ResidualSummary;
using xu::SeriesSummary;
using DataFrame = xu::DataFrame;

int main(int argc, char** argv) {
    try {
        auto args = vol::parse_args(argc, argv);
        const std::string file = vol::get_s(args, "file", "prices.csv", false);
        const std::string columns_opt = vol::get_s(args, "columns", "", false);
        const std::string table_rows_opt = vol::get_s(args, "table-rows", "models", false);
        const std::string models_opt = vol::get_s(
            args,
            "models",
            "nagarch_normal,nagarch_student_t,garch_student_t,garch_normal,gjr_student_t,gjr_normal,st_student_t,st_normal,igarch_student_t,igarch_normal,constant_vol",
            false);
        const long long min_rows = vol::get_ll(args, "min-rows", 250, false);
        const double scale_returns = vol::get_d(args, "scale", 100.0, false);
        if (!(scale_returns > 0.0)) {
            vol::die("--scale must be > 0");
        }
        const int param_precision = 6;
        const int summary_precision = 4;
        const int summary_width = 15;
        const int autocorr_lags = 5;
        const int autocorr_precision = 3;
        const int autocorr_width = 12;
        const int resid_table_width = 15;
        bool print_resid_stats = true;
        if (vol::has(args, "no-resid-stats")) print_resid_stats = false;
        bool print_cond_sd_stats = true;
        if (vol::has(args, "no-cond-sd-stats")) print_cond_sd_stats = false;
        bool demean_returns = true;
        if (vol::has(args, "no-demean")) demean_returns = false;
        const long long max_columns = vol::get_ll(args, "max-columns", -1, false);
        if (max_columns == 0) {
            vol::die("--max-columns must be positive when provided");
        }

        std::string table_rows_key = xu::to_lower(xu::trim(table_rows_opt));
        bool model_row_layout = true;
        if (table_rows_key == "models" || table_rows_key == "model" || table_rows_key == "rotated") {
            model_row_layout = true;
        } else if (table_rows_key == "parameters" || table_rows_key == "parameter" || table_rows_key == "legacy") {
            model_row_layout = false;
        } else {
            vol::die("invalid --table-rows (use models or parameters): " + table_rows_opt);
        }

        DataFrame prices = xu::load_price_dataframe(file);
        std::cout << "loaded price data from " << file
                  << " with " << prices.rows() << " rows and " << prices.cols() << " columns\n";

        if (prices.rows() > 0) {
            int first_idx = prices.index().front();
            int last_idx = prices.index().back();
            auto format_idx = [](int idx) {
                try {
                    return df::io::format_int_date(idx);
                } catch (...) {
                    return std::to_string(idx);
                }
            };
            std::cout << "date range: " << format_idx(first_idx)
                      << " to " << format_idx(last_idx) << "\n";
        }

        std::cout << "return scaling factor: " << scale_returns << "\n";
        std::cout << "demean returns: " << (demean_returns ? "yes" : "no") << "\n";

        auto column_names = xu::resolve_columns(prices, columns_opt);
        if (column_names.empty()) {
            vol::die("no columns selected");
        }
        if (max_columns > 0 && static_cast<long long>(column_names.size()) > max_columns) {
            column_names.resize(static_cast<size_t>(max_columns));
            std::cout << "limiting to first " << max_columns << " column(s)\n";
        }

        std::vector<ColumnSummary> column_summaries;
        column_summaries.reserve(column_names.size());

        const std::unordered_set<std::string> allowed_models = {
            "nagarch_normal",  "nagarch_student_t", "garch_normal",
            "garch_student_t", "gjr_normal",        "gjr_student_t",
            "st_normal",       "st_student_t",     "igarch_normal",
            "igarch_student_t", "constant_vol"};
        auto requested_models_raw = xu::split_csv_list(models_opt);
        if (requested_models_raw.empty()) {
            vol::die("--models must list at least one model");
        }
        std::vector<std::string> requested_models;
        std::unordered_set<std::string> seen_models;
        requested_models.reserve(requested_models_raw.size());
        for (const auto& token : requested_models_raw) {
            const std::string key = xu::to_lower(token);
            if (allowed_models.find(key) == allowed_models.end()) {
                vol::die("invalid model in --models: " + token);
            }
            if (!seen_models.insert(key).second) {
                vol::die("duplicate model in --models: " + token);
            }
            requested_models.push_back(key);
        }

        const bool wants_st_normal = seen_models.find("st_normal") != seen_models.end();
        const bool wants_st_student = seen_models.find("st_student_t") != seen_models.end();

        const auto nagarch_param_count = [](bool student_t) { return student_t ? 6 : 5; };
        const auto garch_param_count = [](bool student_t) { return student_t ? 5 : 4; };
        const auto gjr_param_count = [](bool student_t) { return student_t ? 6 : 5; };
        const auto st_param_count = [](bool student_t) { return student_t ? 7 : 6; };
        const double nan = std::numeric_limits<double>::quiet_NaN();

        for (const auto& column : column_names) {
            auto prices_vec = prices.column_data(column);
            auto returns = xu::compute_log_returns(prices_vec);
            for (auto& r : returns) r *= scale_returns;
            if (demean_returns) {
                double m = stats::mean(returns);
                if (std::isfinite(m)) {
                    for (auto& r : returns) r -= m;
                }
            }
            const long long n_obs = static_cast<long long>(returns.size());
            std::cout << "\n==== column: " << column << " ====\n";
            std::cout << "price observations: " << prices_vec.size()
                      << ", log returns used: " << n_obs << "\n";
            if (n_obs < min_rows) {
                std::cout << "insufficient data (need at least " << min_rows
                          << ") -- skipping model fits\n";
                continue;
            }

            stats::print_summary(returns, std::cout, summary_width, summary_precision, true, true);
            std::cout << "\nautocorrelations (lag 1-" << autocorr_lags << ")\n";
            stats::print_autocorr_table(returns,
                                        autocorr_lags,
                                        std::cout,
                                        autocorr_width,
                                        autocorr_precision,
                                        true);
	    std::cout << "\n";
            vol::NagarchFit est_nag_n;
            vol::NagarchFit est_nag_t;
            vol::GarchFit est_g_n;
            vol::GarchFit est_g_t;
            vol::GjrFit est_gjr_n;
            vol::GjrFit est_gjr_t;
            vol::StFit est_st_n;
            vol::StFit est_st_t;
            vol::ConstantVolFit est_const;
            bool have_nag_n = false;
            bool have_nag_t = false;
            bool have_g_n = false;
            bool have_g_t = false;
            bool have_gjr_n = false;
            bool have_gjr_t = false;
            bool have_st_n = false;
            bool have_st_t = false;
            bool have_const = false;
            vol::NagarchFit warm_from_st_n;
            vol::NagarchFit warm_from_st_t;
            bool have_warm_st_n = false;
            bool have_warm_st_t = false;
            vol::StFit warm_st_from_nag_n;
            vol::StFit warm_st_from_nag_t;
            bool have_warm_st_from_nag_n = false;
            bool have_warm_st_from_nag_t = false;
            int st_normal_index = -1;
            int st_student_index = -1;

            if (wants_st_normal) {
                est_st_n = vol::fit_st(returns, "normal", 0.0);
                have_st_n = true;
                warm_from_st_n = xu::st_to_nagarch_warm(est_st_n);
                have_warm_st_n = true;
            }
            if (wants_st_student) {
                est_st_t = vol::fit_st(returns, "student-t", 6.0);
                have_st_t = true;
                warm_from_st_t = xu::st_to_nagarch_warm(est_st_t);
                have_warm_st_t = true;
            }

            std::vector<vol::ReportModel> report_models;
            report_models.reserve(requested_models.size());
            std::vector<ResidualSummary> residual_summaries;
            if (print_resid_stats) residual_summaries.reserve(requested_models.size());
            std::vector<SeriesSummary> cond_sd_summaries;
            if (print_cond_sd_stats) cond_sd_summaries.reserve(requested_models.size());

            auto add_cond_sd_stats = [&](const std::string& label,
                                        const std::vector<double>& cond_sd) {
                if (!print_cond_sd_stats) return;
                if (cond_sd.size() != returns.size() || cond_sd.empty()) return;
                cond_sd_summaries.push_back(SeriesSummary{label, stats::summary_stats(cond_sd)});
            };

            auto add_resid_stats = [&](const std::string& label,
                                       const std::vector<double>& cond_sd) {
                if (!print_resid_stats) return;
                if (cond_sd.size() != returns.size() || cond_sd.empty()) return;
                auto std_resid = stats::standardize_returns(returns, cond_sd);
                residual_summaries.push_back(ResidualSummary{label, stats::summary_stats(std_resid)});
            };

            for (const auto& model_name : requested_models) {
                if (model_name == "nagarch_normal") {
                    if (!have_nag_n) {
                        const vol::NagarchFit* warm_ptr = have_warm_st_n ? &warm_from_st_n : nullptr;
                        est_nag_n = vol::fit_nagarch(returns, "normal", 0.0, warm_ptr);
                        have_nag_n = true;
                    }
                    vol::ReportModel row;
                    row.shift = est_nag_n.p.gamma;
                    row.label = model_name;
                    row.mu = est_nag_n.p.mu;
                    row.omega = est_nag_n.p.omega;
                    row.alpha = est_nag_n.p.alpha;
                    row.beta = est_nag_n.p.beta;
                    row.gamma = nan;
                    row.uncond_sd = xu::compute_nagarch_uncond(est_nag_n);
                    row.dof = nan;
                    row.loglik = -vol::neg_loglik_nagarch(returns, est_nag_n.p, "normal", 0.0);
                    row.n_params = nagarch_param_count(false);
                    report_models.push_back(row);
                    warm_st_from_nag_n = xu::nagarch_to_st_warm(est_nag_n);
                    have_warm_st_from_nag_n = true;
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_nagarch(returns, est_nag_n.p);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                } else if (model_name == "nagarch_student_t") {
                    if (!have_nag_t) {
                        const vol::NagarchFit* warm_ptr = have_warm_st_t ? &warm_from_st_t : nullptr;
                        est_nag_t = vol::fit_nagarch(returns, "student-t", 6.0, warm_ptr);
                        have_nag_t = true;
                    }
                    vol::ReportModel row;
                    row.shift = est_nag_t.p.gamma;
                    row.label = model_name;
                    row.mu = est_nag_t.p.mu;
                    row.omega = est_nag_t.p.omega;
                    row.alpha = est_nag_t.p.alpha;
                    row.beta = est_nag_t.p.beta;
                    row.gamma = nan;
                    row.uncond_sd = xu::compute_nagarch_uncond(est_nag_t);
                    row.dof = est_nag_t.dof;
                    row.loglik = -vol::neg_loglik_nagarch(returns, est_nag_t.p, "student-t", est_nag_t.dof);
                    row.n_params = nagarch_param_count(true);
                    report_models.push_back(row);
                    warm_st_from_nag_t = xu::nagarch_to_st_warm(est_nag_t);
                    have_warm_st_from_nag_t = true;
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_nagarch(returns, est_nag_t.p);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                } else if (model_name == "garch_normal") {
                    if (!have_g_n) {
                        est_g_n = vol::fit_garch(returns, "normal", 0.0);
                        have_g_n = true;
                    }
                    vol::ReportModel row;
                    row.shift = nan;
                    row.label = model_name;
                    row.mu = est_g_n.p.mu;
                    row.omega = est_g_n.p.omega;
                    row.alpha = est_g_n.p.alpha;
                    row.beta = est_g_n.p.beta;
                    row.gamma = nan;
                    row.uncond_sd = xu::compute_garch_uncond(est_g_n);
                    row.dof = nan;
                    row.loglik = -vol::neg_loglik_garch(returns, est_g_n.p, "normal", 0.0);
                    row.n_params = garch_param_count(false);
                    report_models.push_back(row);
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_garch(returns, est_g_n.p);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                } else if (model_name == "garch_student_t") {
                    if (!have_g_t) {
                        est_g_t = vol::fit_garch(returns, "student-t", 6.0);
                        have_g_t = true;
                    }
                    vol::ReportModel row;
                    row.shift = nan;
                    row.label = model_name;
                    row.mu = est_g_t.p.mu;
                    row.omega = est_g_t.p.omega;
                    row.alpha = est_g_t.p.alpha;
                    row.beta = est_g_t.p.beta;
                    row.gamma = nan;
                    row.uncond_sd = xu::compute_garch_uncond(est_g_t);
                    row.dof = est_g_t.dof;
                    row.loglik = -vol::neg_loglik_garch(returns, est_g_t.p, "student-t", est_g_t.dof);
                    row.n_params = garch_param_count(true);
                    report_models.push_back(row);
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_garch(returns, est_g_t.p);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                } else if (model_name == "gjr_normal") {
                    if (!have_gjr_n) {
                        est_gjr_n = vol::fit_gjr(returns, "normal", 0.0);
                        have_gjr_n = true;
                    }
                    vol::ReportModel row;
                    row.shift = nan;
                    row.label = model_name;
                    row.mu = est_gjr_n.p.mu;
                    row.omega = est_gjr_n.p.omega;
                    row.alpha = est_gjr_n.p.alpha;
                    row.beta = est_gjr_n.p.beta;
                    row.gamma = est_gjr_n.p.gamma;
                    row.uncond_sd = xu::compute_gjr_uncond(est_gjr_n);
                    row.dof = nan;
                    row.loglik = -vol::neg_loglik_gjr(returns, est_gjr_n.p, "normal", 0.0);
                    row.n_params = gjr_param_count(false);
                    report_models.push_back(row);
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_gjr(returns, est_gjr_n.p);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                } else if (model_name == "igarch_normal" || model_name == "igarch_student_t") {
                    const bool student = (model_name == "igarch_student_t");
                    vol::IgarchFit est = vol::fit_igarch(returns, student ? "student-t" : "normal", 6.0);
                    vol::ReportModel row;
                    row.label = model_name;
                    row.mu = est.p.mu;
                    row.omega = 0.0;
                    row.alpha = est.p.alpha;
                    row.beta = est.p.beta;
                    row.gamma = nan;
                    row.shift = nan;
                    row.uncond_sd = nan;
                    row.dof = student ? est.dof : nan;
                    row.loglik = -vol::neg_loglik_igarch(returns, est.p, student ? "student-t" : "normal", student ? est.dof : 0.0);
                    row.n_params = student ? 3 : 2;
                    report_models.push_back(row);
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_igarch(returns, est.p);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                } else if (model_name == "gjr_student_t") {
                    if (!have_gjr_t) {
                        est_gjr_t = vol::fit_gjr(returns, "student-t", 6.0);
                        have_gjr_t = true;
                    }
                    vol::ReportModel row;
                    row.shift = nan;
                    row.label = model_name;
                    row.mu = est_gjr_t.p.mu;
                    row.omega = est_gjr_t.p.omega;
                    row.alpha = est_gjr_t.p.alpha;
                    row.beta = est_gjr_t.p.beta;
                    row.gamma = est_gjr_t.p.gamma;
                    row.uncond_sd = xu::compute_gjr_uncond(est_gjr_t);
                    row.dof = est_gjr_t.dof;
                    row.loglik = -vol::neg_loglik_gjr(returns, est_gjr_t.p, "student-t", est_gjr_t.dof);
                    row.n_params = gjr_param_count(true);
                    report_models.push_back(row);
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_gjr(returns, est_gjr_t.p);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                } else if (model_name == "st_normal") {
                    if (!have_st_n) {
                        est_st_n = vol::fit_st(returns, "normal", 0.0);
                        have_st_n = true;
                        warm_from_st_n = xu::st_to_nagarch_warm(est_st_n);
                        have_warm_st_n = true;
                    }
                    vol::ReportModel row;
                    row.shift = nan;
                    row.label = model_name;
                    row.mu = est_st_n.p.mu;
                    row.omega = est_st_n.p.omega;
                    row.alpha = est_st_n.p.alpha;
                    row.beta = est_st_n.p.beta;
                    row.gamma = est_st_n.p.gamma;
                    row.shift = est_st_n.p.shift;
                    row.dof = nan;
                    row.loglik = -vol::neg_loglik_st(returns, est_st_n.p, "normal", 0.0);
                    row.n_params = st_param_count(false);
                    report_models.push_back(row);
                    st_normal_index = static_cast<int>(report_models.size() - 1);
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_st(returns, est_st_n.p);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                } else if (model_name == "st_student_t") {
                    if (!have_st_t) {
                        est_st_t = vol::fit_st(returns, "student-t", 6.0);
                        have_st_t = true;
                        if (!have_warm_st_t) {
                            warm_from_st_t = xu::st_to_nagarch_warm(est_st_t);
                            have_warm_st_t = true;
                        }
                    }
                    vol::ReportModel row;
                    row.shift = nan;
                    row.label = model_name;
                    row.mu = est_st_t.p.mu;
                    row.omega = est_st_t.p.omega;
                    row.alpha = est_st_t.p.alpha;
                    row.beta = est_st_t.p.beta;
                    row.gamma = est_st_t.p.gamma;
                    row.shift = est_st_t.p.shift;
                    row.dof = est_st_t.dof;
                    row.loglik = -vol::neg_loglik_st(returns, est_st_t.p, "student-t", est_st_t.dof);
                    row.n_params = st_param_count(true);
                    report_models.push_back(row);
                    st_student_index = static_cast<int>(report_models.size() - 1);
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_st(returns, est_st_t.p);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                } else if (model_name == "constant_vol") {
                    if (!have_const) {
                        est_const = vol::fit_constant_vol(returns);
                        have_const = true;
                    }
                    vol::ReportModel row;
                    row.shift = nan;
                    row.label = model_name;
                    row.mu = 0.0;
                    row.omega = est_const.omega;
                    row.alpha = nan;
                    row.beta = nan;
                    row.gamma = nan;
                    row.uncond_sd = std::sqrt(est_const.omega);
                    row.dof = nan;
                    row.loglik = -vol::neg_loglik_constant(returns, 0.0, est_const.omega);
                    row.n_params = 1;
                    report_models.push_back(row);
                    if (print_resid_stats || print_cond_sd_stats) {
                        auto cond_sd = xu::compute_cond_sd_constant(returns, est_const.omega);
                        add_resid_stats(row.label, cond_sd);
                        add_cond_sd_stats(row.label, cond_sd);
                    }
                }
            }

            auto refresh_st_row = [&](int idx, const vol::StFit& fit, const std::string& dist) {
                if (idx < 0) return;
                auto& row = report_models[idx];
                row.mu = fit.p.mu;
                row.omega = fit.p.omega;
                row.alpha = fit.p.alpha;
                row.beta = fit.p.beta;
                row.gamma = fit.p.gamma;
                row.shift = fit.p.shift;
                if (dist == "student-t") {
                    row.dof = fit.dof;
                } else {
                    row.dof = nan;
                }
                row.loglik = -vol::neg_loglik_st(returns, fit.p, dist, (dist == "student-t") ? fit.dof : 0.0);
                row.n_params = (dist == "student-t") ? st_param_count(true) : st_param_count(false);
            };

            const double loglik_tol = 1e-6;
            if (st_student_index >= 0 && have_nag_t && have_warm_st_from_nag_t) {
                double st_ll = report_models[st_student_index].loglik;
                double nag_ll = -vol::neg_loglik_nagarch(returns, est_nag_t.p, "student-t", est_nag_t.dof);
                if (nag_ll - st_ll > loglik_tol) {
                    vol::StFit refined = vol::fit_st(returns, "student-t",
                                                      (est_nag_t.dof > 2.0001) ? est_nag_t.dof : 6.0,
                                                      &warm_st_from_nag_t);
                    double refined_ll = -vol::neg_loglik_st(returns, refined.p, "student-t", refined.dof);
                    if (refined_ll > st_ll) {
                        est_st_t = refined;
                        have_st_t = true;
                        warm_from_st_t = xu::st_to_nagarch_warm(est_st_t);
                        have_warm_st_t = true;
                        refresh_st_row(st_student_index, est_st_t, "student-t");
                    }
                }
            }
            if (st_normal_index >= 0 && have_nag_n && have_warm_st_from_nag_n) {
                double st_ll = report_models[st_normal_index].loglik;
                double nag_ll = -vol::neg_loglik_nagarch(returns, est_nag_n.p, "normal", 0.0);
                if (nag_ll - st_ll > loglik_tol) {
                    vol::StFit refined = vol::fit_st(returns, "normal", 0.0, &warm_st_from_nag_n);
                    double refined_ll = -vol::neg_loglik_st(returns, refined.p, "normal", 0.0);
                    if (refined_ll > st_ll) {
                        est_st_n = refined;
                        have_st_n = true;
                        warm_from_st_n = xu::st_to_nagarch_warm(est_st_n);
                        have_warm_st_n = true;
                        refresh_st_row(st_normal_index, est_st_n, "normal");
                    }
                }
            }

            vol::print_all_models(report_models, n_obs, model_row_layout, param_precision);
            column_summaries.push_back(ColumnSummary{column,
                                                     vol::summarize_criteria(report_models, n_obs)});
            if (print_resid_stats) {
                std::cout << "\nstandardized residual stats\n";
                xu::print_residual_summary_table(residual_summaries,
                                             resid_table_width,
                                             summary_precision,
                                             std::cout);
            }
            if (print_cond_sd_stats) {
                std::cout << "\nconditional sd stats\n";
                xu::print_series_summary_table(cond_sd_summaries,
                                                resid_table_width,
                                                summary_precision,
                                                std::cout);
            }
        }

        std::cout << "\n==== best models by asset and information criterion ====\n";
        if (column_summaries.empty()) {
            std::cout << "no fitted models to summarize\n";
        } else {
            for (const auto& entry : column_summaries) {
                auto print_line = [&](const std::string& ic,
                                      const std::vector<std::string>& names) {
                    std::cout << entry.name << ' ' << ic << ": ";
                    if (names.empty()) {
                        std::cout << "(no models)";
                    } else {
                        for (size_t i = 0; i < names.size(); ++i) {
                            if (i > 0) std::cout << ", ";
                            std::cout << names[i];
                        }
                    }
                    std::cout << '\n';
                };
                print_line("AICC", entry.criteria.aicc_order);
                print_line("BIC", entry.criteria.bic_order);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        std::cerr << "usage:\n"
                  << "  xgarch_data [--file FILE] [--columns COL1,COL2,...]\n"
                  << "             [--models MODEL1,MODEL2,...] [--table-rows models|parameters]\n"
                  << "             [--min-rows N] [--max-columns K] [--scale FACTOR] [--no-resid-stats] [--no-cond-sd-stats]\n"
                  << "             [--no-demean]\n";
        return 1;
    }
}
