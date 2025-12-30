#include "report.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct TableRow {
    vol::ReportModel model;
    double aicc;
    double bic;
    int loglik_rank;
    int aicc_rank;
    int bic_rank;
};

double compute_aicc(double loglik, int k, long long n) {
    if (n <= 0) return std::numeric_limits<double>::quiet_NaN();
    const long long denom = n - k - 1;
    if (denom <= 0) return std::numeric_limits<double>::quiet_NaN();
    const double aic = 2.0 * k - 2.0 * loglik;
    return aic + (2.0 * k * (k + 1)) / static_cast<double>(denom);
}

double compute_bic(double loglik, int k, long long n) {
    if (n <= 0) return std::numeric_limits<double>::quiet_NaN();
    return std::log(static_cast<double>(n)) * k - 2.0 * loglik;
}

std::vector<int> sorted_indices(const std::vector<double>& values, bool ascending = true) {
    std::vector<int> idx(values.size());
    std::iota(idx.begin(), idx.end(), 0);
    auto is_finite = [&](int i) { return std::isfinite(values[i]); };
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        const bool fa = is_finite(a);
        const bool fb = is_finite(b);
        if (fa != fb) return fa && !fb;
        if (!fa && !fb) return a < b;
        if (ascending) {
            return values[a] < values[b];
        }
        return values[a] > values[b];
    });
    return idx;
}

std::vector<int> compute_ranks(const std::vector<double>& values, bool ascending = true) {
    std::vector<int> ranks(values.size(), static_cast<int>(values.size()));
    auto order = sorted_indices(values, ascending);
    for (size_t pos = 0; pos < order.size(); ++pos) {
        ranks[order[pos]] = static_cast<int>(pos + 1);
    }
    return ranks;
}

void print_metric(double value) {
    if (std::isfinite(value)) {
        std::cout << std::setw(18) << value;
    } else {
        std::cout << std::setw(18) << "NA";
    }
}

void print_selections(const std::string& label,
                      const std::vector<std::string>& names,
                      const std::vector<double>& values) {
    auto order = sorted_indices(values);
    std::cout << "\n" << label << " selects\n";
    if (order.empty()) {
        std::cout << "  (no models)\n";
        return;
    }
    const double base = std::isfinite(values[order[0]]) ? values[order[0]]
                                                        : std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < order.size(); ++i) {
        const int idx = order[i];
        std::cout << std::left << std::setw(18) << names[idx] << " " << std::right;
        if (std::isfinite(values[idx])) {
            std::cout << std::setw(18) << values[idx];
        } else {
            std::cout << std::setw(18) << "NA";
        }
        std::cout << "  ";
        if (std::isfinite(values[idx]) && std::isfinite(base)) {
            const double diff = values[idx] - base;
            std::cout << std::setw(18) << (std::abs(diff) < 1e-12 ? 0.0 : diff);
        } else {
            std::cout << std::setw(18) << "NA";
        }
        if (i == 0) std::cout << "  best";
        std::cout << '\n';
    }
    std::cout << std::right;
}

}  // namespace

namespace vol {

void print_all_models(const std::vector<ReportModel>& models,
                      long long n_obs,
                      bool model_rows,
                      int precision) {
    std::cout.setf(std::ios::fixed);
    if (precision < 0) precision = 0;
    std::cout << std::setprecision(precision);
    if (models.empty()) {
        std::cout << "no models to report\n";
        return;
    }

    std::vector<double> aicc_vals;
    std::vector<double> bic_vals;
    std::vector<double> loglik_vals;
    aicc_vals.reserve(models.size());
    bic_vals.reserve(models.size());
    loglik_vals.reserve(models.size());
    for (const auto& m : models) {
        aicc_vals.push_back(compute_aicc(m.loglik, m.n_params, n_obs));
        bic_vals.push_back(compute_bic(m.loglik, m.n_params, n_obs));
        loglik_vals.push_back(m.loglik);
    }

    const auto aicc_ranks = compute_ranks(aicc_vals);
    const auto bic_ranks = compute_ranks(bic_vals);
    const auto loglik_ranks = compute_ranks(loglik_vals, /*ascending=*/false);

    std::vector<TableRow> rows;
    rows.reserve(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
        rows.push_back(TableRow{models[i], aicc_vals[i], bic_vals[i], loglik_ranks[i], aicc_ranks[i], bic_ranks[i]});
    }

    auto print_parameter_major = [&]() {
        std::cout << std::left << std::setw(18) << "parameter";
        for (const auto& row : rows) {
            std::cout << "  " << std::left << std::setw(18) << row.model.label;
        }
        std::cout << std::right << "\n";

        auto print_double_row = [&](const std::string& label, double ReportModel::*field) {
            std::cout << std::left << std::setw(18) << label << std::right;
            for (const auto& row : rows) {
                std::cout << "  ";
                print_metric(row.model.*field);
            }
            std::cout << "\n";
        };

        auto print_int_row = [&](const std::string& label, int ReportModel::*field) {
            std::cout << std::left << std::setw(18) << label << std::right;
            for (const auto& row : rows) {
                std::cout << "  " << std::setw(18) << row.model.*field;
            }
            std::cout << "\n";
        };

        print_double_row("uncond_sd", &ReportModel::uncond_sd);
        print_double_row("mu", &ReportModel::mu);
        print_double_row("omega", &ReportModel::omega);
        print_double_row("alpha", &ReportModel::alpha);
        print_double_row("beta", &ReportModel::beta);
        print_double_row("gamma", &ReportModel::gamma);
        print_double_row("shift", &ReportModel::shift);
        print_double_row("dof", &ReportModel::dof);
        print_double_row("loglik", &ReportModel::loglik);
        print_int_row("n_params", &ReportModel::n_params);
        auto print_extra_double = [&](const std::string& label, double TableRow::*field) {
            std::cout << std::left << std::setw(18) << label << std::right;
            for (const auto& row : rows) {
                std::cout << "  ";
                print_metric(row.*field);
            }
            std::cout << "\n";
        };
        auto print_extra_int = [&](const std::string& label, int TableRow::*field) {
            std::cout << std::left << std::setw(18) << label << std::right;
            for (const auto& row : rows) {
                std::cout << "  " << std::setw(18) << row.*field;
            }
            std::cout << "\n";
        };
        print_extra_double("AICC", &TableRow::aicc);
        print_extra_double("BIC", &TableRow::bic);
        print_extra_int("loglik_rank", &TableRow::loglik_rank);
        print_extra_int("AICC_rank", &TableRow::aicc_rank);
        print_extra_int("BIC_rank", &TableRow::bic_rank);
    };

    auto print_model_major = [&]() {
        const std::vector<std::string> headers = {
            "uncond_sd", "mu", "omega", "alpha", "beta", "gamma", "shift", "dof",
            "loglik", "n_params", "AICC", "BIC", "loglik_rank", "AICC_rank", "BIC_rank"};

        std::cout << std::left << std::setw(18) << "model";
        for (const auto& h : headers) {
            std::cout << "  " << std::right << std::setw(18) << h;
        }
        std::cout << std::right << "\n";

        for (const auto& row : rows) {
            std::cout << std::left << std::setw(18) << row.model.label << std::right;
            auto print_double = [&](double value) {
                std::cout << "  ";
                print_metric(value);
            };
            print_double(row.model.uncond_sd);
            print_double(row.model.mu);
            print_double(row.model.omega);
            print_double(row.model.alpha);
            print_double(row.model.beta);
            print_double(row.model.gamma);
            print_double(row.model.shift);
            print_double(row.model.dof);
            print_double(row.model.loglik);
            std::cout << "  " << std::setw(18) << row.model.n_params;
            print_double(row.aicc);
            print_double(row.bic);
            std::cout << "  " << std::setw(18) << row.loglik_rank;
            std::cout << "  " << std::setw(18) << row.aicc_rank;
            std::cout << "  " << std::setw(18) << row.bic_rank;
            std::cout << "\n";
        }
    };

    if (model_rows) {
        print_model_major();
    } else {
        print_parameter_major();
    }

    std::vector<std::string> labels;
    labels.reserve(rows.size());
    for (const auto& row : rows) labels.push_back(row.model.label);
    auto neg_loglik = loglik_vals;
    for (auto& v : neg_loglik) v = std::isfinite(v) ? -v : v;
    print_selections("loglik", labels, neg_loglik);
    print_selections("AICC", labels, aicc_vals);
    print_selections("BIC", labels, bic_vals);
}

CriteriaSummary summarize_criteria(const std::vector<ReportModel>& models,
                                   long long n_obs) {
    CriteriaSummary summary;
    if (models.empty()) {
        return summary;
    }

    std::vector<double> aicc_vals;
    std::vector<double> bic_vals;
    aicc_vals.reserve(models.size());
    bic_vals.reserve(models.size());
    for (const auto& model : models) {
        aicc_vals.push_back(compute_aicc(model.loglik, model.n_params, n_obs));
        bic_vals.push_back(compute_bic(model.loglik, model.n_params, n_obs));
    }

    auto names_for_metric = [&](const std::vector<double>& metric_values) {
        std::vector<std::string> ordered;
        auto idx = sorted_indices(metric_values);
        ordered.reserve(idx.size());
        for (int i : idx) {
            ordered.push_back(models[i].label);
        }
        return ordered;
    };

    summary.aicc_order = names_for_metric(aicc_vals);
    summary.bic_order = names_for_metric(bic_vals);
    return summary;
}

}  // namespace vol
