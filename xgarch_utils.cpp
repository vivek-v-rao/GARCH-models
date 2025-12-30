#include "xgarch_utils.h"

#include "date_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace xgarch {

vol::NagarchFit st_to_nagarch_warm(const vol::StFit& st) {
    vol::NagarchFit warm;
    warm.p.mu = st.p.mu;
    warm.p.omega = st.p.omega;
    warm.p.alpha = st.p.alpha;
    warm.p.beta = st.p.beta;
    warm.p.gamma = st.p.shift;
    warm.dof = st.dof;
    return warm;
}

vol::StFit nagarch_to_st_warm(const vol::NagarchFit& nag) {
    vol::StFit warm;
    warm.p.mu = nag.p.mu;
    warm.p.omega = nag.p.omega;
    warm.p.alpha = nag.p.alpha;
    warm.p.beta = nag.p.beta;
    warm.p.gamma = 0.0;
    warm.p.shift = nag.p.gamma;
    warm.dof = nag.dof;
    return warm;
}

double compute_nagarch_uncond(const vol::NagarchParams& p) {
    const double gamma = p.gamma;
    const double u = p.alpha * (1.0 + gamma * gamma) + p.beta;
    const double denom = 1.0 - u;
    if (p.omega > 0.0 && denom > 0.0) return std::sqrt(p.omega / denom);
    return std::numeric_limits<double>::quiet_NaN();
}

double compute_nagarch_uncond(const vol::NagarchFit& fit) {
    return compute_nagarch_uncond(fit.p);
}

double compute_garch_uncond(const vol::GarchFit& fit) {
    const double u = fit.p.alpha + fit.p.beta;
    const double denom = 1.0 - u;
    if (fit.p.omega > 0.0 && denom > 0.0) return std::sqrt(fit.p.omega / denom);
    return std::numeric_limits<double>::quiet_NaN();
}

double compute_gjr_uncond(const vol::GjrFit& fit) {
    const double u = fit.p.alpha + 0.5 * fit.p.gamma + fit.p.beta;
    const double denom = 1.0 - u;
    if (fit.p.omega > 0.0 && denom > 0.0) return std::sqrt(fit.p.omega / denom);
    return std::numeric_limits<double>::quiet_NaN();
}

std::string trim(const std::string& s) {
    size_t start = 0;
    size_t end = s.size();
    while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::string to_lower(std::string s) {
    for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

std::vector<std::string> split_csv_list(const std::string& csv) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= csv.size()) {
        size_t end = csv.find(',', start);
        const size_t len = (end == std::string::npos) ? std::string::npos : (end - start);
        std::string token = (end == std::string::npos) ? csv.substr(start) : csv.substr(start, len);
        token = trim(token);
        if (!token.empty()) parts.push_back(token);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return parts;
}

DataFrame load_price_dataframe(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path);
    }

    std::string header;
    if (!std::getline(input, header)) {
        throw std::runtime_error("empty file: " + path);
    }
    auto comma_pos = header.find(',');
    if (comma_pos == std::string::npos) {
        throw std::runtime_error("header missing data columns");
    }

    std::stringstream data_stream;
    data_stream << "index," << header.substr(comma_pos + 1) << '\n';

    std::string line;
    while (std::getline(input, line)) {
        if (trim(line).empty()) continue;
        std::size_t first_comma = line.find(',');
        if (first_comma == std::string::npos) {
            throw std::runtime_error("invalid row: " + line);
        }
        std::string date_str = line.substr(0, first_comma);
        std::string rest = line.substr(first_comma + 1);
        int idx = df::io::parse_iso_date_to_int(date_str);
        data_stream << idx << ',' << rest << '\n';
    }

    DataFrame df = DataFrame::from_csv(data_stream, true);
    df.set_index_name("Date");
    return df;
}

std::vector<std::string> resolve_columns(const DataFrame& df, const std::string& csv_columns) {
    if (trim(csv_columns).empty()) {
        return df.columns();
    }
    auto requested = split_csv_list(csv_columns);
    if (requested.empty()) {
        return df.columns();
    }
    std::vector<std::string> found;
    const auto& existing = df.columns();
    for (const auto& name : requested) {
        auto it = std::find(existing.begin(), existing.end(), name);
        if (it == existing.end()) {
            throw std::runtime_error("unknown column: " + name);
        }
        if (std::find(found.begin(), found.end(), name) == found.end()) {
            found.push_back(name);
        }
    }
    return found;
}

std::vector<double> compute_log_returns(const std::vector<double>& prices) {
    std::vector<double> returns;
    if (prices.empty()) return returns;
    returns.reserve(prices.size());
    double prev = std::numeric_limits<double>::quiet_NaN();
    bool have_prev = false;
    for (double price : prices) {
        if (std::isfinite(price) && price > 0.0) {
            if (have_prev) {
                double r = std::log(price) - std::log(prev);
                if (std::isfinite(r)) returns.push_back(r);
            }
            prev = price;
            have_prev = true;
        } else {
            have_prev = false;
        }
    }
    return returns;
}

std::vector<double> compute_cond_sd_nagarch(const std::vector<double>& returns,
                                            const vol::NagarchParams& p) {
    std::vector<double> cond_sd;
    cond_sd.reserve(returns.size());
    const double gamma = p.gamma;
    const double u = p.alpha * (1.0 + gamma * gamma) + p.beta;
    const double denom = 1.0 - u;
    double h = (p.omega > 0.0 && denom > 1e-12) ? (p.omega / denom) : p.omega;
    if (!(h > 0.0)) h = 1e-6;
    for (double r : returns) {
        if (!(h > 0.0)) h = 1e-6;
        const double sd = std::sqrt(std::max(h, 1e-12));
        cond_sd.push_back(sd);
        const double eps = r - p.mu;
        const double shock = eps - gamma * sd;
        double h_next = p.omega + p.alpha * (shock * shock) + p.beta * h;
        if (!(h_next > 0.0) || !std::isfinite(h_next)) h_next = 1e-8;
        h = h_next;
    }
    return cond_sd;
}

std::vector<double> compute_cond_sd_garch(const std::vector<double>& returns,
                                          const vol::GarchParams& p) {
    std::vector<double> cond_sd;
    cond_sd.reserve(returns.size());
    const double u = p.alpha + p.beta;
    const double denom = 1.0 - u;
    double h = (p.omega > 0.0 && denom > 1e-12) ? (p.omega / denom) : p.omega;
    if (!(h > 0.0)) h = 1e-6;
    for (double r : returns) {
        if (!(h > 0.0)) h = 1e-6;
        const double sd = std::sqrt(std::max(h, 1e-12));
        cond_sd.push_back(sd);
        const double eps = r - p.mu;
        double h_next = p.omega + p.alpha * (eps * eps) + p.beta * h;
        if (!(h_next > 0.0) || !std::isfinite(h_next)) h_next = 1e-8;
        h = h_next;
    }
    return cond_sd;
}

std::vector<double> compute_cond_sd_igarch(const std::vector<double>& returns,
                                           const vol::IgarchParams& p) {
    std::vector<double> cond_sd;
    cond_sd.reserve(returns.size());
    double var = 0.0;
    if (!returns.empty()) {
        for (double r : returns) {
            const double eps = r - p.mu;
            var += eps * eps;
        }
        var /= static_cast<double>(returns.size());
    }
    double h = (var > 1e-12) ? var : 1e-6;
    for (double r : returns) {
        if (!(h > 0.0)) h = 1e-6;
        const double sd = std::sqrt(std::max(h, 1e-12));
        cond_sd.push_back(sd);
        const double eps = r - p.mu;
        double h_next = p.alpha * (eps * eps) + p.beta * h;
        if (!(h_next > 0.0) || !std::isfinite(h_next)) h_next = 1e-8;
        h = h_next;
    }
    return cond_sd;
}

std::vector<double> compute_cond_sd_gjr(const std::vector<double>& returns,
                                        const vol::GjrParams& p) {
    std::vector<double> cond_sd;
    cond_sd.reserve(returns.size());
    const double u = p.alpha + 0.5 * p.gamma + p.beta;
    const double denom = 1.0 - u;
    double h = (p.omega > 0.0 && denom > 1e-12) ? (p.omega / denom) : p.omega;
    if (!(h > 0.0)) h = 1e-6;
    for (double r : returns) {
        if (!(h > 0.0)) h = 1e-6;
        const double sd = std::sqrt(std::max(h, 1e-12));
        cond_sd.push_back(sd);
        const double eps = r - p.mu;
        const double indicator = (eps < 0.0) ? 1.0 : 0.0;
        double h_next = p.omega + (p.alpha + p.gamma * indicator) * (eps * eps) + p.beta * h;
        if (!(h_next > 0.0) || !std::isfinite(h_next)) h_next = 1e-8;
        h = h_next;
    }
    return cond_sd;
}

std::vector<double> compute_cond_sd_st(const std::vector<double>& returns,
                                       const vol::StParams& p) {
    std::vector<double> cond_sd;
    cond_sd.reserve(returns.size());
    const double u = p.alpha + 0.5 * p.gamma + p.beta;
    const double denom = 1.0 - u;
    double h = (p.omega > 0.0 && denom > 1e-12) ? (p.omega / denom) : p.omega;
    if (!(h > 0.0)) h = 1e-6;
    for (double r : returns) {
        if (!(h > 0.0)) h = 1e-6;
        const double sd = std::sqrt(std::max(h, 1e-12));
        cond_sd.push_back(sd);
        const double eps = r - p.mu;
        const double indicator = (eps < 0.0) ? 1.0 : 0.0;
        const double shock = eps - p.shift * sd;
        double h_next = p.omega + (p.alpha + p.gamma * indicator) * (shock * shock) + p.beta * h;
        if (!(h_next > 0.0) || !std::isfinite(h_next)) h_next = 1e-8;
        h = h_next;
    }
    return cond_sd;
}

std::vector<double> compute_cond_sd_constant(const std::vector<double>& returns,
                                             double omega) {
    std::vector<double> cond_sd;
    cond_sd.reserve(returns.size());
    const double sd = std::sqrt(std::max(omega, 1e-12));
    for (size_t i = 0; i < returns.size(); ++i) cond_sd.push_back(sd);
    return cond_sd;
}

void print_residual_summary_table(const std::vector<ResidualSummary>& rows,
                                  int width,
                                  int precision,
                                  std::ostream& os) {
    if (rows.empty()) {
        os << "no residual stats\n";
        return;
    }

    const int model_width = 18;
    const int n_width = 10;
    const int col_width = (width < 8) ? 8 : width;
    const int prec = (precision < 0) ? 0 : precision;

    std::ios::fmtflags old_flags = os.flags();
    std::streamsize old_prec = os.precision();
    os.setf(std::ios::fixed, std::ios::floatfield);
    os << std::setprecision(prec);

    auto print_value = [&](double value) {
        if (std::isfinite(value)) {
            os << std::setw(col_width) << value;
        } else {
            os << std::setw(col_width) << "NA";
        }
    };

    os << std::left << std::setw(model_width) << "model" << std::right << ' '
       << std::setw(n_width) << "n" << ' '
       << std::setw(col_width) << "mean" << ' '
       << std::setw(col_width) << "sd" << ' '
       << std::setw(col_width) << "skew" << ' '
       << std::setw(col_width) << "ex_kurtosis" << ' '
       << std::setw(col_width) << "min" << ' '
       << std::setw(col_width) << "max" << "\n";

    for (const auto& row : rows) {
        os << std::left << std::setw(model_width) << row.label << std::right << ' ';
        os << std::setw(n_width) << row.stats.n << ' ';
        print_value(row.stats.mean);
        os << ' ';
        print_value(row.stats.sd);
        os << ' ';
        print_value(row.stats.skew);
        os << ' ';
        print_value(row.stats.ex_kurtosis);
        os << ' ';
        print_value(row.stats.min);
        os << ' ';
        print_value(row.stats.max);
        os << "\n";
    }

    os.precision(old_prec);
    os.flags(old_flags);
}

void print_series_summary_table(const std::vector<SeriesSummary>& rows,
                                int width,
                                int precision,
                                std::ostream& os) {
    if (rows.empty()) {
        os << "no conditional sd stats\n";
        return;
    }

    const int model_width = 18;
    const int n_width = 10;
    const int col_width = (width < 8) ? 8 : width;
    const int prec = (precision < 0) ? 0 : precision;

    std::ios::fmtflags old_flags = os.flags();
    std::streamsize old_prec = os.precision();
    os.setf(std::ios::fixed, std::ios::floatfield);
    os << std::setprecision(prec);

    auto print_value = [&](double value) {
        if (std::isfinite(value)) {
            os << std::setw(col_width) << value;
        } else {
            os << std::setw(col_width) << "NA";
        }
    };

    os << std::left << std::setw(model_width) << "model" << std::right << ' '
       << std::setw(n_width) << "n" << ' '
       << std::setw(col_width) << "mean" << ' '
       << std::setw(col_width) << "sd" << ' '
       << std::setw(col_width) << "skew" << ' '
       << std::setw(col_width) << "ex_kurtosis" << ' '
       << std::setw(col_width) << "min" << ' '
       << std::setw(col_width) << "max" << "\n";

    for (const auto& row : rows) {
        os << std::left << std::setw(model_width) << row.label << std::right << ' ';
        os << std::setw(n_width) << row.stats.n << ' ';
        print_value(row.stats.mean);
        os << ' ';
        print_value(row.stats.sd);
        os << ' ';
        print_value(row.stats.skew);
        os << ' ';
        print_value(row.stats.ex_kurtosis);
        os << ' ';
        print_value(row.stats.min);
        os << ' ';
        print_value(row.stats.max);
        os << "\n";
    }

    os.precision(old_prec);
    os.flags(old_flags);
}

}  // namespace xgarch
