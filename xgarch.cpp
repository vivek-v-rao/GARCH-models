// nagarch_sim_main.cpp
//
// doc: simulate an NAGARCH path, write CSV output, fit models, and print comparison table.

#include "dist.h"
#include "cli.h"
#include "fit_garch.h"
#include "fit_nagarch.h"
#include "report.h"
#include "simulate.h"
#include "stats.h"
#include "util.h"
#include "vol_models.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

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

double compute_nagarch_uncond(const vol::NagarchFit& fit) {
	const double gamma = fit.p.gamma;
	const double u = fit.p.alpha * (1.0 + gamma * gamma) + fit.p.beta;
	const double denom = 1.0 - u;
	if (fit.p.omega > 0.0 && denom > 0.0) return std::sqrt(fit.p.omega / denom);
	return std::numeric_limits<double>::quiet_NaN();
}

double compute_nagarch_uncond(const vol::NagarchParams& p) {
	const double gamma = p.gamma;
	const double u = p.alpha * (1.0 + gamma * gamma) + p.beta;
	const double denom = 1.0 - u;
	if (p.omega > 0.0 && denom > 0.0) return std::sqrt(p.omega / denom);
	return std::numeric_limits<double>::quiet_NaN();
}

double compute_garch_uncond(const vol::GarchFit& fit) {
	const double u = fit.p.alpha + fit.p.beta;
	const double denom = 1.0 - u;
	if (fit.p.omega > 0.0 && denom > 0.0) return std::sqrt(fit.p.omega / denom);
	return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

int main(int argc, char** argv) {
	try {
		auto args = vol::parse_args(argc, argv);

    // plausible defaults for daily returns
		const long long n      = vol::get_ll(args, "n", 50000, false);
		const long long burnin = vol::get_ll(args, "burnin", 2000, false);
		const double mu        = vol::get_d(args, "mu", 0.0, false);

		const double omega     = vol::get_d(args, "omega", 1.0e-6, false);
		const double alpha     = vol::get_d(args, "alpha", 0.05, false);
		const double beta      = vol::get_d(args, "beta", 0.8, false);
		const double gamma     = vol::get_d(args, "gamma", 1.0, false);

    // default student-t with 6 dof
		const std::string dist = vol::get_s(args, "dist", "student-t", false);
		const double dof       = vol::get_d(args, "dof", 6.0, false);

    // default seed: time-based; allow override with --seed
		std::uint64_t seed = 0;
		if (vol::has(args, "seed")) {
			long long s = vol::get_ll(args, "seed", 0, true);
			seed = (std::uint64_t)s;
		} else {
			seed = vol::default_seed_from_time();
		}

    // default output file (simulation path)
		const std::string out  = vol::get_s(args, "out", "garch_sim.csv", false);

		auto trim = [](const std::string& s) {
			size_t start = 0;
			size_t end = s.size();
			while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
			while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
			return s.substr(start, end - start);
		};

		auto to_lower = [](std::string s) {
			for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
			return s;
		};

		const std::string table_rows_opt = vol::get_s(args, "table-rows", "models", false);
		const std::string table_rows = to_lower(trim(table_rows_opt));
		bool model_row_layout = true;
		if (table_rows == "models" || table_rows == "model" || table_rows == "rotated") {
			model_row_layout = true;
		} else if (table_rows == "parameters" || table_rows == "parameter" || table_rows == "legacy") {
			model_row_layout = false;
		} else {
			vol::die("invalid --table-rows (use models or parameters): " + table_rows_opt);
		}

		auto split_csv = [&](const std::string& csv) {
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
		};

//		const std::string default_models = "nagarch_normal,nagarch_student_t,garch_normal,garch_student_t";
	const std::string default_models = "garch_student_t,nagarch_student_t,st_student_t,st_normal";
	const std::string models_opt = vol::get_s(args, "models", default_models, false);
	const int param_precision = 6;
	const int summary_precision = 4;
	const int summary_width = 15;
		auto requested_models_raw = split_csv(models_opt);
		const std::unordered_set<std::string> allowed_models = {
			"nagarch_normal", "nagarch_student_t", "garch_normal", "garch_student_t",
			"st_normal", "st_student_t"};
		std::vector<std::string> requested_models;
		requested_models.reserve(requested_models_raw.size());
		std::unordered_set<std::string> seen_models;
		for (const auto& token : requested_models_raw) {
			const std::string key = to_lower(token);
			if (allowed_models.find(key) == allowed_models.end()) {
				vol::die("invalid model in --models: " + token);
			}
			if (!seen_models.insert(key).second) {
				vol::die("duplicate model in --models: " + token);
			}
			requested_models.push_back(key);
		}

		if (n <= 0) vol::die("--n must be positive");
		if (burnin < 0) vol::die("--burnin must be >= 0");
		if (!(omega > 0.0)) vol::die("--omega must be > 0");
		if (alpha < 0.0) vol::die("--alpha must be >= 0");
		if (beta < 0.0) vol::die("--beta must be >= 0");

		if (dist == "student-t") {
			if (!(dof > 2.0001)) vol::die("--dof must be > 2 for unit-variance Student-t innovations");
		} else if (dist == "normal") {
      // ok
		} else {
			vol::die("invalid --dist (use normal or student-t): " + dist);
		}

    // true NAGARCH parameters
		vol::NagarchParams true_p;
		true_p.mu = mu;
		true_p.omega = omega;
		true_p.alpha = alpha;
		true_p.beta = beta;
		true_p.gamma = gamma;

    // initial variance h0 (optional)
		double h0 = 0.0;
		bool use_h0 = false;
		if (vol::has(args, "h0")) {
			h0 = vol::get_d(args, "h0", 0.0, true);
			if (!(h0 > 0.0)) vol::die("--h0 must be > 0");
			use_h0 = true;
		}

    // simulate
		vol::SimPath path = vol::simulate_nagarch(n, burnin, true_p, dist, dof, seed, h0, use_h0);
		vol::write_path_csv(out, path);

		const std::vector<double>& returns = path.r;

		std::vector<double> cond_sd;
		cond_sd.reserve(path.h.size());
		for (size_t i = 0; i < path.h.size(); ++i) {
			double h = path.h[i];
			if (h < 0.0) h = 0.0;
			cond_sd.push_back(std::sqrt(h));
		}

		std::cout << "\nsummary: returns\n";
		stats::print_summary(returns, std::cout, summary_width, summary_precision, true, true);
		std::cout << "\nsummary: conditional_sd\n";
		stats::print_summary(cond_sd, std::cout, summary_width, summary_precision, true, true);
		std::cout << "\nsummary: std_returns\n";
		std::vector<double> std_returns = stats::standardize_returns(returns, cond_sd);
		stats::print_summary(std_returns, std::cout, summary_width, summary_precision, true, true);
		std::cout << "\ntheoretical kurtosis of std_returns: " << vol::innovations_excess_kurtosis(dist, dof);
    // fit NAGARCH (normal and student-t) to the same returns
		const auto nagarch_param_count = [](bool student_t) { return student_t ? 6 : 5; };
		const auto garch_param_count = [](bool student_t) { return student_t ? 5 : 4; };
		const auto st_param_count = [](bool student_t) { return student_t ? 7 : 6; };
		const double nan = std::numeric_limits<double>::quiet_NaN();

		std::vector<vol::ReportModel> report_models;
		report_models.reserve(1 + requested_models.size());

		vol::ReportModel true_row;
		true_row.label = "true";
		true_row.mu = true_p.mu;
		true_row.omega = true_p.omega;
		true_row.alpha = true_p.alpha;
		true_row.beta = true_p.beta;
		true_row.gamma = nan;
		true_row.shift = true_p.gamma;
		true_row.uncond_sd = compute_nagarch_uncond(true_p);
		true_row.dof = (dist == "student-t") ? dof : nan;

    // fit symmetric GARCH(1,1) (normal and student-t) to the same returns
		vol::NagarchFit est_nag_n;
		vol::NagarchFit est_nag_t;
		vol::GarchFit est_g_n;
		vol::GarchFit est_g_t;
		vol::StFit est_st_n;
		vol::StFit est_st_t;
		bool have_nag_n = false;
		bool have_nag_t = false;
		bool have_g_n = false;
		bool have_g_t = false;
		bool have_st_n = false;
		bool have_st_t = false;
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

		const bool wants_st_normal = (seen_models.find("st_normal") != seen_models.end());
		const bool wants_st_student = (seen_models.find("st_student_t") != seen_models.end());
		if (wants_st_normal) {
			est_st_n = vol::fit_st(returns, "normal", 0.0);
			have_st_n = true;
			warm_from_st_n = st_to_nagarch_warm(est_st_n);
			have_warm_st_n = true;
		}
		if (wants_st_student) {
			est_st_t = vol::fit_st(returns, "student-t", dof);
			have_st_t = true;
			warm_from_st_t = st_to_nagarch_warm(est_st_t);
			have_warm_st_t = true;
		}

    // logliks
		const double loglik_true =
					  -vol::neg_loglik_nagarch(returns, true_p, dist, (dist == "student-t") ? dof : 0.0);

		true_row.loglik = loglik_true;
		true_row.n_params = nagarch_param_count(dist == "student-t");
		report_models.push_back(true_row);

		for (const auto& model_name : requested_models) {
			if (model_name == "nagarch_normal") {
				if (!have_nag_n) {
					const vol::NagarchFit* warm_ptr = have_warm_st_n ? &warm_from_st_n : nullptr;
					est_nag_n = vol::fit_nagarch(returns, "normal", 0.0, warm_ptr);
					have_nag_n = true;
				}
				vol::ReportModel row;
				row.shift = nan;
				row.label = model_name;
					row.mu = est_nag_n.p.mu;
					row.omega = est_nag_n.p.omega;
					row.alpha = est_nag_n.p.alpha;
					row.beta = est_nag_n.p.beta;
				row.gamma = nan;
				row.shift = est_nag_n.p.gamma;
				row.uncond_sd = compute_nagarch_uncond(est_nag_n);
				row.dof = nan;
				row.loglik = -vol::neg_loglik_nagarch(returns, est_nag_n.p, "normal", 0.0);
				row.n_params = nagarch_param_count(false);
				report_models.push_back(row);
				warm_st_from_nag_n = nagarch_to_st_warm(est_nag_n);
				have_warm_st_from_nag_n = true;
			} else if (model_name == "nagarch_student_t") {
				if (!have_nag_t) {
					const vol::NagarchFit* warm_ptr = have_warm_st_t ? &warm_from_st_t : nullptr;
					est_nag_t = vol::fit_nagarch(returns, "student-t", dof, warm_ptr);
					have_nag_t = true;
				}
				vol::ReportModel row;
				row.shift = nan;
				row.label = model_name;
					row.mu = est_nag_t.p.mu;
					row.omega = est_nag_t.p.omega;
					row.alpha = est_nag_t.p.alpha;
					row.beta = est_nag_t.p.beta;
				row.gamma = nan;
				row.shift = est_nag_t.p.gamma;
				row.uncond_sd = compute_nagarch_uncond(est_nag_t);
				row.dof = est_nag_t.dof;
				row.loglik = -vol::neg_loglik_nagarch(returns, est_nag_t.p, "student-t", est_nag_t.dof);
				row.n_params = nagarch_param_count(true);
				report_models.push_back(row);
				warm_st_from_nag_t = nagarch_to_st_warm(est_nag_t);
				have_warm_st_from_nag_t = true;
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
				row.uncond_sd = compute_garch_uncond(est_g_n);
				row.dof = nan;
				row.loglik = -vol::neg_loglik_garch(returns, est_g_n.p, "normal", 0.0);
				row.n_params = garch_param_count(false);
				report_models.push_back(row);
			} else if (model_name == "garch_student_t") {
				if (!have_g_t) {
					est_g_t = vol::fit_garch(returns, "student-t", dof);
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
				row.uncond_sd = compute_garch_uncond(est_g_t);
				row.dof = est_g_t.dof;
				row.loglik = -vol::neg_loglik_garch(returns, est_g_t.p, "student-t", est_g_t.dof);
				row.n_params = garch_param_count(true);
				report_models.push_back(row);
			} else if (model_name == "st_normal") {
				if (!have_st_n) {
					est_st_n = vol::fit_st(returns, "normal", 0.0);
					have_st_n = true;
					warm_from_st_n = st_to_nagarch_warm(est_st_n);
					have_warm_st_n = true;
				}
				if (have_warm_st_from_nag_n) {
					est_st_n = vol::fit_st(returns, "normal", 0.0, &warm_st_from_nag_n);
					warm_from_st_n = st_to_nagarch_warm(est_st_n);
					have_warm_st_n = true;
					have_warm_st_from_nag_n = false;
				}
				vol::ReportModel row;
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
			} else if (model_name == "st_student_t") {
				if (!have_st_t) {
					est_st_t = vol::fit_st(returns, "student-t", dof);
					have_st_t = true;
					warm_from_st_t = st_to_nagarch_warm(est_st_t);
					have_warm_st_t = true;
				}
				if (have_warm_st_from_nag_t) {
					est_st_t = vol::fit_st(returns, "student-t", dof, &warm_st_from_nag_t);
					warm_from_st_t = st_to_nagarch_warm(est_st_t);
					have_warm_st_t = true;
					have_warm_st_from_nag_t = false;
				}
				vol::ReportModel row;
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
			double nag_ll = -vol::neg_loglik_nagarch(returns, est_nag_t.p, dist, est_nag_t.dof);
			if (nag_ll - st_ll > loglik_tol) {
				vol::StFit refined = vol::fit_st(returns, "student-t",
							      (est_nag_t.dof > 2.0001) ? est_nag_t.dof : dof,
							      &warm_st_from_nag_t);
				double refined_ll = -vol::neg_loglik_st(returns, refined.p, "student-t", refined.dof);
				if (refined_ll > st_ll) {
					est_st_t = refined;
					have_st_t = true;
					warm_from_st_t = st_to_nagarch_warm(est_st_t);
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
					warm_from_st_n = st_to_nagarch_warm(est_st_n);
					have_warm_st_n = true;
					refresh_st_row(st_normal_index, est_st_n, "normal");
				}
			}
		}

		std::cout << "\n#obs: " << n << "\n";
		std::cout << "seed: " << (unsigned long long)seed << "\n\n";

		vol::print_all_models(report_models,
			      n,
		      model_row_layout,
		      param_precision);

		return 0;
	} catch (const std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
		std::cerr << "usage:\n"
				<< "  nagarch_sim [--n N] [--burnin B] [--mu MU]\n"
				<< "             [--omega OMEGA] [--alpha ALPHA] [--beta BETA] [--gamma GAMMA]\n"
				<< "             [--dist normal|student-t] [--dof DOF] [--seed SEED] [--h0 H0] [--out FILE]\n"
				<< "             [--table-rows models|parameters]\n"
				<< "             [--models model1,model2,...]\n";
		return 1;
	}
}
