// simulate.cpp
#include "simulate.h"

#include "dist.h"
#include "util.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>

namespace vol {

SimPath simulate_nagarch(long long n,
			 long long burnin,
			 const NagarchParams& p,
			 const std::string& dist,
			 double dof,
			 std::uint64_t seed,
			 double h0,
			 bool use_h0) {
	if (n <= 0) die("simulate_nagarch: n must be positive");
	if (burnin < 0) die("simulate_nagarch: burnin must be >= 0");

	if (!(p.omega > 0.0)) die("simulate_nagarch: omega must be > 0");
	if (p.alpha < 0.0) die("simulate_nagarch: alpha must be >= 0");
	if (p.beta < 0.0) die("simulate_nagarch: beta must be >= 0");

	if (dist == "student-t") {
		if (!(dof > 2.0001)) die("simulate_nagarch: dof must be > 2 for unit-variance Student-t");
	} else if (dist == "normal") {
    // ok
	} else {
		die("simulate_nagarch: invalid dist (use normal or student-t): " + dist);
	}

	double h = 0.0;
	if (use_h0) {
		h = h0;
		if (!(h > 0.0)) die("simulate_nagarch: h0 must be > 0");
	} else {
		const double denom = 1.0 - (p.alpha * (1.0 + p.gamma * p.gamma) + p.beta);
		if (denom > 0.0) {
			h = p.omega / denom;
		} else {
			h = p.omega;
		}
	}

	std::mt19937_64 rng(seed);
	std::normal_distribution<double> ndist(0.0, 1.0);
	std::student_t_distribution<double> tdist(dof);

	auto draw_z = [&]() -> double {
		if (dist == "normal") return ndist(rng);
		const double x = tdist(rng);
		const double scale = standardized_t_scale(dof);
		return x / scale;
	};

	SimPath path;
	path.r.reserve((size_t)n);
	path.eps.reserve((size_t)n);
	path.h.reserve((size_t)n);
	path.z.reserve((size_t)n);

	const long long total = burnin + n;

	for (long long show_t = 0; show_t < total; ++show_t) {
		if (!(h > 0.0) || !std::isfinite(h)) die("simulate_nagarch: variance became non-positive or non-finite");

		const double z = draw_z();
		const double sqrt_h = std::sqrt(h);
		const double eps = sqrt_h * z;
		const double r = p.mu + eps;

		if (show_t >= burnin) {
			path.r.push_back(r);
			path.eps.push_back(eps);
			path.h.push_back(h);
			path.z.push_back(z);
		}

		const double shock = eps - p.gamma * sqrt_h;
		const double h_next = p.omega + p.alpha * (shock * shock) + p.beta * h;
		h = h_next;
	}

	return path;
}

void write_path_csv(const std::string& out, const SimPath& path) {
	std::ofstream fout(out.c_str(), std::ios::out | std::ios::trunc);
	if (!fout) die("write_path_csv: failed to open file: " + out);

	fout.setf(std::ios::fixed);
	fout << std::setprecision(10);
	fout << "t,r,eps,h,z\n";

	const long long n = (long long)path.r.size();
	for (long long t = 0; t < n; ++t) {
		fout << t << "," << path.r[(size_t)t]
				<< "," << path.eps[(size_t)t]
				<< "," << path.h[(size_t)t]
				<< "," << path.z[(size_t)t]
				<< "\n";
	}
	fout.close();
}

}  // namespace vol
