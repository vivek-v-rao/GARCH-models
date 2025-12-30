// nelder_mead.cpp
#include "nelder_mead.h"

#include "util.h"

#include <algorithm>
#include <cmath>

namespace vol {

std::vector<double> nelder_mead_minimize(const std::vector<double>& x0,
					 const std::vector<double>& step,
					 const std::function<double(const std::vector<double>&)>& f,
					 int max_iter,
					 double tol_f) {
	const int n = (int)x0.size();
	if ((int)step.size() != n) die("nelder_mead: step size mismatch");
	if (n <= 0) die("nelder_mead: empty parameter vector");

  // simplex: n+1 points
	std::vector<std::vector<double> > x(n + 1, x0);
	for (int i = 0; i < n; ++i) x[i + 1][i] += step[i];

	std::vector<double> fx(n + 1, 0.0);
	for (int i = 0; i < n + 1; ++i) fx[i] = f(x[i]);

	const double alpha = 1.0;
	const double gamma = 2.0;
	const double rho   = 0.5;
	const double sigma = 0.5;

	std::vector<int> idx(n + 1);
	for (int i = 0; i < n + 1; ++i) idx[i] = i;

	std::vector<double> xc(n, 0.0), xr(n, 0.0), xe(n, 0.0), xk(n, 0.0);

	for (int it = 0; it < max_iter; ++it) {
		std::sort(idx.begin(), idx.end(), [&](int a, int b) { return fx[a] < fx[b]; });

		const double fbest  = fx[idx[0]];
		const double fworst = fx[idx[n]];
		if (std::fabs(fworst - fbest) <= tol_f) break;

    // centroid excluding worst
		for (int j = 0; j < n; ++j) xc[j] = 0.0;
		for (int i = 0; i < n; ++i) {
			const std::vector<double>& xi = x[idx[i]];
			for (int j = 0; j < n; ++j) xc[j] += xi[j];
		}
		for (int j = 0; j < n; ++j) xc[j] /= (double)n;

    // reflection
		const std::vector<double>& xw = x[idx[n]];
		for (int j = 0; j < n; ++j) xr[j] = xc[j] + alpha * (xc[j] - xw[j]);
		double fr = f(xr);

		const double f2 = fx[idx[n - 1]];

		if (fr < fbest) {
      // expansion
			for (int j = 0; j < n; ++j) xe[j] = xc[j] + gamma * (xr[j] - xc[j]);
			double fe = f(xe);
			if (fe < fr) {
				x[idx[n]] = xe;
				fx[idx[n]] = fe;
			} else {
				x[idx[n]] = xr;
				fx[idx[n]] = fr;
			}
			continue;
		}

		if (fr < f2) {
      // accept reflection
			x[idx[n]] = xr;
			fx[idx[n]] = fr;
			continue;
		}

    // contraction
		if (fr < fworst) {
      // outside contraction
			for (int j = 0; j < n; ++j) xk[j] = xc[j] + rho * (xr[j] - xc[j]);
		} else {
      // inside contraction
			for (int j = 0; j < n; ++j) xk[j] = xc[j] + rho * (xw[j] - xc[j]);
		}
		double fk = f(xk);

		if (fk < fworst) {
			x[idx[n]] = xk;
			fx[idx[n]] = fk;
			continue;
		}

    // shrink toward best
		const std::vector<double>& xb = x[idx[0]];
		for (int i = 1; i < n + 1; ++i) {
			std::vector<double>& xi = x[idx[i]];
			for (int j = 0; j < n; ++j) xi[j] = xb[j] + sigma * (xi[j] - xb[j]);
			fx[idx[i]] = f(xi);
		}
	}

	std::sort(idx.begin(), idx.end(), [&](int a, int b) { return fx[a] < fx[b]; });
	return x[idx[0]];
}

}  // namespace vol
