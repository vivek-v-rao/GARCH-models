// simulate.h
#ifndef SIMULATE_H
#define SIMULATE_H

#include "param_maps.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vol {

struct SimPath {
	std::vector<double> r;
	std::vector<double> eps;
	std::vector<double> h;
	std::vector<double> z;
};

// doc: simulate an NAGARCH(1,1) path and return series after burnin.
SimPath simulate_nagarch(long long n,
			 long long burnin,
			 const NagarchParams& p,
			 const std::string& dist,
			 double dof,
			 std::uint64_t seed,
			 double h0,
			 bool use_h0);

// doc: write a simulation path to CSV with columns t,r,eps,h,z.
void write_path_csv(const std::string& out, const SimPath& path);

}  // namespace vol

#endif
