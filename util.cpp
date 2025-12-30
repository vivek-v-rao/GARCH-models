// util.cpp
#include "util.h"

#include <chrono>
#include <stdexcept>

namespace vol {

[[noreturn]] void die(const std::string& msg) {
	throw std::runtime_error(msg);
}

bool starts_with(const std::string& s, const std::string& p) {
	return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

std::uint64_t default_seed_from_time() {
	return (std::uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

}  // namespace vol
