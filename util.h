// util.h
#ifndef UTIL_H
#define UTIL_H

#include <cstdint>
#include <string>

namespace vol {

// doc: throw a runtime_error with the given message.
[[noreturn]] void die(const std::string& msg);

// doc: return true if s begins with prefix p.
bool starts_with(const std::string& s, const std::string& p);

// doc: return a time-based seed suitable for RNG initialization.
std::uint64_t default_seed_from_time();

}  // namespace vol

#endif
