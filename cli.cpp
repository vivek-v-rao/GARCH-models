// cli.cpp
#include "cli.h"

#include "util.h"

#include <string>

namespace vol {

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
	std::unordered_map<std::string, std::string> m;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (!starts_with(a, "--")) die("bad argument (expected --key value): " + a);
		std::string key = a.substr(2);

		if (i + 1 >= argc) die("missing value for --" + key);
		std::string val = argv[++i];

		m[key] = val;
	}
	return m;
}

bool has(const std::unordered_map<std::string, std::string>& m, const std::string& k) {
	return m.find(k) != m.end();
}

long long get_ll(const std::unordered_map<std::string, std::string>& m,
		 const std::string& k,
		 long long default_val,
		 bool required) {
	if (!has(m, k)) {
		if (required) die("missing required argument --" + k);
		return default_val;
	}
	try {
		return std::stoll(m.at(k));
	} catch (...) {
		die("invalid integer for --" + k + ": " + m.at(k));
	}
	return default_val;
}

double get_d(const std::unordered_map<std::string, std::string>& m,
	     const std::string& k,
	     double default_val,
	     bool required) {
	if (!has(m, k)) {
		if (required) die("missing required argument --" + k);
		return default_val;
	}
	try {
		return std::stod(m.at(k));
	} catch (...) {
		die("invalid number for --" + k + ": " + m.at(k));
	}
	return default_val;
}

std::string get_s(const std::unordered_map<std::string, std::string>& m,
		  const std::string& k,
		  const std::string& default_val,
		  bool required) {
	if (!has(m, k)) {
		if (required) die("missing required argument --" + k);
		return default_val;
	}
	return m.at(k);
}

}  // namespace vol
