// cli.h
#ifndef CLI_H
#define CLI_H

#include <string>
#include <unordered_map>

namespace vol {

// doc: parse command line args of the form --key value into a string map.
std::unordered_map<std::string, std::string> parse_args(int argc, char** argv);

// doc: return true if key k exists in map m.
bool has(const std::unordered_map<std::string, std::string>& m, const std::string& k);

// doc: read a long long option from map, with default/required handling.
long long get_ll(const std::unordered_map<std::string, std::string>& m,
		 const std::string& k,
		 long long default_val,
		 bool required);

// doc: read a double option from map, with default/required handling.
double get_d(const std::unordered_map<std::string, std::string>& m,
	     const std::string& k,
	     double default_val,
	     bool required);

// doc: read a string option from map, with default/required handling.
std::string get_s(const std::unordered_map<std::string, std::string>& m,
		  const std::string& k,
		  const std::string& default_val,
		  bool required);

}  // namespace vol

#endif
