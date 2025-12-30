#include "date_utils.h"

#include <cctype>
#include <cstdio>
#include <stdexcept>

namespace df {
namespace io {

int parse_iso_date_to_int(const std::string& iso_date) {
  if (iso_date.size() != 10 || iso_date[4] != '-' || iso_date[7] != '-') {
    throw std::runtime_error("invalid date format: " + iso_date);
  }
  auto digits_only = [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)); };
  for (char ch : iso_date) {
    if (ch == '-') continue;
    if (!digits_only(ch)) throw std::runtime_error("invalid date characters: " + iso_date);
  }
  int year = std::stoi(iso_date.substr(0, 4));
  int month = std::stoi(iso_date.substr(5, 2));
  int day = std::stoi(iso_date.substr(8, 2));
  return year * 10000 + month * 100 + day;
}

std::string format_int_date(int yyyymmdd) {
  if (yyyymmdd <= 0) return std::to_string(yyyymmdd);
  int year = yyyymmdd / 10000;
  int month_day = yyyymmdd % 10000;
  int month = month_day / 100;
  int day = month_day % 100;
  if (month < 1 || month > 12 || day < 1 || day > 31) {
    return std::to_string(yyyymmdd);
  }
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", year, month, day);
  return std::string(buf);
}

}  // namespace io
}  // namespace df
