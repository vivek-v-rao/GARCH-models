#ifndef DATAFRAME_DATE_UTILS_H
#define DATAFRAME_DATE_UTILS_H

#include <string>

namespace df {
namespace io {

int parse_iso_date_to_int(const std::string& iso_date);
std::string format_int_date(int yyyymmdd);

}  // namespace io
}  // namespace df

#endif
