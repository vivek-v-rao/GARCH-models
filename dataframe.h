#ifndef DATAFRAME_DATAFRAME_H
#define DATAFRAME_DATAFRAME_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "stats.h"

namespace df {

namespace detail {

inline std::string trim(const std::string& s) {
  std::size_t start = 0;
  std::size_t end = s.size();
  while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
  return s.substr(start, end - start);
}

inline std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream ss(line);
  while (std::getline(ss, field, ',')) {
    fields.push_back(trim(field));
  }
  if (!line.empty() && line.back() == ',') {
    fields.emplace_back();
  }
  return fields;
}

template <typename T>
T parse_token(const std::string& token) {
  if constexpr (std::is_same_v<T, std::string>) {
    return token;
  } else {
    std::istringstream ss(token);
    T value{};
    ss >> value;
    if (!ss || (ss >> std::ws && !ss.eof())) {
      throw std::runtime_error("failed to parse token");
    }
    return value;
  }
}

}  // namespace detail

template <typename IndexT>
class DataFrame {
 public:
  template <typename> friend class DataFrame;
  static DataFrame from_csv(std::istream& input, bool has_index);
  static DataFrame random_normal(std::size_t rows,
                                 const std::vector<std::string>& columns,
                                 double mean = 0.0,
                                 double stddev = 1.0,
                                 std::uint32_t seed = 0);

  DataFrame differences() const;
  DataFrame log_changes() const;
  DataFrame proportional_changes() const;
  DataFrame add(double value) const;
  DataFrame subtract(double value) const;
  DataFrame multiply(double value) const;
  DataFrame divide(double value) const;
  DataFrame add(const DataFrame& other) const;
  DataFrame subtract(const DataFrame& other) const;
  DataFrame multiply(const DataFrame& other) const;
  DataFrame divide(const DataFrame& other) const;
  DataFrame log_elements() const;
  DataFrame exp_elements() const;
  DataFrame power(double exponent) const;
  DataFrame power_int(int exponent) const;
  DataFrame standardize() const;
  DataFrame normalize() const;
  DataFrame select_rows(const std::vector<IndexT>& values) const;
  DataFrame select_columns(const std::vector<std::string>& names) const;
  template <typename T = IndexT,
            typename = std::enable_if_t<std::is_integral_v<T>>>
  DataFrame slice_rows_range(IndexT start,
                             IndexT end,
                             bool inclusive_end = true) const;
  DataFrame head_rows(std::size_t count) const;
  DataFrame tail_rows(std::size_t count) const;
  DataFrame head_columns(std::size_t count) const;
  DataFrame tail_columns(std::size_t count) const;
  std::vector<double> column_data(const std::string& name) const;
  std::vector<double> row_data(const IndexT& index_value) const;
  DataFrame rolling_mean(std::size_t window) const;
  DataFrame rolling_std(std::size_t window) const;
  DataFrame rolling_rms(std::size_t window) const;
  DataFrame exponential_moving_average(double alpha) const;
  DataFrame resample_rows(std::size_t sample_size = 0,
                          bool reset_index = true) const;
  DataFrame remove_rows_with_nan() const;
  DataFrame remove_columns_with_nan() const;
  DataFrame<std::string> column_stats_dataframe() const;
  DataFrame<std::string> correlation_matrix() const;

  std::size_t rows() const { return data_.size(); }
  std::size_t cols() const { return columns_.size(); }
  const std::vector<std::string>& columns() const { return columns_; }
  const std::vector<IndexT>& index() const { return index_; }
  const std::string& index_name() const { return index_name_; }
  void set_index_name(const std::string& name) { index_name_ = name; }

  double value(std::size_t row, std::size_t col) const;

 private:
  std::vector<std::string> columns_;
  std::vector<IndexT> index_;
  std::vector<std::vector<double>> data_;
  std::string index_name_ = "index";

  template <typename Func>
  DataFrame apply_scalar(Func func) const;

  template <typename Func>
  DataFrame apply_unary(Func func, const char* name) const;

  template <typename Func>
  DataFrame apply_binary(const DataFrame& other, Func func, const char* name) const;

  DataFrame select_rows_by_positions(const std::vector<std::size_t>& positions) const;

  DataFrame select_columns_by_positions(const std::vector<std::size_t>& positions) const;

  std::vector<std::size_t> find_row_positions_in_range(IndexT start,
                                                       IndexT end,
                                                       bool inclusive_end) const;

  std::size_t find_column_index(const std::string& name) const;

  std::size_t find_row_position(const IndexT& value) const;
};

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::from_csv(std::istream& input, bool has_index) {
  std::string header;
  if (!std::getline(input, header)) {
    throw std::runtime_error("dataframe::from_csv: missing header row");
  }
  auto header_fields = detail::split_csv(header);
  if (header_fields.empty()) {
    throw std::runtime_error("dataframe::from_csv: header has no columns");
  }

  std::size_t start_col = has_index ? 1 : 0;
  if (has_index && header_fields.size() < 2) {
    throw std::runtime_error("dataframe::from_csv: need at least one data column when reading indices");
  }

  DataFrame<IndexT> df;
  df.columns_.assign(header_fields.begin() + static_cast<std::ptrdiff_t>(start_col), header_fields.end());
  df.index_name_ = has_index ? header_fields[0] : "index";
  if (df.columns_.empty()) {
    throw std::runtime_error("dataframe::from_csv: no data columns found");
  }

  std::string line;
  while (std::getline(input, line)) {
    if (detail::trim(line).empty()) continue;
    auto fields = detail::split_csv(line);
    const std::size_t expected = df.columns_.size() + (has_index ? 1 : 0);
    if (fields.size() != expected) {
      throw std::runtime_error("dataframe::from_csv: row has unexpected number of columns");
    }

    IndexT idx{};
    std::size_t offset = 0;
    if (has_index) {
      try {
        idx = detail::parse_token<IndexT>(fields[0]);
      } catch (const std::exception&) {
        throw std::runtime_error("dataframe::from_csv: invalid index value");
      }
      offset = 1;
    } else {
      if constexpr (std::is_convertible_v<std::size_t, IndexT>) {
        idx = static_cast<IndexT>(df.index_.size());
      } else {
        throw std::runtime_error("dataframe::from_csv: index type cannot be auto-generated");
      }
    }

    std::vector<double> row;
    row.reserve(df.columns_.size());
    for (std::size_t c = 0; c < df.columns_.size(); ++c) {
      const std::string& token = fields[c + offset];
      if (token.empty()) {
        row.push_back(std::numeric_limits<double>::quiet_NaN());
        continue;
      }
      try {
        row.push_back(std::stod(token));
      } catch (const std::exception&) {
        throw std::runtime_error("dataframe::from_csv: invalid numeric value");
      }
    }

    df.index_.push_back(idx);
    df.data_.push_back(std::move(row));
  }

  return df;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::random_normal(std::size_t rows,
                                                   const std::vector<std::string>& columns,
                                                   double mean,
                                                   double stddev,
                                                   std::uint32_t seed) {
  static_assert(std::is_integral_v<IndexT>, "random_normal requires integral indices");
  if (columns.empty()) {
    throw std::runtime_error("random_normal: at least one column is required");
  }
  if (stddev <= 0.0) {
    throw std::runtime_error("random_normal: standard deviation must be positive");
  }

  if (rows > static_cast<std::size_t>(std::numeric_limits<IndexT>::max())) {
    throw std::runtime_error("random_normal: row count exceeds index capacity");
  }

  DataFrame<IndexT> df;
  df.columns_ = columns;
  df.index_name_ = "index";
  df.index_.reserve(rows);
  df.data_.reserve(rows);

  std::mt19937 rng(seed == 0 ? std::mt19937::result_type(std::random_device{}()) : seed);
  std::normal_distribution<double> dist(mean, stddev);

  for (std::size_t row = 0; row < rows; ++row) {
    df.index_.push_back(static_cast<IndexT>(row));
    std::vector<double> row_values;
    row_values.reserve(df.columns_.size());
    for (std::size_t col = 0; col < df.columns_.size(); ++col) {
      row_values.push_back(dist(rng));
    }
    df.data_.push_back(std::move(row_values));
  }

  return df;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::differences() const {
  if (data_.size() < 2) {
    throw std::runtime_error("dataframe::differences: need at least two rows");
  }
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_.assign(index_.begin() + 1, index_.end());
  out.index_name_ = index_name_;
  out.data_.resize(data_.size() - 1, std::vector<double>(columns_.size(), 0.0));
  for (std::size_t r = 1; r < data_.size(); ++r) {
    for (std::size_t c = 0; c < columns_.size(); ++c) {
      out.data_[r - 1][c] = data_[r][c] - data_[r - 1][c];
    }
  }
  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::log_changes() const {
  if (data_.size() < 2) {
    throw std::runtime_error("dataframe::log_changes: need at least two rows");
  }
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_.assign(index_.begin() + 1, index_.end());
  out.index_name_ = index_name_;
  out.data_.resize(data_.size() - 1, std::vector<double>(columns_.size(), 0.0));
  for (std::size_t r = 1; r < data_.size(); ++r) {
    for (std::size_t c = 0; c < columns_.size(); ++c) {
      double prev = data_[r - 1][c];
      double curr = data_[r][c];
      if (!(prev > 0.0) || !(curr > 0.0)) {
        throw std::runtime_error("dataframe::log_changes: non-positive value encountered");
      }
      out.data_[r - 1][c] = std::log(curr) - std::log(prev);
    }
  }
  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::proportional_changes() const {
  if (data_.size() < 2) {
    throw std::runtime_error("dataframe::proportional_changes: need at least two rows");
  }
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_.assign(index_.begin() + 1, index_.end());
  out.index_name_ = index_name_;
  out.data_.resize(data_.size() - 1, std::vector<double>(columns_.size(), 0.0));
  for (std::size_t r = 1; r < data_.size(); ++r) {
    for (std::size_t c = 0; c < columns_.size(); ++c) {
      double prev = data_[r - 1][c];
      double curr = data_[r][c];
      if (prev == 0.0) {
        throw std::runtime_error("dataframe::proportional_changes: zero value encountered");
      }
      out.data_[r - 1][c] = (curr - prev) / prev;
    }
  }
  return out;
}

template <typename IndexT>
DataFrame<std::string> DataFrame<IndexT>::column_stats_dataframe() const {
  static const std::vector<std::string> labels = {"n",       "mean",  "sd",
                                                  "skew",    "ex_kurtosis",
                                                  "min",     "max"};
  DataFrame<std::string> out;
  out.columns_ = columns_;
  out.index_ = labels;
  out.index_name_ = "statistic";
  out.data_.assign(labels.size(), std::vector<double>(columns_.size(), 0.0));

  for (std::size_t c = 0; c < columns_.size(); ++c) {
    std::vector<double> values;
    values.reserve(rows());
    for (std::size_t r = 0; r < rows(); ++r) {
      values.push_back(data_[r][c]);
    }
    stats::SummaryStats summary = stats::summary_stats(values);
    out.data_[0][c] = static_cast<double>(summary.n);
    out.data_[1][c] = summary.mean;
    out.data_[2][c] = summary.sd;
    out.data_[3][c] = summary.skew;
    out.data_[4][c] = summary.ex_kurtosis;
    out.data_[5][c] = summary.min;
    out.data_[6][c] = summary.max;
  }

  return out;
}

template <typename IndexT>
DataFrame<std::string> DataFrame<IndexT>::correlation_matrix() const {
  if (columns_.empty()) {
    throw std::runtime_error("dataframe::correlation_matrix: no columns");
  }
  if (rows() < 2) {
    throw std::runtime_error("dataframe::correlation_matrix: need at least two rows");
  }
  std::vector<std::size_t> valid_rows;
  valid_rows.reserve(rows());
  for (std::size_t r = 0; r < rows(); ++r) {
    bool has_nan = false;
    for (std::size_t c = 0; c < cols(); ++c) {
      const double v = data_[r][c];
      if (!(v == v)) {
        has_nan = true;
        break;
      }
    }
    if (!has_nan) valid_rows.push_back(r);
  }
  if (valid_rows.size() < 2) {
    throw std::runtime_error("dataframe::correlation_matrix: need at least two non-NaN rows");
  }
  DataFrame<std::string> out;
  out.columns_ = columns_;
  out.index_ = columns_;
  out.index_name_ = "column";
  out.data_.assign(columns_.size(), std::vector<double>(columns_.size(), 0.0));

  std::vector<double> means(columns_.size(), 0.0);
  for (std::size_t c = 0; c < columns_.size(); ++c) {
    for (std::size_t r_index : valid_rows) {
      means[c] += data_[r_index][c];
    }
    means[c] /= static_cast<double>(valid_rows.size());
  }

  std::vector<double> sds(columns_.size(), 0.0);
  for (std::size_t c = 0; c < columns_.size(); ++c) {
    double accum = 0.0;
    for (std::size_t r_index : valid_rows) {
      double diff = data_[r_index][c] - means[c];
      accum += diff * diff;
    }
    const double var = accum / static_cast<double>(valid_rows.size() - 1);
    sds[c] = (var > 0.0) ? std::sqrt(var) : 0.0;
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t i = 0; i < columns_.size(); ++i) {
    for (std::size_t j = 0; j < columns_.size(); ++j) {
      if (i == j) {
        out.data_[i][j] = 1.0;
        continue;
      }
      double accum = 0.0;
      for (std::size_t r_index : valid_rows) {
        accum += (data_[r_index][i] - means[i]) * (data_[r_index][j] - means[j]);
      }
      const double cov = accum / static_cast<double>(valid_rows.size() - 1);
      if (sds[i] <= 0.0 || sds[j] <= 0.0) {
        out.data_[i][j] = nan;
      } else {
        out.data_[i][j] = cov / (sds[i] * sds[j]);
      }
    }
  }

  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::add(double value) const {
  return apply_scalar([&](double v) { return v + value; });
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::subtract(double value) const {
  return apply_scalar([&](double v) { return v - value; });
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::multiply(double value) const {
  return apply_scalar([&](double v) { return v * value; });
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::divide(double value) const {
  if (value == 0.0) {
    throw std::runtime_error("dataframe::divide: division by zero");
  }
  return apply_scalar([&](double v) { return v / value; });
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::add(const DataFrame& other) const {
  return apply_binary(other, [](double a, double b) { return a + b; }, "add");
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::subtract(const DataFrame& other) const {
  return apply_binary(other, [](double a, double b) { return a - b; }, "subtract");
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::multiply(const DataFrame& other) const {
  return apply_binary(other, [](double a, double b) { return a * b; }, "multiply");
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::divide(const DataFrame& other) const {
  return apply_binary(other,
                      [](double a, double b) {
                        if (b == 0.0) {
                          throw std::runtime_error("dataframe::divide: division by zero element");
                        }
                        return a / b;
                      },
                      "divide");
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::log_elements() const {
  return apply_unary([](double v) {
    if (std::isnan(v)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    if (!(v > 0.0)) {
      throw std::runtime_error("dataframe::log_elements: non-positive value encountered");
    }
    return std::log(v);
  }, "log_elements");
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::exp_elements() const {
  return apply_unary([](double v) { return std::exp(v); }, "exp_elements");
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::power(double exponent) const {
  return apply_unary([&](double v) { return std::pow(v, exponent); }, "power");
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::power_int(int exponent) const {
  return apply_unary([&](double v) { return std::pow(v, exponent); }, "power_int");
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::standardize() const {
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_ = index_;
  out.index_name_ = index_name_;
  out.data_.assign(rows(), std::vector<double>(cols(), 0.0));
  if (rows() == 0 || cols() == 0) {
    return out;
  }

  std::vector<double> means(cols(), 0.0);
  for (std::size_t r = 0; r < rows(); ++r) {
    for (std::size_t c = 0; c < cols(); ++c) {
      means[c] += data_[r][c];
    }
  }
  for (double& m : means) {
    m /= static_cast<double>(rows());
  }

  std::vector<double> sds(cols(), 0.0);
  for (std::size_t c = 0; c < cols(); ++c) {
    double accum = 0.0;
    for (std::size_t r = 0; r < rows(); ++r) {
      double diff = data_[r][c] - means[c];
      accum += diff * diff;
    }
    if (rows() > 1) {
      const double var = accum / static_cast<double>(rows() - 1);
      sds[c] = (var > 0.0) ? std::sqrt(var) : 0.0;
    }
  }

  for (std::size_t r = 0; r < rows(); ++r) {
    for (std::size_t c = 0; c < cols(); ++c) {
      const double denom = sds[c];
      if (denom > 0.0) {
        out.data_[r][c] = (data_[r][c] - means[c]) / denom;
      } else {
        out.data_[r][c] = 0.0;
      }
    }
  }

  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::normalize() const {
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_ = index_;
  out.index_name_ = index_name_;
  out.data_.assign(rows(), std::vector<double>(cols(), 0.0));
  if (rows() == 0 || cols() == 0) {
    return out;
  }

  std::vector<double> mins(cols());
  std::vector<double> maxs(cols());
  for (std::size_t c = 0; c < cols(); ++c) {
    mins[c] = data_[0][c];
    maxs[c] = data_[0][c];
  }

  for (std::size_t r = 0; r < rows(); ++r) {
    for (std::size_t c = 0; c < cols(); ++c) {
      mins[c] = std::min(mins[c], data_[r][c]);
      maxs[c] = std::max(maxs[c], data_[r][c]);
    }
  }

  for (std::size_t r = 0; r < rows(); ++r) {
    for (std::size_t c = 0; c < cols(); ++c) {
      const double spread = maxs[c] - mins[c];
      if (spread > 0.0) {
        out.data_[r][c] = (data_[r][c] - mins[c]) / spread;
      } else {
        out.data_[r][c] = 0.0;
      }
    }
  }

  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::select_rows(const std::vector<IndexT>& values) const {
  std::vector<std::size_t> positions;
  positions.reserve(values.size());
  for (const auto& v : values) {
    auto it = std::find(index_.begin(), index_.end(), v);
    if (it == index_.end()) {
      throw std::runtime_error("dataframe::select_rows: requested index not found");
    }
    positions.push_back(static_cast<std::size_t>(it - index_.begin()));
  }
  return select_rows_by_positions(positions);
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::select_columns(const std::vector<std::string>& names) const {
  std::vector<std::size_t> positions;
  positions.reserve(names.size());
  for (const auto& name : names) {
    positions.push_back(find_column_index(name));
  }
  return select_columns_by_positions(positions);
}

template <typename IndexT>
template <typename T, typename>
DataFrame<IndexT> DataFrame<IndexT>::slice_rows_range(IndexT start,
                                                      IndexT end,
                                                      bool inclusive_end) const {
  auto positions = find_row_positions_in_range(start, end, inclusive_end);
  return select_rows_by_positions(positions);
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::head_rows(std::size_t count) const {
  if (count == 0) {
    std::vector<std::size_t> empty;
    return select_rows_by_positions(empty);
  }
  if (count >= rows()) {
    return *this;
  }
  std::vector<std::size_t> positions(count);
  for (std::size_t i = 0; i < count; ++i) positions[i] = i;
  return select_rows_by_positions(positions);
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::tail_rows(std::size_t count) const {
  if (count == 0) {
    std::vector<std::size_t> empty;
    return select_rows_by_positions(empty);
  }
  if (count >= rows()) {
    return *this;
  }
  std::vector<std::size_t> positions(count);
  const std::size_t start = rows() - count;
  for (std::size_t i = 0; i < count; ++i) positions[i] = start + i;
  return select_rows_by_positions(positions);
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::head_columns(std::size_t count) const {
  if (count == 0) {
    std::vector<std::size_t> empty;
    return select_columns_by_positions(empty);
  }
  if (count >= cols()) {
    return *this;
  }
  std::vector<std::size_t> positions(count);
  for (std::size_t i = 0; i < count; ++i) positions[i] = i;
  return select_columns_by_positions(positions);
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::tail_columns(std::size_t count) const {
  if (count == 0) {
    std::vector<std::size_t> empty;
    return select_columns_by_positions(empty);
  }
  if (count >= cols()) {
    return *this;
  }
  std::vector<std::size_t> positions(count);
  const std::size_t start = cols() - count;
  for (std::size_t i = 0; i < count; ++i) positions[i] = start + i;
  return select_columns_by_positions(positions);
}

template <typename IndexT>
std::vector<double> DataFrame<IndexT>::column_data(const std::string& name) const {
  std::size_t col = find_column_index(name);
  std::vector<double> values;
  values.reserve(rows());
  for (std::size_t r = 0; r < rows(); ++r) {
    values.push_back(data_[r][col]);
  }
  return values;
}

template <typename IndexT>
std::vector<double> DataFrame<IndexT>::row_data(const IndexT& index_value) const {
  std::size_t pos = find_row_position(index_value);
  return data_[pos];
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::rolling_mean(std::size_t window) const {
  if (window == 0) {
    throw std::runtime_error("dataframe::rolling_mean: window must be positive");
  }
  if (window > rows()) {
    throw std::runtime_error("dataframe::rolling_mean: window exceeds row count");
  }
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_name_ = index_name_;
  out.index_.assign(index_.begin() + static_cast<std::ptrdiff_t>(window - 1), index_.end());
  out.data_.assign(rows() - window + 1, std::vector<double>(cols(), 0.0));

  std::vector<double> sums(cols(), 0.0);
  std::vector<int> valid_counts(cols(), 0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t r = 0; r < rows(); ++r) {
    for (std::size_t c = 0; c < cols(); ++c) {
      double value = data_[r][c];
      const bool is_nan = !(value == value);
      if (!is_nan) {
        sums[c] += value;
        ++valid_counts[c];
      }
      if (r >= window) {
        double old_value = data_[r - window][c];
        const bool old_is_nan = !(old_value == old_value);
        if (!old_is_nan) {
          sums[c] -= old_value;
          --valid_counts[c];
        }
      }
      if (r + 1 >= window) {
        if (valid_counts[c] == static_cast<int>(window)) {
          out.data_[r + 1 - window][c] = sums[c] / static_cast<double>(window);
        } else {
          out.data_[r + 1 - window][c] = nan;
        }
      }
    }
  }

  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::rolling_std(std::size_t window) const {
  if (window == 0) {
    throw std::runtime_error("dataframe::rolling_std: window must be positive");
  }
  if (window > rows()) {
    throw std::runtime_error("dataframe::rolling_std: window exceeds row count");
  }
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_name_ = index_name_;
  out.index_.assign(index_.begin() + static_cast<std::ptrdiff_t>(window - 1), index_.end());
  out.data_.assign(rows() - window + 1, std::vector<double>(cols(), 0.0));

  std::vector<double> sums(cols(), 0.0);
  std::vector<double> sums_sq(cols(), 0.0);
  std::vector<int> valid_counts(cols(), 0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t r = 0; r < rows(); ++r) {
    for (std::size_t c = 0; c < cols(); ++c) {
      double value = data_[r][c];
      const bool is_nan = !(value == value);
      if (!is_nan) {
        sums[c] += value;
        sums_sq[c] += value * value;
        ++valid_counts[c];
      }
      if (r >= window) {
        double old = data_[r - window][c];
        const bool old_is_nan = !(old == old);
        if (!old_is_nan) {
          sums[c] -= old;
          sums_sq[c] -= old * old;
          --valid_counts[c];
        }
      }
      if (r + 1 >= window) {
        double result = nan;
        if (valid_counts[c] == static_cast<int>(window)) {
          if (window == 1) {
            result = 0.0;
          } else {
            double mean = sums[c] / static_cast<double>(window);
            double numerator = sums_sq[c] - sums[c] * mean;
            double variance = numerator / static_cast<double>(window - 1);
            if (variance < 0.0 && variance > -1e-12) {
              variance = 0.0;
            }
            result = (variance > 0.0) ? std::sqrt(variance) : 0.0;
          }
        }
        out.data_[r + 1 - window][c] = result;
      }
    }
  }

  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::rolling_rms(std::size_t window) const {
  if (window == 0) {
    throw std::runtime_error("dataframe::rolling_rms: window must be positive");
  }
  if (window > rows()) {
    throw std::runtime_error("dataframe::rolling_rms: window exceeds row count");
  }
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_name_ = index_name_;
  out.index_.assign(index_.begin() + static_cast<std::ptrdiff_t>(window - 1), index_.end());
  out.data_.assign(rows() - window + 1, std::vector<double>(cols(), 0.0));
  if (cols() == 0) return out;

  std::vector<double> sums_sq(cols(), 0.0);
  std::vector<int> valid_counts(cols(), 0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t r = 0; r < rows(); ++r) {
    for (std::size_t c = 0; c < cols(); ++c) {
      double value = data_[r][c];
      if (value == value) {
        sums_sq[c] += value * value;
        ++valid_counts[c];
      }
      if (r >= window) {
        double old = data_[r - window][c];
        if (old == old) {
          sums_sq[c] -= old * old;
          --valid_counts[c];
        }
      }
      if (r + 1 >= window) {
        if (valid_counts[c] == static_cast<int>(window)) {
          out.data_[r + 1 - window][c] = std::sqrt(sums_sq[c] / static_cast<double>(window));
        } else {
          out.data_[r + 1 - window][c] = nan;
        }
      }
    }
  }

  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::exponential_moving_average(double alpha) const {
  if (!(alpha > 0.0) || !(alpha < 1.0)) {
    throw std::runtime_error(
        "dataframe::exponential_moving_average: alpha must be in (0,1)");
  }
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_ = index_;
  out.index_name_ = index_name_;
  out.data_.assign(rows(), std::vector<double>(cols(), std::numeric_limits<double>::quiet_NaN()));
  if (rows() == 0 || cols() == 0) return out;

  for (std::size_t c = 0; c < cols(); ++c) {
    double ema = std::numeric_limits<double>::quiet_NaN();
    bool has_ema = false;
    for (std::size_t r = 0; r < rows(); ++r) {
      double value = data_[r][c];
      if (!(value == value)) {
        out.data_[r][c] = std::numeric_limits<double>::quiet_NaN();
        continue;
      }
      if (!has_ema) {
        ema = value;
        has_ema = true;
      } else {
        ema = alpha * value + (1.0 - alpha) * ema;
      }
      out.data_[r][c] = ema;
    }
  }

  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::resample_rows(std::size_t sample_size,
                                                   bool reset_index) const {
  if (rows() == 0) {
    throw std::runtime_error("dataframe::resample_rows: no rows to sample");
  }
  if (sample_size == 0) sample_size = rows();
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_name_ = index_name_;
  out.data_.reserve(sample_size);
  out.index_.reserve(sample_size);

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<std::size_t> dist(0, rows() - 1);

  for (std::size_t i = 0; i < sample_size; ++i) {
    std::size_t pick = dist(rng);
    out.data_.push_back(data_[pick]);
    if (reset_index) {
      if constexpr (std::is_convertible_v<std::size_t, IndexT>) {
        out.index_.push_back(static_cast<IndexT>(i));
      } else {
        throw std::runtime_error(
            "dataframe::resample_rows: cannot reset index for this index type");
      }
    } else {
      out.index_.push_back(index_[pick]);
    }
  }

  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::remove_rows_with_nan() const {
  std::vector<std::size_t> keep_positions;
  for (std::size_t r = 0; r < rows(); ++r) {
    bool has_nan = false;
    for (double value : data_[r]) {
      if (std::isnan(value)) {
        has_nan = true;
        break;
      }
    }
    if (!has_nan) {
      keep_positions.push_back(r);
    }
  }
  return select_rows_by_positions(keep_positions);
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::remove_columns_with_nan() const {
  std::vector<std::size_t> keep_positions;
  for (std::size_t c = 0; c < cols(); ++c) {
    bool has_nan = false;
    for (std::size_t r = 0; r < rows(); ++r) {
      if (std::isnan(data_[r][c])) {
        has_nan = true;
        break;
      }
    }
    if (!has_nan) {
      keep_positions.push_back(c);
    }
  }
  return select_columns_by_positions(keep_positions);
}

template <typename IndexT>
template <typename Func>
DataFrame<IndexT> DataFrame<IndexT>::apply_scalar(Func func) const {
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_ = index_;
  out.index_name_ = index_name_;
  out.data_ = data_;
  for (auto& row : out.data_) {
    for (double& value : row) {
      value = func(value);
    }
  }
  return out;
}

template <typename IndexT>
template <typename Func>
DataFrame<IndexT> DataFrame<IndexT>::apply_unary(Func func, const char*) const {
  return apply_scalar(func);
}

template <typename IndexT>
template <typename Func>
DataFrame<IndexT> DataFrame<IndexT>::apply_binary(const DataFrame& other,
                                                  Func func,
                                                  const char* name) const {
  if (rows() != other.rows() || cols() != other.cols()) {
    throw std::runtime_error(std::string("dataframe::") + name + ": shape mismatch");
  }
  if (columns_ != other.columns_) {
    throw std::runtime_error(std::string("dataframe::") + name + ": column mismatch");
  }
  if (index_ != other.index_) {
    throw std::runtime_error(std::string("dataframe::") + name + ": index mismatch");
  }
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_ = index_;
  out.index_name_ = index_name_;
  out.data_.assign(rows(), std::vector<double>(cols(), 0.0));
  for (std::size_t r = 0; r < rows(); ++r) {
    for (std::size_t c = 0; c < cols(); ++c) {
      out.data_[r][c] = func(data_[r][c], other.data_[r][c]);
    }
  }
  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::select_rows_by_positions(
    const std::vector<std::size_t>& positions) const {
  DataFrame<IndexT> out;
  out.columns_ = columns_;
  out.index_name_ = index_name_;
  out.index_.reserve(positions.size());
  out.data_.reserve(positions.size());
  for (std::size_t pos : positions) {
    if (pos >= rows()) {
      throw std::runtime_error("dataframe::select_rows: position out of bounds");
    }
    out.index_.push_back(index_[pos]);
    out.data_.push_back(data_[pos]);
  }
  return out;
}

template <typename IndexT>
DataFrame<IndexT> DataFrame<IndexT>::select_columns_by_positions(
    const std::vector<std::size_t>& positions) const {
  DataFrame<IndexT> out;
  out.index_ = index_;
  out.index_name_ = index_name_;
  out.columns_.reserve(positions.size());
  for (std::size_t pos : positions) {
    if (pos >= cols()) {
      throw std::runtime_error("dataframe::select_columns: position out of bounds");
    }
    out.columns_.push_back(columns_[pos]);
  }
  out.data_.assign(rows(), std::vector<double>(positions.size(), 0.0));
  for (std::size_t r = 0; r < rows(); ++r) {
    for (std::size_t c = 0; c < positions.size(); ++c) {
      out.data_[r][c] = data_[r][positions[c]];
    }
  }
  return out;
}

template <typename IndexT>
std::vector<std::size_t> DataFrame<IndexT>::find_row_positions_in_range(
    IndexT start,
    IndexT end,
    bool inclusive_end) const {
  std::vector<std::size_t> positions;
  if (rows() == 0) return positions;
  IndexT lo = start;
  IndexT hi = end;
  if (hi < lo) std::swap(lo, hi);
  for (std::size_t i = 0; i < index_.size(); ++i) {
    const bool lower_ok = index_[i] >= lo;
    const bool upper_ok = inclusive_end ? (index_[i] <= hi) : (index_[i] < hi);
    if (lower_ok && upper_ok) {
      positions.push_back(i);
    }
  }
  return positions;
}

template <typename IndexT>
std::size_t DataFrame<IndexT>::find_column_index(const std::string& name) const {
  for (std::size_t i = 0; i < columns_.size(); ++i) {
    if (columns_[i] == name) {
      return i;
    }
  }
  throw std::runtime_error("dataframe::select_columns: column not found");
}

template <typename IndexT>
std::size_t DataFrame<IndexT>::find_row_position(const IndexT& value) const {
  for (std::size_t i = 0; i < index_.size(); ++i) {
    if (index_[i] == value) {
      return i;
    }
  }
  throw std::runtime_error("dataframe::select_rows: index not found");
}

template <typename IndexT>
double DataFrame<IndexT>::value(std::size_t row, std::size_t col) const {
  if (row >= data_.size() || col >= columns_.size()) {
    throw std::out_of_range("dataframe::value: index out of range");
  }
  return data_[row][col];
}

using IntDataFrame = DataFrame<int>;
using StringDataFrame = DataFrame<std::string>;

}  // namespace df

#endif


