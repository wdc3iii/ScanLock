#pragma once

// Resolve a map-PCD parameter into a loadable path.
//
// Accepted forms (after $ENV expansion):
//   - absolute path            -> used as-is
//   - bare name / relative path -> resolved under <package src>/pcd/ (legacy)
//
// $VAR and ${VAR} tokens are expanded from the environment; an unset
// variable throws (better a clear error than a mangled path handed to PCL).
//
// NOTE: an identical copy of this header lives in
// relocalization_bringup/relocalization_bringup/pcd_path.h (kept duplicated
// to avoid a cross-package dependency for ~40 lines) -- keep them in sync.

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace pcd_path {

inline std::string expand_env_vars(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size();) {
    if (in[i] != '$') {
      out += in[i++];
      continue;
    }
    size_t start = i + 1;
    const bool braced = start < in.size() && in[start] == '{';
    if (braced) ++start;
    size_t end = start;
    while (end < in.size() &&
           (std::isalnum(static_cast<unsigned char>(in[end])) || in[end] == '_')) {
      ++end;
    }
    if (end == start) {  // lone '$' (or "${}") -- keep it literal
      out += in[i++];
      continue;
    }
    const std::string name = in.substr(start, end - start);
    if (braced) {
      if (end >= in.size() || in[end] != '}') {
        throw std::runtime_error("Unterminated ${ in path: " + in);
      }
      ++end;
    }
    const char* val = std::getenv(name.c_str());
    if (val == nullptr) {
      throw std::runtime_error("Environment variable '" + name +
                               "' is not set (required by PCD path: " + in + ")");
    }
    out += val;
    i = end;
  }
  return out;
}

inline std::string resolve_pcd_path(const std::string& raw, const char* root_dir) {
  const std::string expanded = expand_env_vars(raw);
  if (!expanded.empty() && expanded.front() == '/') {
    return expanded;
  }
  return std::string(root_dir) + "pcd/" + expanded;
}

}  // namespace pcd_path
