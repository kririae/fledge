#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#ifdef __GNUC__
#include <execinfo.h>
#endif

#include <math.h>
#include <stddef.h>
#include <stdio.h>

// keep assert in release mode
#undef NDEBUG
#include <assert.h>

#define SV_FG_BLACK        "\33[1;30m"
#define SV_FG_RED          "\33[1;31m"
#define SV_FG_GREEN        "\33[1;32m"
#define SV_FG_YELLOW       "\33[1;33m"
#define SV_FG_BLUE         "\33[1;34m"
#define SV_FG_MAGENTA      "\33[1;35m"
#define SV_FG_CYAN         "\33[1;36m"
#define SV_FG_WHITE        "\33[1;37m"
#define SV_NONE            "\33[0m"
#define SV_COLOR(str, col) col str SV_NONE
#define __FILENAME__                                                       \
  (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 \
                                    : __FILE__)

inline void backtrace() {
#ifdef __GNUC__
  void  *array[10];
  char **strings;
  int    size, i;

  size    = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  if (strings != NULL) {
    printf("Obtained %d stack frames.\n", size);
    for (i = 0; i < size; i++) printf("  %s\n", strings[i]);
  }

  free(strings);
#endif
}

#define SLog(format, ...)                                                  \
  do {                                                                     \
    fprintf(stdout, SV_COLOR("[%16s:%3d %14s] " format, SV_FG_GREEN) "\n", \
            __FILENAME__, __LINE__, __func__, ##__VA_ARGS__);              \
  } while (false)
#define SErr(format, ...)                                                \
  do {                                                                   \
    fprintf(stderr, SV_COLOR("[%16s:%3d %14s] " format, SV_FG_RED) "\n", \
            __FILENAME__, __LINE__, __func__, ##__VA_ARGS__);            \
    backtrace();                                                         \
    assert(false);                                                       \
  } while (false)
#define TODO() SErr("please implement me")

#define LFloat(val) SLog(#val "=%f", val)
#define LVec3(vec3) SLog(#vec3 "=[%f, %f, %f]", vec3.x(), vec3.y(), vec3.z())
#define LClass(cls) SLog(#cls "=%s", cls.toString().c_str())

#include <memory>
#include <type_traits>

#include "common/vector.h"

template <class T, template <class...> class Template>
struct is_specialization : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

template <typename>
struct is_vector : std::false_type {};

template <typename T, int N>
struct is_vector<fledge::Vector<T, N>> : std::true_type {};

// template <typename T>
// using is_vector = typename std::is_base_of<Eigen::MatrixBase<T>, T>;

template <typename>
struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template <typename>
struct is_unique_ptr : std::false_type {};

template <typename T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type {};

#if defined(__clang__)
#include <experimental/source_location>
using source_location = std::experimental::source_location;
#elif defined(__GNUC__)
#include <source_location>
using source_location = std::source_location;
#endif

#if 1
// naive pointer check
template <typename T,
          std::enable_if_t<std::is_pointer_v<T> || is_shared_ptr<T>::value ||
                               is_unique_ptr<T>::value,
                           bool> = true>
inline void C(T                      v,
              const source_location &location = source_location::current()) {
  if (v == nullptr)
    SErr("check pointer failed in %s:%d; nullptr found", location.file_name(),
         location.line());
}

// naive value check
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
inline void C(T                      v,
              const source_location &location = source_location::current()) {
  if (isnan(v) || isinf(v))
    SErr("check value failed in %s:%d; nan or inf found", location.file_name(),
         location.line());
}

template <typename T, std::enable_if_t<is_vector<T>::value, bool> = true>
inline void C(T                      v,
              const source_location &location = source_location::current()) {
  for (int i = 0; i < v.size; ++i) C(v[i], location);
  C(v.norm(), location);
}

template <typename T1, typename T2>
inline void C(T1 v1, T2 v2,
              const source_location &location = source_location::current()) {
  C(v1, location);
  C(v2, location);
}

template <typename T1, typename T2, typename T3>
inline void C(T1 v1, T2 v2, T3 v3,
              const source_location &location = source_location::current()) {
  C(v1, location);
  C(v2, location);
  C(v3, location);
}

template <typename T1, typename T2, typename T3, typename T4>
inline void C(T1 v1, T2 v2, T3 v3, T4 v4,
              const source_location &location = source_location::current()) {
  C(v1, location);
  C(v2, location);
  C(v3, location);
  C(v4, location);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
inline void C(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5,
              const source_location &location = source_location::current()) {
  C(v1, location);
  C(v2, location);
  C(v3, location);
  C(v4, location);
  C(v5, location);
}
#elif 0
// OMG.. WTF is this... ;(

struct Checker {
  template <typename T>
  inline std::enable_if_t<std::is_pointer_v<T> || is_shared_ptr<T>::value ||
                              is_unique_ptr<T>::value,
                          Checker &>
  operator<<(const std::pair<T, source_location> &v) {
    if (v.first == nullptr)
      SErr("check pointer failed in %s:%d; nullptr found", v.second.file_name(),
           v.second.line());
    return *this;
  }

  template <typename T>
  inline std::enable_if_t<std::is_arithmetic_v<T>, Checker &> operator<<(
      const std::pair<T, source_location> &v) {
    if (isnan(v.first) || isinf(v.first))
      SErr("check value failed in %s:%d; nan or inf found",
           v.second.file_name(), v.second.line());
    return *this;
  }

  template <typename T>
  inline std::enable_if_t<is_vector<T>::value, Checker &> operator<<(
      const std::pair<T, source_location> &v) {
    Checker checker;
    for (int i = 0; i < v.first.size; ++i)
      checker << std::make_pair(v.first[i], v.second);
    checker << std::make_pair(v.first.norm(), v.second);
    return *this;
  }
};

template <typename... Ts>
struct C_ {
  explicit C_(Ts &&...ts,
              const source_location &source = source_location::current()) {
    Checker checker;
    ((checker << std::make_pair(std::forward<Ts>(ts), source)), ...);
  }
};

template <typename... Ts>
C_(Ts &&...args) -> C_<Ts...>;

// To avoid "error: conflicting declaration ‘C<...auto...> a’"
#define C(...) C_(1, ##__VA_ARGS__)
#endif

#endif
