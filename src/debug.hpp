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

#include <Eigen/Dense>
#include <memory>
#include <type_traits>

template <typename, std::size_t N = 0>
struct is_vector : std::false_type {};

template <typename T, std::size_t N>
struct is_vector<Eigen::Vector<T, N>> {
  static constexpr bool value = std::is_arithmetic<T>::value;
};

template <typename T, std::size_t N>
struct is_vector<Eigen::Vector<T, N>, N> {
  static constexpr bool value = std::is_arithmetic<T>::value;
};

template <typename>
struct is_shared_ptr : std::false_type {};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template <typename>
struct is_unique_ptr : std::false_type {};

template <typename T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type {};

template <typename T>
T C(T v) {
  static_assert("Cannot check the type");
  return v;
}

// naive pointer check
template <typename T,
          std::enable_if_t<std::is_pointer_v<T> || is_shared_ptr<T>::value ||
                           is_unique_ptr<T>::value>>
T C(T v) {
  assert(v != nullptr);
  return v;
}

// naive value check
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>>>
T C(T v) {
  assert(!isnan(v));
  assert(!isinf(v));
  return v;
}

template <typename T, std::size_t N, std::enable_if_t<is_vector<T, N>::value>>
T C(T v) {
  for (size_t i = 0; i < N; ++i) C(v[i]);
  C(v.norm());
  return v;
}

#endif
