#ifndef __FWD_HPP__
#define __FWD_HPP__

#ifdef __GNUC__
#include <execinfo.h>
#endif

#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#if !defined(SV_NAMESPACE_BEGIN)
#define SV_NAMESPACE_BEGIN namespace SmallVolNS {
#endif

#if !defined(SV_NAMESPACE_END)
#define SV_NAMESPACE_END }  // namespace SmallVolNS
#endif

using namespace std::string_literals;

using Float    = float;
using Vector3f = Eigen::Vector3f;
using Vector3d = Eigen::Vector3d;
using Vector2f = Eigen::Vector2f;
using Vector2d = Eigen::Vector2d;
using Matrix3f = Eigen::Matrix3f;
using Matrix3d = Eigen::Matrix3d;

// currently not used
using Color3f = Vector3f;

constexpr Float INF        = std::numeric_limits<Float>::infinity();
constexpr Float PI         = static_cast<Float>(3.14159265358979323846);
constexpr Float PI_OVER2   = PI / 2;
constexpr Float PI_OVER4   = PI / 4;
constexpr Float INV_PI     = static_cast<Float>(0.31830988618379067154);
constexpr Float INV_2PI    = static_cast<Float>(0.15915494309189533577);
constexpr Float INV_4PI    = static_cast<Float>(0.07957747154594766788);
constexpr Float SHADOW_EPS = 1e-4;
constexpr Float NORMAL_EPS = 1e-4;

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

#define SV_Log(format, ...)                                                   \
  do {                                                                     \
    fprintf(stdout, SV_COLOR("[%16s:%3d %14s] " format, SV_FG_GREEN) "\n", \
            __FILENAME__, __LINE__, __func__, ##__VA_ARGS__);              \
  } while (false)
#define SV_Err(format, ...)                                                 \
  do {                                                                   \
    fprintf(stderr, SV_COLOR("[%16s:%3d %14s] " format, SV_FG_RED) "\n", \
            __FILENAME__, __LINE__, __func__, ##__VA_ARGS__);            \
    backtrace();                                                         \
    assert(false);                                                       \
  } while (false)
#define TODO() SV_Err("please implement me")

#define LogFloat(val) SV_Log(#val "=%f", val)
#define LogVec3(vec3) SV_Log(#vec3 "=[%f, %f, %f]", vec3.x(), vec3.y(), vec3.z())

#endif
