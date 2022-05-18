#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

#ifdef __GNUC__
#include <execinfo.h>
#endif

#include <math.h>

#include <cstdio>

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

#define CPtr(ptr)   assert(ptr != nullptr)
#define CFloat(val) assert(!isnan(val) && !isinf(val))
#define CVec3(vec3)   \
  do {                \
    CFloat(vec3.x()); \
    CFloat(vec3.y()); \
    CFloat(vec3.z()); \
  } while (false)
#define CNorm(vec3)                            \
  do {                                         \
    if (abs(vec3.norm() - 1.0) > 1e-4) {       \
      SLog(#vec3 ".norm() = %f", vec3.norm()); \
      assert(vec3.norm() == 1.0);              \
    }                                          \
  } while (false)

#endif
