#ifndef __STATS_HPP__
#define __STATS_HPP__

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

class AccumMeasure {
public:
  void        addCount() { ++m_count; }
  std::string toString() const {
    std::ostringstream oss;
    oss << m_count;
    return oss.str();
  }

private:
  std::atomic<uint64_t> m_count;
};

class AverageMeasure {
public:
  void addValue(Float value) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_value *= m_count;
    m_value += value;
    m_count++;
    m_value /= m_count;
    // lock release
  }
  std::string toString() const {
    std::ostringstream oss;
    oss << m_value;
    return oss.str();
  }

private:
  std::mutex m_mutex;
  double     m_value;
  uint64_t   m_count;
};

#define __ACC_MEASURE_NAME(name) name##_accumulate
#define __AVE_MEASURE_NAME(name) name##_average
#define DEF_ACC_MEASURE(name)    AccumMeasure __ACC_MEASURE_NAME(name)
#define DEF_AVE_MEASURE(name)    AverageMeasure __AVE_MEASURE_NAME(name)
#define EXT_ACC_MEASURE(name)    extern AccumMeasure __ACC_MEASURE_NAME(name)
#define EXT_AVE_MEASURE(name)    extern AverageMeasure __AVE_MEASURE_NAME(name)
#define ADD_ACC_MEASURE(name)            \
  {                                      \
    EXT_ACC_MEASURE(name);               \
    __ACC_MEASURE_NAME(name).addCount(); \
  }
#define ADD_AVE_MEASURE(name, val)          \
  {                                         \
    EXT_AVE_MEASURE(name);                  \
    __AVE_MEASURE_NAME(name).addValue(val); \
  }
#define PRINT_ACC_MEASURE(name)       \
  {                                   \
    EXT_ACC_MEASURE(name);            \
    LClass(__ACC_MEASURE_NAME(name)); \
  }
#define PRINT_AVE_MEASURE(name)                                                \
  {                                                                            \
    EXT_AVE_MEASURE(name);                                                     \
    SLog("average " #name "=%s", __AVE_MEASURE_NAME(name).toString().c_str()); \
  }

SV_NAMESPACE_END

#endif
