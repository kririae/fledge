#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <type_traits>

#include "fledge.h"

// ispc optimization
#ifdef USE_ISPC
#include "spec/ispc/vector_ispc.h"
#endif

FLG_NAMESPACE_BEGIN

inline void Deprecated() {
  static_assert("This function is deprecated");
  assert(false);
}

#define EIGEN_FUNC
// #define EIGEN_FUNC __attribute__((deprecated))

using std::abs;
using std::max;
using std::min;
using std::sqrt;

template <typename T, int N,
          std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
struct Vector {
  /** Vector class
   *   implemented only for cpu currently
   * Usage:
   *   Vector(): initialize to zero
   *   Vector({...}): initialize through initializer list
   *   forEach(function): return a new Vector<T, N> with function<T(T)>, i.e.,
   *     accepting one parameters and return the modified element, applied for
   *     each element
   *   forEach(Vector, function): accept another Vector<T, N> as parameter and a
   *     function<T(T, T)>, i.e., accepting two parameters and return x op y,
   *     and return a new Vector
   * Notice that all normal operators are implemented in a piecewise(linear) way
   * Note: All the operations except cast<T_, N_>() will preserve
   * T, i.e., not promote int to float like standard operator
   **/
  using type                = T;
  constexpr static int size = N;

  // data
  T m_vec[N];

// Following the hints from
// https://stackoverflow.com/questions/23757876/sfinae-not-working-although-template-methods-are-used
#include "vector_spec.inc"

  // To support multi platforms, those ctors are trivially implemented
  F_CPU_GPU Vector() {
    for (int i = 0; i < N; ++i) m_vec[i] = 0;
  };
  F_CPU_GPU Vector(T x) {
    for (int i = 0; i < N; ++i) m_vec[i] = x;
  }
  template <int N_>
  F_CPU_GPU Vector(T const (&x)[N_]) {
    static_assert(
        N_ == N,
        "# of elements in initializer list should match the vector size");
    for (int i = 0; i < N; ++i) m_vec[i] = x[i];
  }
  // Copy constructor
  F_CPU_GPU Vector(const Vector &x) {
    for (int i = 0; i < x.size; ++i) m_vec[i] = x.m_vec[i];
  }
  F_CPU_GPU Vector(Vector &&x) {
    std::move(std::begin(x.m_vec), std::end(x.m_vec), m_vec);
  }

  F_CPU_GPU bool operator==(const Vector &rhs) const {
    for (int i = 0; i < size; ++i)
      if (m_vec[i] != rhs.m_vec[i]) return false;
    return true;
  }
  F_CPU_GPU bool operator!=(const Vector &rhs) const {
    return !((*this) == rhs);
  }
  F_CPU_GPU Vector &operator=(const Vector &rhs) {
    for (int i = 0; i < rhs.size; ++i) m_vec[i] = rhs.m_vec[i];
    return *this;
  }
  template <int N_>
  F_CPU_GPU Vector &operator=(T const (&x)[N_]) {
    for (int i = 0; i < N_; ++i) m_vec[i] = x.m_vec[i];
    return *this;
  }
  F_CPU_GPU const T &operator[](int i) const {
    return m_vec[i];
  }
  F_CPU_GPU T &operator[](int i) {
    return m_vec[i];
  }

  // Notice that type T will not be promoted currently
  // TODO: specification using SSE
#if defined(USE_ISPC) && !defined(__CUDACC__)
  Vector operator-() const {
    return forEach([](const T &x) -> T { return -x; });
  }
  Vector operator*(const Vector &rhs) const {
    Vector res;
    ispc::Mul(res.m_vec, m_vec, rhs.m_vec, N);
    return res;
  }
  Vector operator/(const Vector &rhs) const {
    Vector res;
    ispc::Div(res.m_vec, m_vec, rhs.m_vec, N);
    return res;
  }
  Vector operator+(const Vector &rhs) const {
    Vector res;
    ispc::Add(res.m_vec, m_vec, rhs.m_vec, N);
    return res;
  }
  Vector operator-(const Vector &rhs) const {
    Vector res;
    ispc::Sub(res.m_vec, m_vec, rhs.m_vec, N);
    return res;
  }
  Vector operator*(const T &rhs) const {
    Vector res;
    ispc::MulConst(res.m_vec, m_vec, rhs, N);
    return res;
  }
  Vector operator/(const T &rhs) const {
    assert(rhs != 0);
    Vector res;
    ispc::DivConst(res.m_vec, m_vec, rhs, N);
    return res;
  }
#else
  Vector operator-() const {
    return forEach([](const T &x) -> T { return -x; });
  }
  Vector operator*(const Vector &rhs) const {
    return forEach(rhs, [](const T &x, const T &y) -> T { return x * y; });
  }
  Vector operator/(const Vector &rhs) const {
    return forEach(rhs, [](const T &x, const T &y) -> T { return x / y; });
  }
  Vector operator+(const Vector &rhs) const {
    return forEach(rhs, [](const T &x, const T &y) -> T { return x + y; });
  }
  Vector operator-(const Vector &rhs) const {
    return forEach(rhs, [](const T &x, const T &y) -> T { return x - y; });
  }
  Vector operator*(const T &rhs) const {
    return forEach([rhs](const T &x) -> T { return x * rhs; });
  }
  Vector operator/(const T &rhs) const {
    return forEach([rhs](const T &x) -> T { return x / rhs; });
  }
#endif

  F_CPU_GPU Vector &operator+=(const Vector &rhs) {
    return (*this) = (*this) + rhs;
  }
  F_CPU_GPU Vector &operator-=(const Vector &rhs) {
    return (*this) = (*this) - rhs;
  }
  F_CPU_GPU Vector &operator*=(const T &rhs) {
    return (*this) = (*this) * rhs;
  }
  F_CPU_GPU Vector &operator/=(const T &rhs) {
    assert(rhs != 0);
    return (*this) = (*this) / rhs;
  }

  // Depreciated Functions from Eigen
  F_CPU_GPU EIGEN_FUNC T norm() const {
    return Norm(*this);
  }
  F_CPU_GPU EIGEN_FUNC T squaredNorm() const {
    return SquaredNorm(*this);
  }
  F_CPU_GPU EIGEN_FUNC Vector normalized() const {
    return Normalize(*this);
  }
  F_CPU_GPU EIGEN_FUNC Vector stableNormalized() const {
    return Normalize(*this);
  }
  F_CPU_GPU EIGEN_FUNC Vector cwiseInverse() const {
    return 1.0 / (*this);
  }
  F_CPU_GPU EIGEN_FUNC T dot(const Vector &rhs) const {
    return Dot(*this, rhs);
  }
  F_CPU_GPU EIGEN_FUNC Vector cross(const Vector &rhs) const {
    return Cross(*this, rhs);
  }
  F_CPU_GPU EIGEN_FUNC Vector cwiseProduct(const Vector &rhs) const {
    return (*this) * rhs;
  }
  F_CPU_GPU EIGEN_FUNC T maxCoeff() const {
    return MaxElement(*this);
  }
  F_CPU_GPU EIGEN_FUNC T minCoeff() const {
    return MinElement(*this);
  }
  F_CPU_GPU EIGEN_FUNC bool isZero() const {
    return SquaredNorm(*this) == 0;
  }
  F_CPU_GPU EIGEN_FUNC void normalize() {
    (*this) = normalized();
  }
  F_CPU_GPU EIGEN_FUNC void stableNormalize() {
    (*this) = normalized();
  }
  F_CPU_GPU EIGEN_FUNC Vector cwiseMax(const Vector &rhs) const {
    return forEach(rhs, [](T x, T y) -> T { return max(x, y); });
  }
  F_CPU_GPU EIGEN_FUNC Vector cwiseMin(const Vector &rhs) const {
    return forEach(rhs, [](T x, T y) -> T { return min(x, y); });
  }

  // The function implemented as helper
  F_CPU_GPU Vector forEach(std::function<T(T)> func) const {
    Vector res;
    for (int i = 0; i < N; ++i) res[i] = func(m_vec[i]);
    return res;
  }
  F_CPU_GPU Vector forEach(const Vector          &rhs,
                           std::function<T(T, T)> func) const {
    Vector res;
    for (int i = 0; i < N; ++i) res[i] = func(m_vec[i], rhs[i]);
    return res;
  }
  F_CPU_GPU T reduce(std::function<T(T, T)> func) const {
    T res = m_vec[0];
    for (int i = 1; i < N; ++i) res = func(res, m_vec[i]);
    return res;
  }
  template <typename T_, int N_>
  F_CPU_GPU Vector<T_, N_> cast() const {
    // The lower N is selected
    Vector<T_, N_> res;
    for (int i = 0; i < min(N, N_); ++i) res[i] = m_vec[i];
    return res;
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < size - 1; ++i) oss << m_vec[i] << ", ";
    oss << m_vec[size - 1] << "]";
    return oss.str();
  }
  friend std::ostream &operator<<(std::ostream &os, const Vector &x) {
    os << x.toString();
    return os;
  }
};

// Type will not be promoted here yet
template <typename T, typename T_, int N>
F_CPU_GPU Vector<T, N> operator*(T_ const &s, const Vector<T, N> &rhs) {
  return rhs * s;
}

template <typename T, typename T_, int N>
F_CPU_GPU Vector<T, N> operator/(T_ const &s, const Vector<T, N> &rhs) {
  return rhs.forEach([s](T x) -> T { return s / x; });
}

template <typename T, int N>
F_CPU_GPU inline T Sum(const Vector<T, N> &x) {
  return x.reduce([](T x, T y) -> T { return x + y; });
}

template <typename T>
F_CPU_GPU inline T Sum(const Vector<T, 3> &x) {
  return x.m_vec[0] + x.m_vec[1] + x.m_vec[2];
}

template <typename T, int N>
F_CPU_GPU inline T SquaredNorm(const Vector<T, N> &x) {
  return x.forEach([](T x) -> T { return x * x; }).reduce([](T x, T y) -> T {
    return x + y;
  });
}

template <typename T>
F_CPU_GPU inline T SquaredNorm(const Vector<T, 3> &x) {
  const T a = x.m_vec[0];
  const T b = x.m_vec[1];
  const T c = x.m_vec[2];
  return a * a + b * b + c * c;
}

template <typename T, int N>
F_CPU_GPU inline T Norm(const Vector<T, N> &x) {
  return std::sqrt(SquaredNorm(x));
}

template <typename T, int N>
F_CPU_GPU inline Vector<T, N> Normalize(const Vector<T, N> &x) {
  return x / Norm(x);
}

template <typename T, int N>
F_CPU_GPU inline void NormalizeInplace(Vector<T, N> &x) {
  x /= Norm(x);
}

template <typename T, int N>
F_CPU_GPU inline T Dot(const Vector<T, N> &x, const Vector<T, N> &y) {
  return Sum(x * y);
}

template <typename T>
F_CPU_GPU inline Vector<T, 3> Cross(const Vector<T, 3> &x,
                                    const Vector<T, 3> &y) {
  return {x.m_vec[1] * y.m_vec[2] - x.m_vec[2] * y.m_vec[1],
          x.m_vec[2] * y.m_vec[0] - x.m_vec[0] * y.m_vec[2],
          x.m_vec[0] * y.m_vec[1] - x.m_vec[1] * y.m_vec[0]};
}

template <typename T, int N>
F_CPU_GPU inline T MaxElement(const Vector<T, N> &x) {
  return x.reduce([](T x, T y) -> T { return max<T>(x, y); });
}

template <typename T, int N>
F_CPU_GPU inline T MinElement(const Vector<T, N> &x) {
  return x.reduce([](T x, T y) -> T { return min<T>(x, y); });
}

template <typename T, int N>
F_CPU_GPU inline Vector<T, N> Max(const Vector<T, N> &x,
                                  const Vector<T, N> &y) {
  return x.forEach(y, [](T x, T y) -> T { return max(x, y); });
}

template <typename T, int N>
F_CPU_GPU inline Vector<T, N> Min(const Vector<T, N> &x,
                                  const Vector<T, N> &y) {
  return x.forEach(y, [](T x, T y) -> T { return min(x, y); });
}

template <typename T, int N>
F_CPU_GPU inline Vector<T, N> Abs(const Vector<T, N> &x) {
  return x.forEach([](T x) -> T { return abs(x); });
}

template <typename T, int N>
F_CPU_GPU inline Vector<T, N> RandVec() requires(std::is_arithmetic<T>::value) {
  Deprecated();
}

template <typename T, int N>
F_CPU_GPU inline bool Same(const Vector<T, N> &x, const Vector<T, N> &y,
                           Float eps = 1e-5) {
  return Norm(x - y) < eps;
}

using Vector4f = Vector<Float, 4>;
using Vector4d = Vector<int, 4>;
using Vector3f = Vector<Float, 3>;
using Vector3d = Vector<int, 3>;
using Vector2f = Vector<Float, 2>;
using Vector2d = Vector<int, 2>;

// currently not used
using Color3f  = Vector3f;
using Normal3f = Vector3f;
using Point3f  = Vector3f;
using Spectrum = Vector3f;

FLG_NAMESPACE_END

#endif
