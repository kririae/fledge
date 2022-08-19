#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

#include "debug.hpp"
#include "fwd.hpp"

SV_NAMESPACE_BEGIN

inline void Depreciated() {
  static_assert("This function is depreciated");
  assert(false);
}

template <typename T, int N>
requires(std::is_arithmetic<T>::value) struct Vector {
  // Vector class
  //   implemented only for cpu currently
  // Usage:
  //   Vector(): initialize to zero
  //   Vector({...}): initialize through initializer list
  //   forEach(function): return a new Vector<T, N> with function<T(T)>, i.e.,
  //     accepting one parameters and return the modified element, applied for
  //     each element
  //   forEach(Vector, function): accept another Vector<T, N> as parameter and a
  //     function<T(T, T)>, i.e., accepting two parameters and return x op y,
  //     and return a new Vector
  // Notice that all normal operators are implemented in a piecewise(linear) way
  // Note: All the operations except cast<T_, N_>() will preserve
  // T, i.e., not promote int to float like standard operator
  using type                = T;
  constexpr static int size = N;

  // data
  T m_vec[N];

  Vector() { std::fill_n(m_vec, N, 0); };
  Vector(T x) { std::fill_n(m_vec, N, x); }
  Vector(T x, T y) requires(N == 2) : m_vec{x, y} {}
  Vector(T x, T y, T z) requires(N == 3) : m_vec{x, y, z} {}
  template <int N_>
  Vector(T const (&x)[N_]) {
    static_assert(
        N_ == N,
        "# of elements in initializer list should match the vector size");
    std::copy(x, x + N_, m_vec);
  }
  // Copy constructor
  Vector(const Vector &x) { std::copy(x.m_vec, x.m_vec + x.size, m_vec); }
  const T &x() const requires(N >= 1) { return m_vec[0]; }
  const T &y() const requires(N >= 2) { return m_vec[1]; }
  const T &z() const requires(N >= 3) { return m_vec[2]; }
  const T &w() const requires(N >= 4) { return m_vec[3]; }
  T       &x() requires(N >= 1) { return m_vec[0]; }
  T       &y() requires(N >= 2) { return m_vec[1]; }
  T       &z() requires(N >= 3) { return m_vec[2]; }
  T       &w() requires(N >= 4) { return m_vec[3]; }

  bool operator==(const Vector &rhs) const {
    for (int i = 0; i < size; ++i)
      if (m_vec[i] != rhs.m_vec[i]) return false;
    return true;
  }
  bool    operator!=(const Vector &rhs) const { return !((*this) == rhs); }
  Vector &operator=(const Vector &rhs) {
    std::copy(rhs.m_vec, rhs.m_vec + rhs.size, m_vec);
    return *this;
  }
  template <int N_>
  Vector &operator=(T const (&x)[N_]) {
    std::copy(x, x + N_, m_vec);
    return *this;
  }
  const T &operator[](int i) const { return m_vec[i]; }
  T       &operator[](int i) { return m_vec[i]; }

  // Notice that type T will not be promoted currently
  // TODO: specification using SSE
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

  Vector &operator+=(const Vector &rhs) { return (*this) = (*this) + rhs; }
  Vector &operator-=(const Vector &rhs) { return (*this) = (*this) - rhs; }
  Vector &operator*=(const T &rhs) { return (*this) = (*this) * rhs; }
  Vector &operator/=(const T &rhs) { return (*this) = (*this) / rhs; }

  // Depreciated Functions from Eigen
  T      norm() const { return Norm(*this); }
  T      squaredNorm() const { return SquaredNorm(*this); }
  Vector normalized() const { return Normalize(*this); }
  Vector stableNormalized() const { return Normalize(*this); }
  Vector cwiseInverse() const { return 1.0 / (*this); }
  T      dot(const Vector &rhs) const { return Dot(*this, rhs); }
  Vector cross(const Vector &rhs) const { return Cross(*this, rhs); }
  Vector cwiseProduct(const Vector &rhs) const { return (*this) * rhs; }
  T      maxCoeff() const { return MaxElement(*this); }
  T      minCoeff() const { return MinElement(*this); }
  bool   isZero() const { return SquaredNorm(*this) == 0; }

  void   normalize() { (*this) = normalized(); }
  void   stableNormalize() { (*this) = normalized(); }
  Vector cwiseMax(const Vector &rhs) const {
    return forEach(rhs, [](T x, T y) -> T { return std::max(x, y); });
  }
  Vector cwiseMin(const Vector &rhs) const {
    return forEach(rhs, [](T x, T y) -> T { return std::min(x, y); });
  }

  // The function implemented as helper
  Vector forEach(std::function<T(T)> func) const {
    Vector res;
    for (int i = 0; i < N; ++i) res[i] = func(m_vec[i]);
    return res;
  }
  Vector forEach(const Vector &rhs, std::function<T(T, T)> func) const {
    Vector res;
    for (int i = 0; i < N; ++i) res[i] = func(m_vec[i], rhs[i]);
    return res;
  }
  T reduce(std::function<T(T, T)> func) const {
    T res = m_vec[0];
    for (int i = 1; i < N; ++i) res = func(res, m_vec[i]);
    return res;
  }
  template <typename T_, int N_>
  Vector<T_, N_> cast() const {
    // The lower N is selected
    Vector<T_, N_> res;
    for (int i = 0; i < std::min(N, N_); ++i) res[i] = m_vec[i];
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

  // Specified optimization
  Vector operator-() const requires(N == 3) {
    return {-m_vec[0], -m_vec[1], -m_vec[2]};
  }
  Vector operator*(const Vector &rhs) const requires(N == 3) {
    return {m_vec[0] * rhs.m_vec[0], m_vec[1] * rhs.m_vec[1],
            m_vec[2] * rhs.m_vec[2]};
  }
  Vector operator/(const Vector &rhs) const requires(N == 3) {
    return {m_vec[0] / rhs.m_vec[0], m_vec[1] / rhs.m_vec[1],
            m_vec[2] / rhs.m_vec[2]};
  }
  Vector operator+(const Vector &rhs) const requires(N == 3) {
    return {m_vec[0] + rhs.m_vec[0], m_vec[1] + rhs.m_vec[1],
            m_vec[2] + rhs.m_vec[2]};
  }
  Vector operator-(const Vector &rhs) const requires(N == 3) {
    return {m_vec[0] - rhs.m_vec[0], m_vec[1] - rhs.m_vec[1],
            m_vec[2] - rhs.m_vec[2]};
  }
  Vector operator*(const T &rhs) const requires(N == 3) {
    return {m_vec[0] * rhs, m_vec[1] * rhs, m_vec[2] * rhs};
  }
  Vector operator/(const T &rhs) const requires(N == 3) {
    return {m_vec[0] / rhs, m_vec[1] / rhs, m_vec[2] / rhs};
  }
  Vector &operator=(const Vector &rhs) requires(N == 3) {
    m_vec[0] = rhs.m_vec[0];
    m_vec[1] = rhs.m_vec[1];
    m_vec[2] = rhs.m_vec[2];
    return *this;
  }
  template <int N_>
  Vector &operator=(T const (&x)[N_]) requires(N == 3) {
    m_vec[0] = x[0];
    m_vec[1] = x[1];
    m_vec[2] = x[2];
    return *this;
  }
};

// Type will not be promoted here yet
template <typename T, typename T_, int N>
Vector<T, N> operator*(T_ const &s, const Vector<T, N> &rhs) {
  return rhs * s;
}

template <typename T, typename T_, int N>
Vector<T, N> operator/(T_ const &s, const Vector<T, N> &rhs) {
  return rhs.forEach([s](T x) -> T { return s / x; });
}

template <typename T, int N>
inline T Sum(const Vector<T, N> &x) requires(std::is_arithmetic<T>::value) {
  return x.reduce([](T x, T y) -> T { return x + y; });
}

template <typename T, int N>
inline T SquaredNorm(const Vector<T, N> &x) requires(
    std::is_arithmetic<T>::value) {
  return x.forEach([](T x) -> T { return x * x; }).reduce([](T x, T y) -> T {
    return x + y;
  });
}

template <typename T, int N>
inline T Norm(const Vector<T, N> &x) requires(std::is_arithmetic<T>::value) {
  return std::sqrt(SquaredNorm(x));
}

template <typename T, int N>
inline Vector<T, N> Normalize(const Vector<T, N> &x) requires(
    std::is_arithmetic<T>::value) {
  return x / Norm(x);
}

template <typename T, int N>
inline T Dot(Vector<T, N> x,
             Vector<T, N> y) requires(std::is_arithmetic<T>::value) {
  return Sum(x.forEach(y, [](T a, T b) -> T { return a * b; }));
}

template <typename T>
inline Vector<T, 3> Cross(Vector<T, 3> x, Vector<T, 3> y) requires(
    std::is_arithmetic<T>::value) {
  return {x.m_vec[1] * y.m_vec[2] - x.m_vec[2] * y.m_vec[1],
          x.m_vec[2] * y.m_vec[0] - x.m_vec[0] * y.m_vec[2],
          x.m_vec[0] * y.m_vec[1] - x.m_vec[1] * y.m_vec[0]};
}

template <typename T, int N>
inline T MaxElement(Vector<T, N> x) requires(std::is_arithmetic<T>::value) {
  return x.reduce([](T x, T y) -> T { return std::max(x, y); });
}

template <typename T, int N>
inline T MinElement(Vector<T, N> x) requires(std::is_arithmetic<T>::value) {
  return x.reduce([](T x, T y) -> T { return std::max(x, y); });
}

template <typename T, int N>
inline Vector<T, N> RandVec() requires(std::is_arithmetic<T>::value) {
  TODO();
}

using Vector3f = Vector<Float, 3>;
using Vector3d = Vector<int, 3>;
using Vector2f = Vector<Float, 2>;
using Vector2d = Vector<int, 2>;
// using Vector3f = Eigen::Vector3f;
// using Vector3d = Eigen::Vector3d;
// using Vector2f = Eigen::Vector2f;
// using Vector2d = Eigen::Vector2d;
using Matrix3f = Eigen::Matrix3f;
using Matrix3d = Eigen::Matrix3d;

// currently not used
using Color3f  = Vector3f;
using Normal3f = Vector3f;
using Point3f  = Vector3f;
using Spectrum = Vector3f;

SV_NAMESPACE_END

#endif
