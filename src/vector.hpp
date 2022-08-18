#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

#include <algorithm>
#include <cstddef>
#include <initializer_list>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

inline void Depreciated() {
  static_assert("This function is depreciated");
  assert(false);
}

template <typename T, int N>
struct Vector {
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
  Vector(Vector &x) { std::copy(x.m_vec, x.m_vec + x.size, m_vec); }

  bool operator==(const Vector &rhs) const {
    for (int i = 0; i < size; ++i)
      if (m_vec[i] != rhs.m_vec[i]) return false;
    return true;
  }
  Vector &operator=(const Vector &rhs) {
    std::copy(rhs.m_vec, rhs.m_vec + rhs.size, m_vec);
  }
  template <int N_>
  Vector &operator=(T const (&x)[N_]) {
    std::copy(x, x + N_, m_vec);
  }
  const T &operator[](int i) const { return m_vec[i]; }
  T       &operator[](int i) { return m_vec[i]; }
  Vector   operator-() const {
      return forEach([](const T &x) -> T { return -x; });
  }

  // Notice that type T will not be promoted currently
  // TODO: specification using SSE
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

  // Depreciated Functions from Eigen
  [[noreturn]] float  norm() const { Depreciated(); }
  [[noreturn]] Vector normalized() const { Depreciated(); }
  [[noreturn]] Vector stableNormalized() const { Depreciated(); }
  void                normalize() { Depreciated(); }
  void                stableNormalize() { Depreciated(); }

  // The function implemented as helper
  Vector forEach(std::function<T(T)> func) const {
    Vector res;
    for (int i = 0; i < size; ++i) res[i] = func(m_vec[i]);
    return res;
  }
  Vector forEach(const Vector &rhs, std::function<T(T, T)> func) const {
    Vector res;
    for (int i = 0; i < size; ++i) res[i] = func(m_vec[i], rhs[i]);
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
};

// Type will not be promoted here yet
template <typename T, typename T_, int N>
Vector<T, N> operator*(T_ const &s, Vector<T, N> rhs) {
  return rhs * s;
}

// using Vector3f = Vector<Float, 3>;
// using Vector3d = Vector<int, 3>;
// using Vector2f = Vector<Float, 2>;
// using Vector2d = Vector<int, 2>;
using Vector3f = Eigen::Vector3f;
using Vector3d = Eigen::Vector3d;
using Vector2f = Eigen::Vector2f;
using Vector2d = Eigen::Vector2d;
using Matrix3f = Eigen::Matrix3f;
using Matrix3d = Eigen::Matrix3d;

// currently not used
using Color3f  = Vector3f;
using Normal3f = Vector3f;
using Point3f  = Vector3f;
using Spectrum = Vector3f;

template <typename T, int N>
inline Vector<T, N> Normalize(Vector<T, N> x) {
  return x.normalized();
}

SV_NAMESPACE_END
#endif
