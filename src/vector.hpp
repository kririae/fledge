#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

#include <Eigen/Dense>
#include <initializer_list>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

// Temporary implementation inherited from Eigen::Vector
template <typename T, int N>
class Vector : public Eigen::Vector<T, N> {
public:
  using type                = T;
  constexpr static int size = N;

  Vector() = default;
  Vector(T x) requires(N == 2) : Eigen::Vector<T, N>(x, x) {}
  Vector(T x, T y) requires(N == 2) : Eigen::Vector<T, N>(x, y) {}
  Vector(T x) requires(N == 3) : Eigen::Vector<T, N>(x, x, x) {}
  Vector(T x, T y, T z) requires(N == 3) : Eigen::Vector<T, N>(x, y, z) {}
  Vector(std::initializer_list<T> l) : Eigen::Vector<T, N>(l) {}
  Vector(const Vector &x) requires(N == 2)
      : Eigen::Vector<T, N>(x.x(), x.y()) {}
  Vector(const Vector &x) requires(N == 3)
      : Eigen::Vector<T, N>(x.x(), x.y(), x.z()) {}
  Vector(const Eigen::Vector<T, N> &x) : Eigen::Vector<T, N>(x) {}
  Vector normalized() const {
    static_assert("This function should not be used, use Normalize(x) instead");
    return Vector(0);
  }
  void normalize() {
    static_assert(
        "This function should not be used, use x = Normalize(x) instead");
  }

  Vector& operator=(const Vector& x) requires(N == 2) {}
  Vector& operator=(const Vector& x) requires(N == 2) {}
};

using Vector3f = Vector<Float, 3>;
using Vector3d = Vector<int, 3>;
using Vector2f = Vector<Float, 2>;
using Vector2d = Vector<int, 2>;
using Matrix3f = Eigen::Matrix3f;
using Matrix3d = Eigen::Matrix3d;

// currently not used
using Color3f  = Vector3f;
using Normal3f = Vector3f;
using Point3f  = Vector3f;
using Spectrum = Vector3f;

template <typename T, int N>
inline Vector Normalize(Vector x) {
  return x.normalized();
}

SV_NAMESPACE_END
#endif
