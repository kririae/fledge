F_CPU_GPU          Vector(T x, T y) requires(N == 2) : m_vec{x, y} {}
F_CPU_GPU          Vector(T x, T y, T z) requires(N == 3) : m_vec{x, y, z} {}
F_CPU_GPU const T &x() const requires(N >= 1) {
  return m_vec[0];
}
F_CPU_GPU const T &y() const requires(N >= 2) {
  return m_vec[1];
}
F_CPU_GPU const T &z() const requires(N >= 3) {
  return m_vec[2];
}
F_CPU_GPU const T &w() const requires(N >= 4) {
  return m_vec[3];
}
F_CPU_GPU T &x() requires(N >= 1) {
  return m_vec[0];
}
F_CPU_GPU T &y() requires(N >= 2) {
  return m_vec[1];
}
F_CPU_GPU T &z() requires(N >= 3) {
  return m_vec[2];
}
F_CPU_GPU T &w() requires(N >= 4) {
  return m_vec[3];
}
// Specified optimization
F_CPU_GPU Vector operator-() const requires(N == 3) {
  return {-m_vec[0], -m_vec[1], -m_vec[2]};
}
F_CPU_GPU Vector operator*(const Vector &rhs) const requires(N == 3) {
  return {m_vec[0] * rhs.m_vec[0], m_vec[1] * rhs.m_vec[1],
          m_vec[2] * rhs.m_vec[2]};
}
F_CPU_GPU Vector operator/(const Vector &rhs) const requires(N == 3) {
  assert(rhs.m_vec[0] != 0);
  assert(rhs.m_vec[1] != 0);
  assert(rhs.m_vec[2] != 0);
  return {m_vec[0] / rhs.m_vec[0], m_vec[1] / rhs.m_vec[1],
          m_vec[2] / rhs.m_vec[2]};
}
F_CPU_GPU
Vector operator+(const Vector &rhs) const requires(N == 3) {
  return {m_vec[0] + rhs.m_vec[0], m_vec[1] + rhs.m_vec[1],
          m_vec[2] + rhs.m_vec[2]};
}
F_CPU_GPU
Vector operator-(const Vector &rhs) const requires(N == 3) {
  return {m_vec[0] - rhs.m_vec[0], m_vec[1] - rhs.m_vec[1],
          m_vec[2] - rhs.m_vec[2]};
}
F_CPU_GPU
Vector operator*(const T &rhs) const requires(N == 3) {
  return {m_vec[0] * rhs, m_vec[1] * rhs, m_vec[2] * rhs};
}
F_CPU_GPU Vector operator/(const T &rhs) const requires(N == 3) {
  assert(rhs != 0);
  return {m_vec[0] / rhs, m_vec[1] / rhs, m_vec[2] / rhs};
}
F_CPU_GPU Vector &operator=(const Vector &rhs) requires(N == 3) {
  m_vec[0] = rhs.m_vec[0];
  m_vec[1] = rhs.m_vec[1];
  m_vec[2] = rhs.m_vec[2];
  return *this;
}
template <int N_>
F_CPU_GPU Vector &operator=(T const (&x)[N_]) requires(N == 3 && N_ >= 3) {
  m_vec[0] = x[0];
  m_vec[1] = x[1];
  m_vec[2] = x[2];
  return *this;
}
