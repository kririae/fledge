// Template specification C++ 17
template <bool B_ = true, std::enable_if_t<N == 2 && B_, bool> = false>
F_CPU_GPU Vector(T x, T y) : m_vec{x, y} {}
template <bool B_ = true, std::enable_if_t<N == 3 && B_, bool> = false>
F_CPU_GPU Vector(T x, T y, T z) : m_vec{x, y, z} {}

// Introduce an extra template parameter
template <typename U = T>
F_CPU_GPU const std::enable_if_t<N >= 1, U> &x() const {
  return m_vec[0];
}
template <typename U = T>
F_CPU_GPU const std::enable_if_t<N >= 2, U> &y() const {
  return m_vec[1];
}
template <typename U = T>
F_CPU_GPU const std::enable_if_t<N >= 3, U> &z() const {
  return m_vec[2];
}
template <typename U = T>
F_CPU_GPU const std::enable_if_t<N >= 4, U> &w() const {
  return m_vec[3];
}
template <typename U = T>
F_CPU_GPU std::enable_if_t<N >= 1, U> &x() {
  return m_vec[0];
}
template <typename U = T>
F_CPU_GPU std::enable_if_t<N >= 2, U> &y() {
  return m_vec[1];
}
template <typename U = T>
F_CPU_GPU std::enable_if_t<N >= 3, U> &z() {
  return m_vec[2];
}
template <typename U = T>
F_CPU_GPU std::enable_if_t<N >= 4, U> &w() {
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
