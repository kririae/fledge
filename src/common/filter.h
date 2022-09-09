#include "common/vector.h"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

// adapted from
// https://pbr-book.org/3ed-2018/Sampling_and_Reconstruction/Image_Reconstruction
struct Filter {
public:
  Filter(const Vector2f &radius) : m_radius(radius), m_inv_radius(1 / radius) {}
  virtual ~Filter()                     = default;
  virtual Float eval(const Vector2f &p) = 0;
  Vector2f      m_radius, m_inv_radius;
};

struct GaussianFilter : public Filter {
public:
  GaussianFilter(const Vector2f &radius, Float alpha)
      : Filter(radius),
        m_alpha(alpha),
        m_expX(std::exp(-alpha * radius.x() * radius.x())),
        m_expY(std::exp(-alpha * radius.y() * radius.y())) {}
  ~GaussianFilter() override = default;
  Float eval(const Vector2f &p) override {
    return gauss(p.x(), m_expX) * gauss(p.y(), m_expY);
  }

private:
  Float m_alpha, m_expX, m_expY;
  Float gauss(Float d, Float exp_) const {
    return std::max((Float)0, Float(std::exp(-m_alpha * d * d) - exp_));
  }
};

struct MitchellFilter : public Filter {
public:
  MitchellFilter(const Vector2f &radius, Float B, Float C)
      : Filter(radius), m_B(B), m_C(C) {}
  ~MitchellFilter() override = default;
  Float eval(const Vector2f &p) override {
    return mitchell1D(p.x() * m_inv_radius.x()) *
           mitchell1D(p.y() * m_inv_radius.y());
  }
  Float mitchell1D(Float x) const {
    x = std::abs(2 * x);
    if (x > 1)
      return ((-m_B - 6 * m_C) * x * x * x + (6 * m_B + 30 * m_C) * x * x +
              (-12 * m_B - 48 * m_C) * x + (8 * m_B + 24 * m_C)) *
             (1.f / 6.f);
    else
      return ((12 - 9 * m_B - 6 * m_C) * x * x * x +
              (-18 + 12 * m_B + 6 * m_C) * x * x + (6 - 2 * m_B)) *
             (1.f / 6.f);
  }

private:
  const Float m_B, m_C;
};

FLG_NAMESPACE_END
