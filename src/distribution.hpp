#ifndef __SAMPLER_HPP__
#define __SAMPLER_HPP__

#include <algorithm>
#include <memory>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

// Derived from PBRT
class Dist1D {
public:
  Dist1D(const Float *f, int n) : m_f(f, f + n), m_cdf(n + 1) {
    m_cdf[0] = 0;
    for (int i = 1; i <= n; ++i) m_cdf[i] = m_cdf[i - 1] + m_f[i - 1] / n;
    m_int = m_cdf[n];
    if (m_int == 0)
      for (int i = 1; i < n + 1; ++i) m_cdf[i] = Float(i) / Float(n);
    else
      for (int i = 1; i < n + 1; ++i) m_cdf[i] /= m_int;
  }
  Dist1D(const std::vector<Float> &f) : Dist1D(f.data(), f.size()) {}
  ~Dist1D() = default;
  int size() const { return m_f.size(); }
  int inverseCDF(Float u) const {
    // cdf[n] = 1.0, so assume u \in [0, 1]
    int l = 0, r = size();
    while (r - l > 1) {
      int mid = (l + r) / 2;
      if (m_cdf[mid] <= u) {
        l = mid;
      } else {
        r = mid - 1;
      }
    }

    return l;
  }
  Float dPDF(int index) const { return m_f[index] / (m_int * size()); }
  Float getInt() const { return m_int; }
  Float sampleC(Float u, Float &pdf, int &off) const {
    int idx = inverseCDF(u);
    off     = idx;
    assert(off < int(m_f.size()));
    Float du = u - m_cdf[idx];
    if ((m_cdf[idx + 1] - m_cdf[idx]) > 0) du /= (m_cdf[idx + 1] - m_cdf[idx]);
    pdf = m_f[idx] / m_int;
    return (idx + du) / size();
  }
  Float sampleD(Float u, Float &pdf) const {
    int idx = inverseCDF(u);
    pdf     = dPDF(idx);
    // strictly speaking, u linearly interpolated
    u = (u - m_cdf[idx]) / (m_cdf[idx + 1] - m_cdf[idx]);
    return idx;
  }

private:
  std::vector<Float> m_f, m_cdf;
  Float              m_int;  // integral
};
class Dist2D {
public:
  Dist2D(const Float *f, int nu, int nv) : m_nu(nu), m_nv(nv) {
    for (int v = 0; v < nv; ++v) m_d.emplace_back(new Dist1D(&f[v * nu], nu));
    std::vector<Float> marg;
    for (int v = 0; v < nv; ++v) marg.push_back(m_d[v]->getInt());
    m_marg.reset(new Dist1D(marg));
  }
  Vector2f sampleC(Vector2f u, Float &pdf) const {
    int   offs[2];
    Float pdfs[2];
    Float d1 = m_marg->sampleC(u.y(), pdfs[1], offs[1]);
    C(pdfs[1]);
    Float d0 = m_d[offs[1]]->sampleC(u.x(), pdfs[0], offs[0]);
    C(pdfs[0]);
    pdf = pdfs[0] * pdfs[1];
    C(pdf);
    return {d0, d1};
  }
  Float pdf(Vector2f p) const {
    int u = std::clamp<Float>(p[0] * m_nu, 0, m_nu - 1);
    int v = std::clamp<Float>(p[1] * m_nv, 0, m_nv - 1);
    return m_d[v]->dPDF(u) / m_marg->getInt();
  }

private:
  int                                  m_nu, m_nv;
  std::vector<std::unique_ptr<Dist1D>> m_d;
  std::unique_ptr<Dist1D>              m_marg;
};

SV_NAMESPACE_END

#endif
