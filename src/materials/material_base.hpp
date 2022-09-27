#ifndef __MATERIAL_BASE_HPP__
#define __MATERIAL_BASE_HPP__

#include "common/dispatcher.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "resource.hpp"

FLG_NAMESPACE_BEGIN

class CoordinateTransition {
public:
  CoordinateTransition(const Vector3f &normal) : m_normal(normal) {
    NormalizeInplace(m_normal);
    if (abs(m_normal[0]) > abs(m_normal[2])) {
      m_binormal = Vector3f(-m_normal[1], m_normal[0], 0);
    } else {
      m_binormal = Vector3f(0, -m_normal[2], m_normal[1]);
    }
    NormalizeInplace(m_binormal);
    m_tangent = Cross(m_binormal, m_normal);

    C(m_normal, m_binormal, m_tangent);
  }

  Vector3f WorldToLocal(const Vector3f &p) const {
    C(p);
    return {Dot(p, m_tangent), Dot(p, m_binormal), Dot(p, m_normal)};
  }

  Vector3f LocalToWorld(const Vector3f &p) const {
    C(p);
    return m_tangent * p[0] + m_binormal * p[1] + m_normal * p[2];
  }

private:
  Vector3f m_normal, m_tangent, m_binormal;
};

class DiffuseMaterial;
class MicrofacetMaterial;
class TransmissionMaterial;
using MaterialsPack =
    TypePack<DiffuseMaterial, MicrofacetMaterial, TransmissionMaterial>;

/**
 * @brief This class serves as the material dynamic dispatcher without using
 * vtable. All the material implementation should be derived from this class.
 * Materials should implement the functions that end with `_impl`.
 */
class MaterialDispatcher
    : public Dispatcher<DiffuseMaterial, MicrofacetMaterial,
                        TransmissionMaterial> {
public:
  using Dispatcher::Dispatcher;

  /**
   * Base Functions
   */
  F_CPU_GPU Vector3f f(const Vector3f &wo, const Vector3f &wi,
                       const Vector2f             &uv,
                       const CoordinateTransition &trans) const {
    // Capture all the parameters
    auto invoker = [&](auto cls) {
      checkType(cls);
      Vector3f wo_l = trans.WorldToLocal(wo), wi_l = trans.WorldToLocal(wi);
      return cls->f_impl(wo_l, wi_l, uv);
    };  // auto invoker()
    return dispatch(invoker);
  }

  F_CPU_GPU Float pdf(const Vector3f &wo, const Vector3f &wi,
                      const CoordinateTransition &trans) const {
    // Capture all the parameters
    auto invoker = [&](auto cls) {
      checkType(cls);
      Vector3f wo_l = trans.WorldToLocal(wo), wi_l = trans.WorldToLocal(wi);
      return cls->pdf_impl(wo_l, wi_l);
    };  // auto invoker()
    return dispatch(invoker);
  }

  F_CPU_GPU Vector3f sampleF(const Vector3f &wo, Vector3f &wi, Float &pdf,
                             const Vector2f &u, const Vector2f &uv,
                             const CoordinateTransition &trans) const {
    // Capture all the parameters
    auto invoker = [&](auto cls) {
      checkType(cls);
      Vector3f wo_l = trans.WorldToLocal(wo), wi_l = trans.WorldToLocal(wi);
      auto     ret = cls->sampleF_impl(wo_l, wi_l, pdf, u, uv);
      wi           = trans.LocalToWorld(wi_l);
      return ret;
    };  // auto invoker()
    return dispatch(invoker);
  }

  F_CPU_GPU bool             isDelta() const { return false; }
  F_CPU_GPU virtual Vector3f getAlbedo() const { return Vector3f(1.0); }

  /**
   * Function implementations
   */
  F_CPU_GPU
  virtual Vector3f f_impl(const Vector3f &wo_l, const Vector3f &wi_l,
                          const Vector2f &uv) const = 0;
  F_CPU_GPU
  virtual Vector3f pdf_impl(const Vector3f &wo_l,
                            const Vector3f &wi_l) const = 0;
  F_CPU_GPU
  virtual Vector3f sampleF_impl(const Vector3f &wo_l, Vector3f &wi_l,
                                Float &pdf, const Vector2f &u,
                                const Vector2f &uv) const = 0;
  F_CPU_GPU
  virtual Vector3f getAlbedo_impl() const = 0;

private:
  constexpr static void checkType(auto cls) {
    static_assert(HasType<typename std::pointer_traits<
                              std::remove_cvref_t<decltype(cls)>>::element_type,
                          type_pack>::value);
  }
};

template <typename T, typename... Args>
std::enable_if<HasType<T, MaterialsPack>::value, MaterialDispatcher *>
MakeMaterialInstance(Args... args, Resource &resource) {
  T *material = resource.alloc<T>(std::forward<Args>(args)...);
  return resource.alloc<MaterialDispatcher>(material);
}

FLG_NAMESPACE_END

#endif
