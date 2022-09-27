#ifndef __MATERIAL_BASE_HPP__
#define __MATERIAL_BASE_HPP__

#include "common/dispatcher.h"
#include "common/vector.h"
#include "fledge.h"
#include "materials/material.hpp"

FLG_NAMESPACE_BEGIN

class DiffuseMaterial;
class MicrofacetMaterial;
class TransmissionMaterial;

class MaterialBase : public Dispatcher<DiffuseMaterial, MicrofacetMaterial,
                                       TransmissionMaterial> {
public:
  using Dispatcher::Dispatcher;

  /**
   * Base Functions
   */
  Vector3f f(const Vector3f &wo, const Vector3f &wi, const Vector2f &uv,
             const CoordinateTransition &trans) const {
    // Capture all the parameters
    auto invoker = [&](auto cls) {
      checkType(cls);
      return cls->funcA_impl(a, b);
    };  // auto invoker()
    return dispatch(invoker);
  }
  Float            pdf(const Vector3f &wo, const Vector3f &wi,
                       const CoordinateTransition &trans) const;
  Vector3f         sampleF(const Vector3f &wo, Vector3f &wi, Float &pdf,
                           const Vector2f &u, const Vector2f &uv,
                           const CoordinateTransition &trans) const;
  bool             isDelta() const { return false; }
  virtual Vector3f getAlbedo() const { return Vector3f(1.0); }

  /**
   * Function implementations
   */
  virtual Vector3f f_impl(const Vector3f &wo_l, const Vector3f &wi_l,
                          const Vector2f &uv) const       = 0;
  virtual Vector3f pdf_impl(const Vector3f &wo_l,
                            const Vector3f &wi_l) const   = 0;
  virtual Vector3f sampleF_impl(const Vector3f &wo_l, const Vector3f &wi_l,
                                Float &pdf, const Vector2f &u,
                                const Vector2f &uv) const = 0;

private:
  constexpr static void checkType(auto cls) {
    static_assert(HasType<typename std::pointer_traits<
                              std::remove_cvref_t<decltype(cls)>>::element_type,
                          type_pack>::value);
  }
};

FLG_NAMESPACE_END

#endif
