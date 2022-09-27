#include "material_base.hpp"

#include <type_traits>

#include "builtin_materials.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

F_CPU_GPU Vector3f MaterialDispatcher::f(
    const Vector3f &w_wo, const Vector3f &w_wi, const Vector2f &uv,
    const CoordinateTransition &trans) const {
  // Capture all the parameters
  auto invoker = [&](auto cls) {
    checkType(cls);
    Vector3f wo = trans.WorldToLocal(w_wo), wi = trans.WorldToLocal(w_wi);
    return cls->f_impl(wo, wi, uv);
  };  // auto invoker()
  return dispatch(invoker);
}

F_CPU_GPU Float
MaterialDispatcher::pdf(const Vector3f &w_wo, const Vector3f &w_wi,
                        const CoordinateTransition &trans) const {
  // Capture all the parameters
  auto invoker = [&](auto cls) {
    checkType(cls);
    Vector3f wo = trans.WorldToLocal(w_wo), wi = trans.WorldToLocal(w_wi);
    return cls->pdf_impl(wo, wi);
  };  // auto invoker()
  return dispatch(invoker);
}

F_CPU_GPU Vector3f MaterialDispatcher::sampleF(
    const Vector3f &w_wo, Vector3f &w_wi, Float &pdf, const Vector2f &u,
    const Vector2f &uv, const CoordinateTransition &trans) const {
  // Capture all the parameters
  auto invoker = [&](auto cls) {
    checkType(cls);
    Vector3f wo = trans.WorldToLocal(w_wo), wi = trans.WorldToLocal(w_wi);
    auto     ret = cls->sampleF_impl(wo, wi, pdf, u, uv);
    w_wi         = trans.LocalToWorld(wi);
    return ret;
  };  // auto invoker()
  return dispatch(invoker);
}

FLG_NAMESPACE_END
