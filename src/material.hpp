#ifndef __MATERIAL_HPP__
#define __MATERIAL_HPP__

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

class CoordinateTransition {
public:
  CoordinateTransition(const Vector3f &normal);
  Vector3f WorldToLocal(const Vector3f &p) const;
  Vector3f LocalToWorld(const Vector3f &p) const;

private:
  Vector3f m_normal, m_tangent, m_binormal;
};

enum class EMaterialType : int { DIFFUSE = 0, EMPTY };
class Material {
public:
  virtual ~Material() = default;

  virtual Vector3f f(const Vector3f &wo, const Vector3f &wi, const Vector2f &uv,
                     const CoordinateTransition &trans) const       = 0;
  virtual Float    pdf(const Vector3f &wo, const Vector3f &wi,
                       const CoordinateTransition &trans) const     = 0;
  virtual Vector3f sampleF(const Vector3f &wo, Vector3f &wi, Float &pdf,
                           const Vector2f &u, const Vector2f &uv,
                           const CoordinateTransition &trans) const = 0;
  EMaterialType    m_matType;
};

class DiffuseMaterial : public Material {
public:
  DiffuseMaterial(const Vector3f &albedo);
  Vector3f f(const Vector3f &wo, const Vector3f &wi, const Vector2f &uv,
             const CoordinateTransition &trans) const override;
  Float    pdf(const Vector3f &wo, const Vector3f &wi,
               const CoordinateTransition &trans) const override;
  Vector3f sampleF(const Vector3f &wo, Vector3f &wi, Float &pdf,
                   const Vector2f &u, const Vector2f &uv,
                   const CoordinateTransition &trans) const override;

private:
  Vector3f m_albedo;
};

SV_NAMESPACE_END

#endif
