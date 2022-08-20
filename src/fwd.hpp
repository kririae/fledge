#ifndef __FWD_HPP__
#define __FWD_HPP__

#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#if !defined(SV_NAMESPACE_BEGIN)
#define SV_NAMESPACE_BEGIN namespace SmallVolNS {
#endif

#if !defined(SV_NAMESPACE_END)
#define SV_NAMESPACE_END }  // namespace SmallVolNS
#endif

using namespace std::string_literals;

using Float                = float;
constexpr Float INF        = std::numeric_limits<Float>::infinity();
constexpr Float PI         = static_cast<Float>(3.14159265358979323846);
constexpr Float PI_OVER2   = PI / 2;
constexpr Float PI_OVER4   = PI / 4;
constexpr Float INV_PI     = static_cast<Float>(0.31830988618379067154);
constexpr Float INV_2PI    = static_cast<Float>(0.15915494309189533577);
constexpr Float INV_4PI    = static_cast<Float>(0.07957747154594766788);
constexpr Float SHADOW_EPS = 1e-5;
constexpr Float NORMAL_EPS = 1e-5;

#include "debug.hpp"

SV_NAMESPACE_BEGIN

class AABB;
class Accel;
class AreaLight;
class Camera;
class ConstTexture;
class CoordinateTransition;
class DiffuseMaterial;
class Film;
class HVolume;
class ImageTexture;
class InfiniteAreaLight;
class Integrator;
class Interaction;
class Light;
class Material;
class NaiveAccel;
class PathIntegrator;
class Primitive;
class Random;
class Ray;
class Render;
class SampleIntegrator;
class ShapePrimitive;
class Shape;
class Triangle;
struct TriangleMesh;
class Scene;
class SInteraction;
class Sphere;
class SVolIntegrator;
class Texture;
class OpenVDBVolume;
class VInteraction;
class Volume;

SV_NAMESPACE_END

#endif
