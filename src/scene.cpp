#include "scene.hpp"

#include <oneapi/tbb/cache_aligned_allocator.h>
#include <oneapi/tbb/scalable_allocator.h>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ptree_fwd.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "accel.hpp"
#include "common/aabb.h"
#include "common/camera.h"
#include "common/vector.h"
#include "debug.hpp"
#include "external/embree/eprimitive.hpp"
#include "film.hpp"
#include "fledge.h"
#include "light.hpp"
#include "materials/builtin_materials.hpp"
#include "materials/material_base.hpp"
#include "plymesh.hpp"
#include "primitive.hpp"
#include "shape.hpp"
#include "texture.hpp"
#include "volume.hpp"

FLG_NAMESPACE_BEGIN
namespace pt = boost::property_tree;

Scene::Scene() {}

Scene::Scene(const std::string &filename) : Scene() {
  parseXML(filename);
}

bool Scene::init() {
  m_camera = m_resource.alloc<Camera>(m_origin, m_target, m_up);
  m_film =
      m_resource.alloc<Film>(m_resX, m_resY, m_resource, EFilmBufferType::EAll);
  m_accel = m_resource.alloc<NaiveBVHAccel>(m_primitives, m_resource);
#if 0
  m_volume = std::make_shared<OpenVDBVolume>("assets/wdas_cloud/wdas_cloud_eighth.vdb");
#endif
  SLog("Scene init fin.");
  return true;
}

// @INIT_INTERACTION
bool Scene::intersect(const Ray &ray, SInteraction &isect) const {
  return m_accel->intersect(ray, isect);
}

AABB Scene::getBound() const {
  AABB aabb;
  aabb = m_accel->getBound();
  return aabb;
}

// https://stackoverflow.com/questions/650162/why-cant-the-switch-statement-be-applied-on-strings
constexpr unsigned int hash(const char *s, int off = 0) {
  return !s[off] ? 5381 : (hash(s, off + 1) * 33) ^ s[off];
}

static Vector3f parseVector3f(const std::string &s) {
  Vector3f                 res;
  std::vector<std::string> tokens;
  boost::split(tokens, s, boost::is_any_of(","));
  assert(tokens.size() == 3);
  for (int i = 0; i < 3; ++i) {
    res[i] = std::stof(tokens[i]);
  }

  return res;
}

// Entries of Mitsuba 0.5
static bool setIntegrator(const pt::ptree &tree, Scene &scene) {
  auto type = tree.get<std::string>("<xmlattr>.type");
  SLog("scene.integrator.type = %s", type.c_str());
  switch (hash(type.c_str())) {
    case hash("path"):
      scene.m_integrator_type = EIntegratorType::EPathIntegrator;
      break;
    case hash("volpath"):
      scene.m_integrator_type = EIntegratorType::EVolPathIntegrator;
      break;
    default:
      TODO();
  }

  for (auto &v : tree) {
    switch (hash(v.first.c_str())) {
      case hash("integer"): {
        auto name = v.second.get<std::string>("<xmlattr>.name");
        if (name == "maxDepth") {
          scene.m_maxDepth = v.second.get<int>("<xmlattr>.value");
          SLog("scene.integrator.maxDepth = %d", scene.m_maxDepth);
        }
        break;
      }  // "integer"
    }
  }

  return true;
}

static bool setSensor(const pt::ptree &tree, Scene &scene) {
  auto type = tree.get<std::string>("<xmlattr>.type");
  SLog("scene.sensor.type = %s", type.c_str());
  assert(type == "perspective");

  for (auto &v : tree) {
    switch (hash(v.first.c_str())) {
      case hash("float"): {
        auto name = v.second.get<std::string>("<xmlattr>.name");
        if (name == "fov") {
          scene.m_FoV = v.second.get<Float>("<xmlattr>.value");
          SLog("scene.sensor.fov = %f", scene.m_FoV);
        }
        break;
      }  // "float"
      case hash("string"): {
        break;
      }  // "string"
    }
  }

  auto look_at = tree.get_child("transform.lookat");
  scene.m_up   = parseVector3f(look_at.get<std::string>("<xmlattr>.up"));
  SLog("scene.sensor.transform.lookat.up = %s", scene.m_up.toString().c_str());
  scene.m_origin = parseVector3f(look_at.get<std::string>("<xmlattr>.origin"));
  SLog("scene.sensor.transform.lookat.origin = %s",
       scene.m_origin.toString().c_str());
  scene.m_target = parseVector3f(look_at.get<std::string>("<xmlattr>.target"));
  SLog("scene.sensor.transform.lookat.target = %s",
       scene.m_target.toString().c_str());
  scene.m_SPP = tree.get_child("sampler.integer").get<int>("<xmlattr>.value");
  SLog("scene.sensor.sampler.sampleCount = %d", scene.m_SPP);

  auto film = tree.get_child("film");
  for (auto &v : film) {
    if (v.first != "integer") continue;
    auto name = v.second.get<std::string>("<xmlattr>.name");
    switch (hash(name.c_str())) {
      case hash("height"):
        scene.m_resY = v.second.get<int>("<xmlattr>.value");
        SLog("scene.sensor.film.height = %d", scene.m_resY);
        break;
      case hash("width"):
        scene.m_resX = v.second.get<int>("<xmlattr>.value");
        SLog("scene.sensor.film.width = %d", scene.m_resX);
        break;
    }
  }

  return true;
}

static bool addShape(const pt::ptree &tree, Scene &scene) {
  auto        type     = tree.get<std::string>("<xmlattr>.type");
  std::string shape_id = std::to_string(scene.m_primitives.size());
  SLog("scene.shape%s.type = %s", shape_id.c_str(), type.c_str());

  MaterialDispatcher *mat;

  auto bsdf      = tree.get_child("bsdf");
  auto bsdf_type = bsdf.get<std::string>("<xmlattr>.type");
  SLog("scene.shape%s.bsdf.type = %s", shape_id.c_str(), bsdf_type.c_str());
  switch (hash(bsdf_type.c_str())) {
    case hash("dielectric"): {
      Float intIOR = 1.0, extIOR = 1.0;
      for (auto &v : bsdf) {
        if (v.first != "float") continue;
        auto name = v.second.get<std::string>("<xmlattr>.name");
        switch (hash(name.c_str())) {
          case hash("intIOR"): {
            intIOR = v.second.get<Float>("<xmlattr>.value");
            SLog("scene.shape%s.bsdf.intIOR = %f", shape_id.c_str(), intIOR);
            break;
          };  // "intIOR"
          case hash("extIOR"): {
            extIOR = v.second.get<Float>("<xmlattr>.value");
            SLog("scene.shape%s.bsdf.extIOR = %f", shape_id.c_str(), extIOR);
            break;
          };  // "extIOR"
        }
      }

      mat = MakeMaterialInstance<TransmissionMaterial>(scene.m_resource, extIOR,
                                                       intIOR);
      break;
    }  // "dielectric"
    case hash("diffuse"): {
      mat = MakeMaterialInstance<DiffuseMaterial>(scene.m_resource,
                                                  Vector3f(1.0));
      break;
    }  // "diffuse"
    case hash("roughconductor"): {
      mat = MakeMaterialInstance<MicrofacetMaterial>(scene.m_resource,
                                                     Vector3f(0.3), 0.04);
      break;
    }  // "roughconductor"
    default:
      TODO();
  }
  assert(mat != nullptr);

  switch (hash(type.c_str())) {
    case hash("ply"): {
      for (auto &v : tree) {  // string filter
        if (v.first != "string") continue;
        auto name = v.second.get<std::string>("<xmlattr>.name");
        if (name == "filename") {
          auto filename = v.second.get<std::string>("<xmlattr>.value");
          filename      = scene.getPath(filename);
          SLog("scene.shape%s.filename = %s", shape_id.c_str(),
               filename.c_str());
          scene.m_primitives.push_back(
              scene.m_resource.alloc<EmbreeMeshPrimitive>(
                  filename, scene.m_resource, mat, nullptr,
                  scene.m_resource.alloc<HVolume>(Vector3f{1.0}, Vector3f{0.1},
                                                  -0.877, 1.0)));
        }
      }

      break;
    }  // "ply"
    case hash("sphere"): {
      Vector3f center;
      center.x()   = tree.get<Float>("point.<xmlattr>.x");
      center.y()   = tree.get<Float>("point.<xmlattr>.y");
      center.z()   = tree.get<Float>("point.<xmlattr>.z");
      Float radius = tree.get<Float>("float.<xmlattr>.value");
      SLog("scene.shape%s.center = %s", shape_id.c_str(),
           center.toString().c_str());
      SLog("scene.shape%s.radius = %f", shape_id.c_str(), radius);
#if 1
      scene.m_primitives.push_back(scene.m_resource.alloc<ShapePrimitive>(
          scene.m_resource.alloc<Sphere>(center, radius), mat, nullptr,
          scene.m_resource.alloc<HVolume>(
              Vector3f{0.0001764, 0.00032095, 0.00019617},
              Vector3f{0.031845, 0.031324, 0.030147}, 0.9,
              1.0)));  // TODO
#else
      scene.m_primitives.push_back(scene.m_resource.alloc<ShapePrimitive>(
          scene.m_resource.alloc<Sphere>(center, radius), mat, nullptr,
          scene.m_resource.alloc<HVolume>(Vector3f{3.0}, Vector3f{0.3}, -0.877,
                                          1.0)));  // TODO
#endif
    }  // "sphere"
  }

  return true;
}

static bool addLight(const pt::ptree &tree, Scene &scene) {
  auto type = tree.get<std::string>("<xmlattr>.type");
  assert(type == "envmap");

  for (auto &v : tree) {
    if (v.first != "string") continue;
    // string filter
    auto name = v.second.get<std::string>("<xmlattr>.name");
    if (name == "filename") {
      auto env_texture = std::make_shared<ImageTexture>(
          scene.getPath(v.second.get<std::string>("<xmlattr>.value")));
      scene.m_light.push_back(
          scene.m_resource.alloc<InfiniteAreaLight>(env_texture));
      scene.m_infLight.push_back(scene.m_light[scene.m_light.size() - 1]);
    }
  }

  return true;
}

void Scene::parseXML(const std::string &filename) {
  auto xml_path = std::filesystem::path(filename);
  assert(xml_path.has_parent_path());
  this->m_base_dir = xml_path.parent_path();

  pt::ptree tree;
  pt::read_xml(filename, tree);
  SLog("xml load from %s", filename.c_str());
  auto      scene_version = tree.get<std::string>("scene.<xmlattr>.version");
  pt::ptree scene         = tree.get_child("scene");
  pt::ptree integrator    = scene.get_child("integrator");
  pt::ptree sensor        = scene.get_child("sensor");
  setIntegrator(integrator, *this);
  setSensor(sensor, *this);

  // Parse shapes and emitters
  for (auto &v : scene) {
    switch (hash(v.first.c_str())) {
      case hash("shape"): {
        addShape(v.second, *this);
        break;
      }  // "shape"
      case hash("emitter"): {
        addLight(v.second, *this);
        break;
      }  // "light"
    }
  }
}

path Scene::getPath(const path &asset_path) {
  if (asset_path.is_absolute()) {
    // handle absolute asset path
    return asset_path;
  } else {
    // handle relative path
    return m_base_dir / asset_path;
  }
}

std::string Scene::getPath(const std::string &asset_path) {
  return getPath(path(asset_path)).string();
}

FLG_NAMESPACE_END
