#include "scene.hpp"

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
#include "camera.hpp"
#include "common/aabb.h"
#include "common/vector.h"
#include "debug.hpp"
#include "spec/embree/eprimitive.hpp"
#include "film.hpp"
#include "fledge.h"
#include "light.hpp"
#include "material.hpp"
#include "plymesh.hpp"
#include "primitive.hpp"
#include "shape.hpp"
#include "texture.hpp"
#include "volume.hpp"

FLG_NAMESPACE_BEGIN
namespace pt = boost::property_tree;

Scene::Scene() {}

Scene::Scene(const std::string &filename) {
  *this = parseXML(filename);
}

bool Scene::init() {
  // AreaLight and Shape are initialized
  // auto mat = std::make_shared<MicrofacetMaterial>(Vector3f(1.0), 0.007,
  //                                                 Vector3f(2.0, 2.0, 2.0));

  m_camera = std::make_shared<Camera>(m_origin, m_target, m_up);
  m_film   = std::make_shared<Film>(m_resX, m_resY, EFilmBufferType::EAll);
  m_accel  = std::make_shared<NaiveBVHAccel>(m_primitives);

  // m_volume = std::make_shared<OpenVDBVolume>(
  // "assets/wdas_cloud/wdas_cloud_eighth.vdb");
  // m_volume = std::make_shared<HVolume>();

  SLog("scene init finished");
  return true;
}

bool Scene::intersect(const Ray &ray, SInteraction &isect) const {
  return m_accel->intersect(ray, isect);
}

AABB Scene::getBound() const {
  AABB aabb;
  aabb = m_accel->getBound();
  if (m_volume != nullptr) aabb = aabb.merge(m_volume->getBound());
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
  assert(type == "path");
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

  std::shared_ptr<Material> mat;

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

      mat = std::make_shared<Transmission>(extIOR, intIOR);
      break;
    }  // "dielectric"
    case hash("diffuse"): {
      mat = std::make_shared<DiffuseMaterial>(Vector3f(1.0));
      break;
    }  // "diffuse"
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
              std::make_shared<EmbreeMeshPrimitive>(filename, mat));
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
      scene.m_primitives.push_back(std::make_shared<ShapePrimitive>(
          std::make_shared<Sphere>(center, radius), mat));
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
      scene.m_light.push_back(std::make_shared<InfiniteAreaLight>(env_texture));
      scene.m_infLight.push_back(scene.m_light[scene.m_light.size() - 1]);
    }
  }

  return true;
}

Scene Scene::parseXML(const std::string &filename) {
  Scene res;

  auto xml_path = std::filesystem::path(filename);
  assert(xml_path.has_parent_path());
  res.m_base_dir = xml_path.parent_path();

  pt::ptree tree;
  pt::read_xml(filename, tree);
  SLog("xml load from %s", filename.c_str());
  auto      scene_version = tree.get<std::string>("scene.<xmlattr>.version");
  pt::ptree scene         = tree.get_child("scene");
  pt::ptree integrator    = scene.get_child("integrator");
  pt::ptree sensor        = scene.get_child("sensor");
  setIntegrator(integrator, res);
  setSensor(sensor, res);

  // Parse shapes and emitters
  for (auto &v : scene) {
    switch (hash(v.first.c_str())) {
      case hash("shape"): {
        addShape(v.second, res);
        break;
      }  // "shape"
      case hash("emitter"): {
        addLight(v.second, res);
        break;
      }  // "light"
    }
  }

  return res;
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
