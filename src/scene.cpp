#include "scene.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ptree_fwd.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <memory>
#include <string>

#include "aabb.hpp"
#include "accel.hpp"
#include "camera.hpp"
#include "debug.hpp"
#include "film.hpp"
#include "light.hpp"
#include "texture.hpp"
#include "volume.hpp"

SV_NAMESPACE_BEGIN
namespace pt = boost::property_tree;

Scene::Scene() {
  auto sphere_1 = std::make_shared<Sphere>(Vector3f{0.0, 0.0, 0.0}, 250);
  auto env_texture =
      std::make_shared<ImageTexture>("assets/venice_sunset_4k.exr");
  auto infAreaLight = std::make_shared<InfiniteAreaLight>(env_texture);
  auto diffuse = std::make_shared<DiffuseMaterial>(Vector3f::Constant(1.0));
  m_resX       = 1280;
  m_resY       = 720;
  m_SPP        = 4;
  m_camera = std::make_shared<Camera>(Vector3f(0, 0, 800), Vector3f(0, 100, 0));
  m_film   = std::make_shared<Film>(m_resX, m_resY);
  m_accel =
      std::make_shared<NaiveAccel>(std::vector<std::shared_ptr<Primitive>>{
          std::make_shared<ShapePrimitive>(sphere_1, diffuse)});
  m_light    = std::vector<std::shared_ptr<Light>>{infAreaLight};
  m_infLight = std::vector<std::shared_ptr<Light>>{infAreaLight};

  // volume
  m_volume = std::make_shared<OpenVDBVolume>(
      "assets/wdas_cloud/wdas_cloud_eighth.vdb");
  // m_volume = std::make_shared<HVolume>();
}

Scene::Scene(const std::string &filename) {
  bool success = loadFromXml(filename);
  assert(success);
}

bool Scene::intersect(const Ray &ray, SInteraction &isect) const {
  return m_accel->intersect(ray, isect);
}

AABB Scene::getBound() const {
  AABB aabb;
  // aabb = m_accel->getBound();
  if (m_volume != nullptr) aabb = aabb.merge(m_volume->getBound());
  return aabb;
}

static bool parseIntegrator(const pt::ptree &integrator, Scene &scene) {
  /*
  auto type = integrator.get<std::string>("<xmlattr>.type");
  SV_Log("integrator.type = %s", type.c_str());
  int maxDepth = integrator.get<int>("integer.<xmlattr>.value");
  SV_Log("integrator.maxDepth = %d", maxDepth);
  scene.m_maxDepth = maxDepth;
  */
  return true;
}

bool Scene::loadFromXml(const std::string &filename) {
  /*
  pt::ptree tree;
  pt::read_xml(filename, tree);
  SV_Log("xml load from %s", filename.c_str());
  auto      scene_version = tree.get<std::string>("scene.<xmlattr>.version");
  pt::ptree scene         = tree.get_child("scene");
  pt::ptree integrator    = scene.get_child("integrator");
  parseIntegrator(integrator, *this);
  // std::cout << scene_version << std::endl;
  */
  TODO();

  return true;
}

SV_NAMESPACE_END
