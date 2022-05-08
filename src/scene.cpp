#include "scene.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ptree_fwd.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <string>

SV_NAMESPACE_BEGIN
namespace pt = boost::property_tree;

static bool parseIntegrator(const pt::ptree &integrator, Scene &scene) {
  auto type = integrator.get<std::string>("<xmlattr>.type");
  Log("integrator.type = %s", type.c_str());
  int maxDepth = integrator.get<int>("integer.<xmlattr>.value");
  Log("integrator.maxDepth = %d", maxDepth);
  scene.m_maxDepth = maxDepth;
  return true;
}

bool Scene::loadFromXml(const std::string &filename) {
  pt::ptree tree;
  pt::read_xml(filename, tree);
  Log("xml load from %s", filename.c_str());
  auto      scene_version = tree.get<std::string>("scene.<xmlattr>.version");
  pt::ptree scene         = tree.get_child("scene");
  pt::ptree integrator    = scene.get_child("integrator");
  parseIntegrator(integrator, *this);
  // std::cout << scene_version << std::endl;
  TODO();

  return true;
}

SV_NAMESPACE_END
