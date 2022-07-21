#include "accel.hpp"

#include <cstddef>
#include <memory>

#include "aabb.hpp"
#include "debug.hpp"
#include "fwd.hpp"
#include "interaction.hpp"

SV_NAMESPACE_BEGIN

bool NaiveAccel::intersect(const Ray &ray, SInteraction &isect) const {
  // Intersect function will consider and modify ray.m_tMax
  // If there is any intersection between m_o and m_o + m_d*m_tMax,
  // the result will be true and the isect will be modified.
  // Otherwise, isect will not be touched
  Float        o_tMax = ray.m_tMax;
  Float        tMax   = o_tMax;
  SInteraction t_isect;
  for (auto &i : m_primitives) {
    if (i->intersect(ray, t_isect) && ray.m_tMax < tMax) {
      isect = t_isect;
      tMax  = ray.m_tMax;
    }
  }

  return tMax != o_tMax;
}

AABB NaiveAccel::getBound() const {
  AABB res;
  for (auto &primitive : m_primitives) res = res.merge(primitive->getBound());
  return res;
}

NaiveBVHAccel::NaiveBVHAccel(std::vector<std::shared_ptr<Primitive>> p,
                             int depth, AABB *box) {
  if (box == nullptr) {
    for (auto &primitive : p) m_box = m_box.merge(primitive->getBound());
  } else {
    m_box = *box;  // copy
  }

  // Termination
  if (p.size() < 50 || depth > 32) {
    m_primitives = p;
    m_left = m_right = nullptr;
    m_memory_usage =
        sizeof(m_box) + sizeof(m_left) * 2 +
        sizeof(std::shared_ptr<Primitive>) * m_primitives.capacity();
    m_depth = 1;
    return;
  }

  // dim will specify the current partition surface
  int dim = depth % 3;
  // allocate space for storing pointers
  std::vector<std::shared_ptr<Primitive>> left, right;

  // sort the primitives by position in the corresponding dim
  std::sort(p.begin(), p.end(), [dim](auto a, auto b) -> bool {
    return a->getBound().center()[dim] < b->getBound().center()[dim];
  });

  auto st  = p.begin();
  auto mid = p.begin() + p.size() / 2;
  auto ed  = p.end();
  left     = std::vector(st, mid);
  right    = std::vector(mid, ed);
  assert(p.size() == (left.size() + right.size()));

  m_left  = std::make_shared<NaiveBVHAccel>(NaiveBVHAccel(left, depth + 1));
  m_right = std::make_shared<NaiveBVHAccel>(NaiveBVHAccel(right, depth + 1));

  m_memory_usage = m_left->m_memory_usage + m_right->m_memory_usage;
  m_depth        = std::max(m_left->m_depth, m_right->m_depth) + 1;

  if (depth == 0) {
    // Print out the BVH info
    SLog("BVH num primitives: %lu", p.size());
    SLog("BVH memory usage: %lu", m_memory_usage);
    SLog("BVH depth: %lu", m_depth);
  }
}

AABB NaiveBVHAccel::getBound() const {
  return m_box;
}

bool NaiveBVHAccel::intersect(const Ray &ray, SInteraction &isect) const {
  // Under the constrain of ray.t_Max, if there's no intersection
  // return false
  Float t_min, t_max;
  bool  inter = m_box.intersect_pbrt(ray, t_min, t_max);
  if (!inter) return false;

  // If there is intersection
  // and if the node is leaf node
  if (!m_primitives.empty()) {
    Float        o_tMax = ray.m_tMax;
    Float        tMax   = o_tMax;
    SInteraction t_isect;
    for (auto &i : m_primitives) {
      if (i->intersect(ray, t_isect) && ray.m_tMax < tMax) {
        isect = t_isect;
        tMax  = ray.m_tMax;
      }
    }  // here, if tMax != o_tMax, there's intersection
    if (tMax != o_tMax) {
      // if there is, ray.m_tMax is already modified and isect is set
      return true;
    }  // else, no intersection, nothing will be modified
    return false;
  }  // if it is not leaf nodes

  bool res_left, res_right;
  res_left = res_right = false;
  if (m_left != nullptr) {
    res_left = m_left->intersect(ray, isect);
    // if there is intersection, the parameters are already fully modified
  }

  if (m_right != nullptr) {
    res_right = m_right->intersect(ray, isect);
  }

  return res_left || res_right;
}

SV_NAMESPACE_END
