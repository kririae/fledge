#include "plymesh.hpp"

#include <cstdlib>
#include <memory>

#include "debug.hpp"
#include "fledge.h"
#include "debug.hpp"
#include "rng.hpp"
#include "shape.hpp"
#include "common/math_utils.h"
#include "common/vector.h"

FLG_NAMESPACE_BEGIN

// Code modified from https://github.com/vilya/miniply
// anyway, it works
std::shared_ptr<TriangleMesh> MakeTriangleMesh(const std::string &path) {
  if (!std::filesystem::exists(path))
    SErr("plymesh file %s do not exists", path.c_str());

  miniply::PLYReader reader(path.c_str());
  if (!reader.valid()) SErr("miniply reader failed to initialize");

  uint32_t             face_idxs[3];
  miniply::PLYElement *face_elem =
      reader.get_element(reader.find_element(miniply::kPLYFaceElement));
  if (face_elem == nullptr) SErr("face element failed to initialize");
  face_elem->convert_list_to_fixed_size(
      face_elem->find_property("vertex_indices"), 3, face_idxs);

  uint32_t indexes[3];
  bool     got_verts = false, got_faces = false;

  SLog("ply mesh loading");
  auto mesh = std::make_shared<TriangleMesh>();
  while (reader.has_element() && (!got_verts || !got_faces)) {
    if (reader.element_is(miniply::kPLYVertexElement) &&
        reader.load_element() && reader.find_pos(indexes)) {
      mesh->nVert = reader.num_rows();

      // extract position
      mesh->p = std::unique_ptr<Vector3f[]>(
          (Vector3f *)std::aligned_alloc(16, sizeof(Vector3f) * mesh->nVert));
      // mesh->p = std::make_unique<Vector3f[]>(mesh->nVert);
      reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float,
                                mesh->p.get());

      // extract normal
      if (reader.find_normal(indexes)) {
        mesh->n = std::unique_ptr<Vector3f[]>(
            (Vector3f *)std::aligned_alloc(16, sizeof(Vector3f) * mesh->nVert));
        // mesh->n = std::make_unique<Vector3f[]>(mesh->nVert);
        reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float,
                                  mesh->n.get());
        SLog("normal found in ply file");
      } else {
        SLog("normal not found in ply file");
      }

      // extract UV
      if (reader.find_texcoord(indexes)) {
        mesh->uv = std::unique_ptr<Vector2f[]>(
            (Vector2f *)std::aligned_alloc(16, sizeof(Vector2f) * mesh->nVert));
        // mesh->uv = std::make_unique<Vector2f[]>(mesh->nVert);
        reader.extract_properties(indexes, 2, miniply::PLYPropertyType::Float,
                                  mesh->uv.get());
      }

      got_verts = true;
    } else if (!got_faces && reader.element_is(miniply::kPLYFaceElement) &&
               reader.load_element()) {
      mesh->nInd = reader.num_rows() * 3;
      mesh->ind  = std::unique_ptr<int[]>(
          (int *)std::aligned_alloc(16, sizeof(int) * mesh->nInd));
      // mesh->ind = std::make_unique<int[]>(mesh->nInd);
      reader.extract_properties(face_idxs, 3, miniply::PLYPropertyType::Int,
                                mesh->ind.get());
      got_faces = true;
    }

    if (got_verts && got_faces) break;
    reader.next_element();
  }

  if (!got_verts || !got_faces) SErr("failed to create mesh");

  SLog("ply mesh loaded");
  return mesh;
}

std::shared_ptr<TriangleMesh> MakeMeshedSphere() {
  // TODO: complete parameters
  Random rng;
  int    n_theta = 16, n_phi = 16;

  int  n_vert = n_theta * n_phi;
  auto mesh   = std::make_shared<TriangleMesh>();
  mesh->nVert = n_vert;
  std::vector<Vector3f> vertices;
  for (int t = 0; t < n_theta; ++t) {
    Float theta     = PI * (Float)t / (Float)(n_theta - 1);
    Float cos_theta = std::cos(theta);
    Float sin_theta = std::sin(theta);
    for (int p = 0; p < n_phi; ++p) {
      Float phi    = 2 * PI * (Float)p / (Float)(n_phi - 1);
      Float radius = 1;
      // Make sure all of the top and bottom vertices are coincident.
      if (t == 0)
        vertices.push_back(Vector3f(0, 0, radius));
      else if (t == n_theta - 1)
        vertices.push_back(Vector3f(0, 0, -radius));
      else if (p == n_phi - 1)
        // Close it up exactly at the end
        vertices.push_back(vertices[vertices.size() - (n_phi - 1)]);
      else {
        vertices.push_back(Vector3f(0, 0, 0) +
                           radius *
                               SphericalDirection(sin_theta, cos_theta, phi));
      }
    }
  }

  std::vector<int> indices;
  // fan at top
  auto offset = [n_phi](int t, int p) { return t * n_phi + p; };
  for (int p = 0; p < n_phi - 1; ++p) {
    indices.push_back(offset(0, 0));
    indices.push_back(offset(1, p));
    indices.push_back(offset(1, p + 1));
  }

  // quads in the middle rows
  for (int t = 1; t < n_theta - 2; ++t) {
    for (int p = 0; p < n_phi - 1; ++p) {
      indices.push_back(offset(t, p));
      indices.push_back(offset(t + 1, p));
      indices.push_back(offset(t + 1, p + 1));

      indices.push_back(offset(t, p));
      indices.push_back(offset(t + 1, p + 1));
      indices.push_back(offset(t, p + 1));
    }
  }

  // fan at bottom
  for (int p = 0; p < n_phi - 1; ++p) {
    indices.push_back(offset(n_theta - 1, 0));
    indices.push_back(offset(n_theta - 2, p));
    indices.push_back(offset(n_theta - 2, p + 1));
  }

  mesh->nInd = indices.size();
  mesh->p    = std::make_unique<Vector3f[]>(mesh->nVert);
  mesh->ind  = std::make_unique<int[]>(mesh->nInd);
  for (int i = 0; i < mesh->nVert; ++i) mesh->p[i] = vertices[i];
  for (int i = 0; i < mesh->nInd; ++i) mesh->ind[i] = indices[i];

  return mesh;
}

FLG_NAMESPACE_END
