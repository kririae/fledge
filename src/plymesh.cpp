#include "plymesh.hpp"

#include <memory>

#include "debug.hpp"
#include "fwd.hpp"
#include "shape.hpp"

SV_NAMESPACE_BEGIN

// Code modified from https://github.com/vilya/miniply
// anyway, it works
std::shared_ptr<TriangleMesh> make_TriangleMesh(const std::string &path) {
  if (!std::filesystem::exists(path))
    SErr("plymesh file %s do not exists", path.c_str());

  miniply::PLYReader reader(path.c_str());
  if (!reader.valid()) SErr("miniply reader failed to initialize");

  uint32_t             faceIdxs[3];
  miniply::PLYElement *faceElem =
      reader.get_element(reader.find_element(miniply::kPLYFaceElement));
  if (faceElem == nullptr) SErr("face element failed to initialize");
  faceElem->convert_list_to_fixed_size(
      faceElem->find_property("vertex_indices"), 3, faceIdxs);

  uint32_t indexes[3];
  bool     gotVerts = false, gotFaces = false;

  SLog("ply mesh loading");
  auto mesh = std::make_shared<TriangleMesh>();
  while (reader.has_element() && (!gotVerts || !gotFaces)) {
    if (reader.element_is(miniply::kPLYVertexElement) &&
        reader.load_element() && reader.find_pos(indexes)) {
      mesh->nVert = reader.num_rows();

      // extract position
      mesh->p = std::make_unique<Vector3f[]>(mesh->nVert);
      reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float,
                                mesh->p.get());

      // extract normal
      if (reader.find_normal(indexes)) {
        mesh->n = std::make_unique<Vector3f[]>(mesh->nVert);
        reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float,
                                  mesh->n.get());
        SLog("normal found in ply file");
      }

      // extract UV
      if (reader.find_texcoord(indexes)) {
        mesh->uv = std::make_unique<Vector2f[]>(mesh->nVert);
        reader.extract_properties(indexes, 2, miniply::PLYPropertyType::Float,
                                  mesh->uv.get());
      }

      gotVerts = true;
    } else if (!gotFaces && reader.element_is(miniply::kPLYFaceElement) &&
               reader.load_element()) {
      mesh->nInd = reader.num_rows() * 3;
      mesh->ind  = std::make_unique<int[]>(mesh->nInd);
      reader.extract_properties(faceIdxs, 3, miniply::PLYPropertyType::Int,
                                mesh->ind.get());
      gotFaces = true;
    }

    if (gotVerts && gotFaces) break;
    reader.next_element();
  }

  if (!gotVerts || !gotFaces) SErr("failed to create mesh");

  SLog("ply mesh loaded");
  return mesh;
}

SV_NAMESPACE_END
