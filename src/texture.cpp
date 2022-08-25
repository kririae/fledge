#include "texture.hpp"

#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/ustring.h>

#include <cstddef>
#include <filesystem>
#include <memory>

#include "debug.hpp"
#include "fwd.hpp"
#include "vector.hpp"

FLG_NAMESPACE_BEGIN

Vector3f Texture::eval(Float u, Float v) const {
  return eval(Vector2f{u, v});
}

ImageTexture::~ImageTexture() {
  SLog("ImageTexture destroy");
  C(m_texture);
  OIIO::TextureSystem::destroy(m_texture);
}

ImageTexture::ImageTexture(const std::string &path) : m_filename(path) {
  SLog("ImageTexture(%s)", path.c_str());
  if (!std::filesystem::exists(path))
    SErr("texture file %s do not exists", path.c_str());
  m_texture = OIIO::TextureSystem::create(false);
  m_texture->attribute("max_memory_MB", 500.0f);
  m_handle = m_texture->get_texture_handle(OIIO::ustring(m_filename));
  if (m_handle == nullptr) {
    SErr("texture handle init failed");
  }
}

Vector3f ImageTexture::eval(const Vector2f &uv) const {
  float result[3];
  auto  opt     = OIIO::TextureOpt();
  bool  success = m_texture->texture(m_handle, nullptr, opt, uv.x(), uv.y(), 0,
                                     0, 0, 0, 3, result);
  // bool  success = m_texture->texture(OIIO::ustring(m_filename), opt, uv.x(),
  //                                    uv.y(), 0, 0, 0, 0, 3, result);
  C(result[0]);
  C(result[1]);
  C(result[2]);
  if (!success) SErr("texture access failed");
  return {result[0], result[1], result[2]};
}

ConstTexture::ConstTexture(const Vector3f &color) : m_color(color) {}
Vector3f ConstTexture::eval(const Vector2f &) const {
  return m_color;
}

FLG_NAMESPACE_END
