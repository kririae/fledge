#include "texture.hpp"

#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/ustring.h>

#include <cstddef>
#include <memory>

#include "debug.hpp"
#include "fwd.hpp"

SV_NAMESPACE_BEGIN

Vector3f Texture::eval(Float u, Float v) const {
  return eval(Vector2f{u, v});
}

ImageTexture::~ImageTexture() {
  SV_Log("ImageTexture destroy");
  CPtr(m_texture);
  OIIO::TextureSystem::destroy(m_texture);
}

ImageTexture::ImageTexture(const std::string &path) : m_filename(path) {
  SV_Log("ImageTexture(%s)", path.c_str());
  m_texture = OIIO::TextureSystem::create(false);
  m_texture->attribute("max_memory_MB", 500.0f);
  m_handle = m_texture->get_texture_handle(OIIO::ustring(m_filename));
  if (m_handle == nullptr) {
    SV_Err("texture handle init failed");
  }
}

Vector3f ImageTexture::eval(const Vector2f &uv) const {
  float result[3];
  auto  opt = OIIO::TextureOpt();
  // bool  success = m_texture->texture(m_handle, nullptr, opt, uv.x(), uv.y(),
  // 0,
  //                                    0, 0, 0, 3, result);
  bool success = m_texture->texture(OIIO::ustring(m_filename), opt, uv.x(),
                                    uv.y(), 0, 0, 0, 0, 3, result);
  if (!success) SV_Err("texture access failed");
  return {result[0], result[1], result[2]};
}

ConstTexture::ConstTexture(const Vector3f &color) : m_color(color) {}
Vector3f ConstTexture::eval(const Vector2f &) const {
  return m_color;
}

SV_NAMESPACE_END
