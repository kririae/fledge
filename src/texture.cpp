#include "texture.hpp"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/texture.h>
#include <OpenImageIO/ustring.h>

#include <memory>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

Vector3f Texture::eval(Float u, Float v) const {
  return eval(Vector2f{u, v});
}

ImageTexture::~ImageTexture() {
  Log("ImageTexture destroy");
  OIIO::TextureSystem::destroy(m_texture);
}

ImageTexture::ImageTexture(const std::string &path) : m_filename(path) {
  Log("ImageTexture(%s)", path.c_str());
  m_texture = OIIO::TextureSystem::create(false);
}

Vector3f ImageTexture::eval(const Vector2f &uv) const {
  float result[3];
  auto  opt     = OIIO::TextureOpt();
  bool  success = m_texture->texture(OIIO::ustring(m_filename), opt, uv.x(),
                                     uv.y(), 0, 0, 0, 0, 3, result);
  assert(success);
  return {result[0], result[1], result[2]};
}

ConstTexture::ConstTexture(const Vector3f &color) : m_color(color) {}
Vector3f ConstTexture::eval(const Vector2f &) const {
  return m_color;
}

SV_NAMESPACE_END
