#ifndef __TEXTURE_HPP__
#define __TEXTURE_HPP__

#include <OpenImageIO/texture.h>

#include <memory>

#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"

FLG_NAMESPACE_BEGIN

class Texture {
public:
  virtual ~Texture() = default;

  virtual Vector3f eval(Float u, Float v) const;
  virtual Vector3f eval(const Vector2f &uv) const = 0;
};

class ImageTexture : public Texture {
public:
  ImageTexture(const std::string &path);  // init from path
  ~ImageTexture() override;

  Vector3f eval(const Vector2f &uv) const override;

private:
  std::string                         m_filename;
  OIIO::TextureSystem                *m_texture;
  OIIO::TextureSystem::TextureHandle *m_handle;
};

class ConstTexture : public Texture {
public:
  ConstTexture(const Vector3f &color);
  ~ConstTexture() override = default;

  Vector3f eval(const Vector2f &) const override;

private:
  Vector3f m_color;
};

FLG_NAMESPACE_END

#endif
