#ifndef __FILM_HPP__
#define __FILM_HPP__

#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <cassert>
#include <execution>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>

#include "fwd.hpp"
#include "vector.hpp"

FLG_NAMESPACE_BEGIN

// TODO: implement interface for `tev`
class Film {
public:
  Film(int resX, int resY);
  ~Film() = default;
  int       getPixelIdx(int x, int y) const;
  Vector3f &getPixel(int x, int y);
  bool      saveImage(const std::string &name);

private:
  int                   m_resX, m_resY;
  std::vector<Vector3f> m_pixels;
  std::mutex            m_mutex;
};

FLG_NAMESPACE_END

#endif