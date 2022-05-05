#ifndef __FILM_HPP__
#define __FILM_HPP__

#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <cassert>
#include <execution>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "fwd.hpp"

SV_NAMESPACE_BEGIN

// TODO: implement interface for `tev`

class Film {
public:
  Film(int resX, int resY) : m_resX(resX), m_resY(resY) {
    m_pixels.resize(m_resX * m_resY);
    m_pixels.assign(m_pixels.size(), Vector3f::Constant(0));
  }

  ~Film() = default;
  int       getPixelIdx(int x, int y) const { return y * m_resX + x; }
  Vector3f &getPixel(int x, int y) { return m_pixels[getPixelIdx(x, y)]; }
  bool      saveImage(const std::string &name) {
    std::filesystem::path path(name);
    auto                  extension = path.extension().string();

    // The transition from vector<Vector3f> to span<Float> must be valid
    assert(sizeof(Vector3f) == 12);
    auto pixels = std::span<Float>(reinterpret_cast<Float *>(m_pixels.data()),
                                   3 * m_pixels.size());

    Log("path ends with %s", extension.c_str());
    if (extension == ".png" || extension == ".jpg") {
      Log("gamma correction is applied");
      std::transform(std::execution::par_unseq, pixels.begin(), pixels.end(),
                          pixels.begin(),
                          [](float v) { return std::pow(v, 1.0f / 2.2f); });
    } else if (extension == ".exr") {
      // nothing is performed
    } else {
      Err("file extension not supported");
    }

    // image output section
    Log("writing image to %s", path.string().c_str());
    std::unique_ptr<OIIO::ImageOutput> out =
        OIIO::ImageOutput::create(path.string());
    const int       scanline_size = m_resX * 3 * sizeof(Float);
    OIIO::ImageSpec spec(m_resX, m_resY, 3, OIIO::TypeDesc::FLOAT);
    out->open(path.string(), spec);
    out->write_image(OIIO::TypeFloat, pixels.data() + (m_resY - 1) * m_resX * 3,
                          OIIO::AutoStride, -scanline_size, OIIO::AutoStride);
    out->close();

    return true;
  }

private:
  int                   m_resX, m_resY;
  std::vector<Vector3f> m_pixels;
};

SV_NAMESPACE_END

#endif