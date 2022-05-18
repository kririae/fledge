#include "film.hpp"

#include <mutex>

SV_NAMESPACE_BEGIN

Film::Film(int resX, int resY) : m_resX(resX), m_resY(resY) {
  m_pixels.resize(m_resX * m_resY);
  m_pixels.assign(m_pixels.size(), Vector3f::Constant(0));
}

int Film::getPixelIdx(int x, int y) const {
  return y * m_resX + x;
}

Vector3f &Film::getPixel(int x, int y) {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_pixels[getPixelIdx(x, y)];
}

bool Film::saveImage(const std::string &name) {
  std::lock_guard<std::mutex> lock(m_mutex);
  std::filesystem::path       path(name);
  auto                        extension = path.extension().string();
  auto                        l_pixels  = m_pixels;

  // The transition from vector<Vector3f> to span<Float> must be valid
  assert(sizeof(Vector3f) == 12);
  auto pixels = std::span<Float>(reinterpret_cast<Float *>(l_pixels.data()),
                                 3 * l_pixels.size());

  SLog("path ends with %s", extension.c_str());
  if (extension == ".png" || extension == ".jpg") {
    SLog("gamma correction is applied");
    std::transform(std::execution::par, pixels.begin(), pixels.end(),
                   pixels.begin(),
                   [](float v) { return std::pow(v, 1.0f / 2.2f); });
  } else if (extension == ".exr") {
    // nothing is performed
  } else {
    SErr("file extension not supported");
  }

  // image output section
  SLog("writing image to %s", path.string().c_str());
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

SV_NAMESPACE_END
