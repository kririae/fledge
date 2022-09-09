#ifndef __FILM_HPP__
#define __FILM_HPP__

#include <OpenImageIO/imageio.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <execution>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "common/filter.h"
#include "common/vector.h"
#include "debug.hpp"
#include "fledge.h"
#include "resource.hpp"

FLG_NAMESPACE_BEGIN

static void checkFilmBufferType(EFilmBufferType buffer_type) {
  using enum EFilmBufferType;
  if ((int(buffer_type) & int(EColor)) == 0)
    SErr("In buffer_type, EColor must always be selected");
  if ((int(buffer_type) & int(EOutput)) == 0)
    SErr("In buffer_type, EOutput must always be selected");
}

// template<int FilmBufferType>
/**
 * @brief Film class will hold all the image buffer related to the output
 * @note This class has two types of interface. One is to directly manipulate
 * the underlying data. The second is designed to add sample with filters and
 * then commit the samples.
 */
struct Film {
public:
  /**
   * @brief Construct a new Film object with resolution and buffer_type settings
   *
   * @param resX The width of the image buffer
   * @param resY The height of the image buffer
   * @param buffer_type See EFilmBufferType
   */
  static constexpr int FILM_BUFFER_TYPE_NUM = 7;
  Film(int resX, int resY, Resource &resource,
       EFilmBufferType buffer_type = EFilmBufferType::EAll)
      : m_resX(resX),
        m_resY(resY),
        m_buffer_type(buffer_type),
        m_resource(&resource) {
    checkFilmBufferType(buffer_type);
    for (int t = 0; t < FILM_BUFFER_TYPE_NUM; ++t)
      if (int(buffer_type) & (1 << t))
        m_buffers[t] =
            m_resource->alloc<Vector3f[]>(m_resX * m_resY, Vector3f{0});
    // m_filter         = m_resource->alloc<GaussianFilter>(2, 2);
    m_filter         = m_resource->alloc<MitchellFilter>(2, 1 / 3, 1 / 3);
    m_sampled_pixels = m_resource->alloc<Pixel[]>(m_resX * m_resY);
  }

  /**
   * @brief Init a new Film object with a existing film object. Data will
   * be copied.
   *
   * @param film The existing film object
   */
  Film(const Film &film) { *this = film; }

  /**
   * @brief Copy a new Film object with a existing film object. Data will
   * be copied.
   *
   * @param film The existing film object
   */
  Film &operator=(const Film &film) {
    std::lock_guard<std::mutex> lock(film.m_mutex);
    m_resX = film.m_resX;
    m_resY = film.m_resY;
    for (int t = 0; t < FILM_BUFFER_TYPE_NUM; ++t) {
      if (!(int(film.m_buffer_type) & (1 << t))) continue;
      m_buffers[t] =
          film.m_resource->alloc<Vector3f[]>(m_resX * m_resY, Vector3f{0});
      std::copy(film.m_buffers[t], film.m_buffers[t] + m_resX * m_resY,
                m_buffers[t]);
    }  // for t
    return *this;
  }

  /**
   * @brief Destroy the Film object
   */
  ~Film() = default;

  static size_t bufferTypeToIdx(EFilmBufferType buffer_type) {
    Float index_ = std::log2(int(buffer_type));
    assert(abs(index_ - std::floor(index_)) < 1e-5);
    assert(int(index_) >= 0);
    return size_t(index_);
  }

  /**
   * @brief Get the pixel index in the buffer with the two indicies.
   *
   * @param x Index in width
   * @param y Index in height
   * @return int The pixel index in the buffer
   */
  int getBufferIdx(int x, int y) const { return y * m_resX + x; }

  /**
   * @brief Get the Buffer with indicies and buffer_type
   *
   * @param x
   * @param y
   * @param buffer_type
   * @return Vector3f&
   */
  Vector3f &getBuffer(int x, int y, EFilmBufferType buffer_type) {
    size_t buffer_id = bufferTypeToIdx(buffer_type);
    assert(buffer_id < FILM_BUFFER_TYPE_NUM);
    return m_buffers[buffer_id][getBufferIdx(x, y)];
  }

  /**
   * @brief Save the whole buffer with the buffer_type to a file
   *
   * @param name The file's name to save to
   * @param buffer_type
   * @return true The image file is successfully created.
   */
  bool saveBuffer(const std::string &name, EFilmBufferType buffer_type);

  /* Sample Interface */
  struct Pixel {
    Spectrum L{0};  // sum filter.eval() * L
    Float    I{0};  // sum filter.eval()
  };

  /**
   * @brief Add a sample on to the film plane
   *
   * @param p The sample's position
   * @param L The sample's value
   */
  void addSample(const Vector2f &p, const Spectrum &L) {
    // consider all the pixels that can cover this sample
    // since pixels are defined as the center of the rectangle
    // offset the pixel by 0.5
    auto     p_          = p - Vector2f(0.5);
    Vector2d lower_bound = Ceil(p_ - m_filter->m_radius).cast<int, 2>();
    Vector2d upper_bound =
        Floor(p_ + m_filter->m_radius).cast<int, 2>() + Vector2d(1);
    lower_bound = Max(lower_bound, Vector2d(0));
    upper_bound = Min(upper_bound, Vector2d(m_resX, m_resY));
    for (int x = lower_bound.x(); x < upper_bound.x(); ++x) {
      for (int y = lower_bound.y(); y < upper_bound.y(); ++y) {
        Pixel &pixel = m_sampled_pixels[getBufferIdx(x, y)];
        auto   I     = m_filter->eval(p_ - Vector2d(x, y).cast<Float, 2>());
        pixel.I += I;
        pixel.L += I * L;
      }
    }
  }

  void commitSamples() {
    std::lock_guard<std::mutex> lock(m_mutex);
    SLog("samples are committed");
    for (int x = 0; x < m_resX; ++x) {
      for (int y = 0; y < m_resY; ++y) {
        const Pixel &pixel = m_sampled_pixels[getBufferIdx(x, y)];
        if (pixel.I == 0) continue;
        m_buffers[bufferTypeToIdx(EFilmBufferType::EColor)]
                 [getBufferIdx(x, y)] = pixel.L / pixel.I;
      }
    }
  }

  int m_resX, m_resY;

  EFilmBufferType    m_buffer_type;
  Resource          *m_resource;
  Vector3f          *m_buffers[FILM_BUFFER_TYPE_NUM];
  mutable std::mutex m_mutex;

  /**
   *  Variables with Sampler Interface
   */
  Filter *m_filter;
  Pixel  *m_sampled_pixels;
};

FLG_NAMESPACE_END

#endif