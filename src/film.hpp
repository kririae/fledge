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

#include "common/vector.h"
#include "debug.hpp"

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
  Film(int resX, int resY, EFilmBufferType buffer_type = EFilmBufferType::EAll)
      : m_resX(resX), m_resY(resY), m_buffer_type(buffer_type) {
    checkFilmBufferType(buffer_type);
    m_buffers.resize(7);
    for (int t = 0; t < 7; ++t)
      if (int(buffer_type) & (1 << t))
        m_buffers[t] = std::vector<Vector3f>(m_resX * m_resY, Vector3f(0.0));
  }

  /**
   * @brief Init a new Film object with a existing film object. Data will
   * be copied.
   *
   * @param film The existing film object
   */
  Film(const Film &film) {
    std::lock_guard<std::mutex> lock(film.m_mutex);
    m_buffers = film.m_buffers;
    m_resX    = film.m_resX;
    m_resY    = film.m_resY;
  }

  /**
   * @brief Copy a new Film object with a existing film object. Data will
   * be copied.
   *
   * @param film The existing film object
   */
  Film &operator=(const Film &film) {
    std::lock_guard<std::mutex> lock(film.m_mutex);
    m_buffers = film.m_buffers;
    m_resX    = film.m_resX;
    m_resY    = film.m_resY;
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
    assert(buffer_id < m_buffers.size());
    std::lock_guard<std::mutex> lock(m_mutex);
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

  int m_resX, m_resY;

  EFilmBufferType                    m_buffer_type;
  std::vector<std::vector<Vector3f>> m_buffers;
  mutable std::mutex                 m_mutex;
};

FLG_NAMESPACE_END

#endif