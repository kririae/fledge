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

#include "fwd.hpp"
#include "vector.hpp"

FLG_NAMESPACE_BEGIN

enum class EFilmBufferType {
  EColor  = (1),
  EAlbedo = (1 << 1),
  ENormal = (1 << 2),
  EOutput = (1 << 3),
  EAll    = (1 << 4) - 1,
  EExtra1 = (1 << 4),
  EExtra2 = (1 << 5),
  EExtra3 = (1 << 6),
};

// TODO: implement interface for `tev`
struct Film {
public:
  // Film class should be implemented comprehensively
  // Those copy manipulations are implemented for denoiser
  Film(int resX, int resY, EFilmBufferType buffer_type = EFilmBufferType::EAll)
      : m_resX(resX), m_resY(resY) {
    m_buffers.resize(7);
    for (int t = 0; t < 7; ++t)
      if (int(buffer_type) & (1 << t))
        m_buffers[t] = std::vector<Vector3f>(m_resX * m_resY, Vector3f(0.0));
  }
  Film(const Film &film) {
    std::lock_guard<std::mutex> lock(film.m_mutex);
    m_buffers = film.m_buffers;
    m_resX    = film.m_resX;
    m_resY    = film.m_resY;
  }
  Film &operator=(const Film &film) {
    std::lock_guard<std::mutex> lock(film.m_mutex);
    m_buffers = film.m_buffers;
    m_resX    = film.m_resX;
    m_resY    = film.m_resY;
    return *this;
  }
  ~Film() = default;

  static size_t bufferTypeToIdx(EFilmBufferType buffer_type) {
    Float index_ = std::log2(int(buffer_type));
    assert(abs(index_ - std::floor(index_)) < 1e-5);
    assert(int(index_) >= 0);
    return size_t(index_);
  }
  int       getBufferIdx(int x, int y) const { return y * m_resX + x; }
  Vector3f &getBuffer(int x, int y, EFilmBufferType buffer_type) {
    size_t buffer_id = bufferTypeToIdx(buffer_type);
    assert(buffer_id < m_buffers.size());
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_buffers[buffer_id][getBufferIdx(x, y)];
  }
  bool saveBuffer(const std::string &name, EFilmBufferType buffer_type);

  int m_resX, m_resY;

  std::vector<std::vector<Vector3f>> m_buffers;
  mutable std::mutex                 m_mutex;
};

FLG_NAMESPACE_END

#endif