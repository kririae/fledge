# Fledge Renderer

![](https://file.yuyuko.cc/images/fledge_sample_0.jpg)
> A simple scene with some artifacts. This geometry is from [Ariel](https://github.com/betajippity/Ariel) simulator.
## Description

_Originally_ A small physically based render to implement volumetric algorithms.
Currently, a platform to implement seemingly non-trivial or non-conventional but (probably) useless render architectures and algorithms.

The ultimate and _not yet achieved goal_ for this render is to integrate as diverse techniques
within a consistent, fast, aggressive, yet elegant structure as possible ;)
In a word, going a step further from conventional PBRT practice!

Anyway, this is solely a personal project aiming to have fun and simultaneously practice my coding skills.

## Usage

`Unix Makefile` is recommended. Use `make help` to get more information.

## Code Structure

- `src/`:
  - `spec/`: Those libraries that could optimize some portions of the renderer.
  - `common`: Some header files that could be used in both CPU and GPU backends.
  - `materials`: Some material implementations that have the common interface.
- `assets/`: Some scene settings for both `mitsuba` and this renderer.
- `tests/`: Unit tests

## TODO

- [x] Integrate Embree as leaf nodes to accelerate intersection.
- [x] Support XML scene loading
- [x] Integrate OpenImageDenoise
- [ ] Add low-discrepancy sampler
- [ ] Add vectorized code generated by `ispc` to optimize extremity calculations, e.g., `EstimateDirect()`, BRDF evaluations, Light Sampling.
- [ ] Add naive OptiX backend
- [ ] Replace OpenVDB with NanoVDB to implement other tracking methods.
