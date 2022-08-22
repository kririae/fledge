# SmallVol

## Description

_Originally_ A small physically based render to implement volumetric algorithms.
Currently, a platform to implement seemingly non-trivial or non-conventional but (probably) useless render architectures and algorithms.

The ultimate and _not yet achieved goal_ for this render is to integrate as diverse techniques
within a consistent, fast, aggressive, yet elegant structure as possible ;)
In a word, stepping a step further from conventional PBRT practice!

Anyway, this is solely a personal project aiming to have fun and simultaneously practice my coding skills.

## Usage

`Unix Makefile` is recommended.`make help` to get more information.

## Code Structure

- `assets/`: Currently some scene settings for `mitsuba` render.
- `tests/`: Unit tests
- `src/`: ...TODO

## TODO

- [x] Integrate Embree as leaf nodes to accelerate intersection.
- [ ] Add vectorized code generated by `ispc` to optimize extremity calculations, e.g., `EstimateDirect()`, BRDF evaluations, Light Sampling.
- [ ] Replace OpenVDB with NanoVDB to implement other tracking methods.
- [ ] Support XML scene loading ;(
