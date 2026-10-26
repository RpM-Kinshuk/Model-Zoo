# README visual system

The README combines an animated system atlas, a visible Mermaid architecture,
three static explanatory diagrams and an expandable routing reference. Graphics describe the implementation;
motion illustrates flow rather than live progress. All SVGs have descriptive
titles and accessible descriptions, with no scripts or external dependencies.

| Asset | What it explains |
| --- | --- |
| [Run atlas](run-atlas.svg) / [animation](run-atlas.gif) | The manifest, dispatcher, isolated workers, spectra, supervision and confirmed resource release |
| [GPU orchestration](gpu-orchestration.svg) | Two model processes, each with its own GPU pair and layer queue; physical-to-local CUDA mapping |
| [Worker lifecycle](worker-lifecycle.svg) | Monitoring and the order of confirmed termination, cache cleanup and GPU release |
| [Weight to spectrum](weight-to-spectrum.svg) | A concrete diagonal matrix, its singular values and its squared, sorted spectrum |

## Visual language

| Color | Role |
| --- | --- |
| `#003262` | Headings and primary text |
| `#003B95`, `#006CE4` | Execution, allocation and the first worker lane |
| `#00693E` | The second worker lane and confirmed cleanup |
| `#FDB515`, `#FFF4D6` | Controls, monitoring, pending work and small accents |
| `#061B36`, `#063E34` | Midnight-blue and evergreen atlas background |
| `#FFD16A`, `#FFF3C9` | Gold supervisory controls and pinned-manifest panels |
| `#9272BE`, `#69B49A` | Additional matrix-component colors, retained across weights, singular values and eigenvalues |
| `#F5F3EC`, `#FFFFFF` | Warm paper and explicit panel backgrounds for consistent light/dark rendering |
| `#52647A` | Secondary labels |

Serif titles, spectral contours, GPU-grid textures and small circuit motifs
give the figures an editorial character. The atlas uses a dark atmospheric
field with restrained blue, green and gold light; the remaining figures use
warm pale surfaces. Keep backgrounds low-contrast and behind opaque label panels.
Sans-serif labels and monospace identifiers distinguish
explanations from executable names. Use color together with labels and shapes;
meaning must survive grayscale or a stopped animation.

The atlas and GPU diagrams show illustrative concurrent runs; the launch default
remains one model worker. The GPU diagram has two GPUs per worker,
not the default launch configuration. Its layer threads use both assigned GPUs;
the estimator's actual compute pool can depend on model placement. The spectrum
diagram uses `diag(4, 3, 2, 1)` and eigenvalues `[1, 4, 9, 16]` for explanation,
not as evidence of a fitted power law or model quality.

## Motion and regeneration

The atlas animation is a 7.2-second loop with 72 frames. A `picture` source selects
the static SVG for `prefers-reduced-motion: reduce` where supported; an explicit
static-view link remains available. The diagram carries its complete meaning
in every frame. Pulses follow paths defined in the SVG; they do not encode
actual job progress or numerical results.

Regenerate the GIF after editing its SVG, using Inkscape and Pillow:

```bash
python docs/assets/render_run_atlas.py
```

The script reads route geometry from the SVG, renders in a temporary directory
and keeps only `run-atlas.gif`. A fixed palette and incremental frames keep
the animation small. These are documentation tools, not worker dependencies.

## Maintenance

Keep the architecture visible in the root README, with the source map and
detailed routing reference available on expansion. Preserve the balanced
Mermaid layout when updating runtime behavior.
Update these SVG labels if the corresponding behavior changes. Preview at both
native width and approximately 830 pixels, and check arrowheads, wrapping and
text contrast. Retain the README image descriptions when replacing an asset.
When simplifying the README, preserve detailed controls in expandable sections
or link to the existing operations guide rather than discarding them.
