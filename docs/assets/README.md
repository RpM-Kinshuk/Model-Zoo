# README visual system

The README combines an animated system atlas, a compact SVG architecture,
two animated detail diagrams, a static lifecycle diagram and an expandable
Mermaid routing reference. Graphics describe the implementation;
motion illustrates flow rather than live progress. Informative SVGs have descriptive
titles and accessible descriptions. All assets are free of scripts and external dependencies.

| Asset | What it explains |
| --- | --- |
| [Page opening](page-opening.svg) / [page closing](page-closing.svg) | Transparent, unframed compositions: floating matrix entries dissolve into spectral ribbons and soft color washes |
| Navigation illustrations: [prepare](nav-prepare.svg), [architecture](nav-architecture.svg), [parallelism](nav-parallel.svg), [controls](nav-control.svg), [spectrum](nav-spectrum.svg), [verification](nav-verify.svg) | Small visual cues for the six chapter links, with all navigation labels kept in native text |
| [Run atlas](run-atlas.svg) / [animation](run-atlas.gif) | The manifest, dispatcher, isolated workers, spectra, supervision and confirmed resource release |
| [Architecture](architecture.svg) | Runner, dispatcher and worker ownership; live supervision and the confirmed-cleanup loop |
| [GPU orchestration](gpu-orchestration.svg) / [animation](gpu-orchestration.gif) | Two model processes, each with its own GPU pair and layer queue; pulses follow dispatch and return paths |
| [Worker lifecycle](worker-lifecycle.svg) | Monitoring and the order of confirmed termination, cache cleanup and GPU release |
| [Weight to spectrum](weight-to-spectrum.svg) / [animation](weight-to-spectrum.gif) | A concrete diagonal matrix, its singular values and its squared, sorted spectrum; matching outlines trace one component at a time |
| Chapter markers (`chapter-01.svg` through `chapter-06.svg`) | Gold, blue, green and lavender numerals beside native section titles |
| Signal cues: [reload](cue-reload.svg), [drain](cue-drain.svg), [stop](cue-stop.svg) | Illustrated controls; signal names remain selectable text |
| Resource cues: [weights](cue-weights.svg), [RAM](cue-ram.svg), [cache](cue-cache.svg) | Visual distinctions between the three configurable resource bounds |
| [Chapter thread](chapter-thread.svg) | A decorative, transparent transition between numbered chapters; no runtime meaning |
| [Compute ornament](chapter-compute.svg) / [spectrum ornament](chapter-spectrum.svg) | Small circuit and spectral motifs distinguish the implementation and measurement chapters |

## Page composition

The surrounding README uses a centered masthead, a six-part navigation grid,
numbered chapters, concise lead sentences and consistent expandable references.
Native HTML tables present navigation and resource bounds; Markdown keeps
commands copyable, source links usable and detailed reference text searchable.
The opening and closing ornaments flow into the page without replacing the
native title, navigation or explanatory atlas. There are no enclosing panels,
hard borders or central badges. Transparent canvases, fading ribbons and soft
radial washes let the host's background show through. Floating matrix entries
on the left answer spectral contours on the right; the footer quietly settles
beneath the final links. Midtone gold, blue, green and lavender strokes work on
both light and dark backgrounds.
An internal SVG media rule raises ribbon opacity in dark mode where supported;
the default artwork remains usable without it. This styling stays inside the
images and does not depend on custom README CSS.

Six small vector illustrations give the navigation cards distinct silhouettes.
They echo actual concepts without encoding extra instructions: pins, component
ownership, GPU lanes, runtime controls, spectra and checks. Keep their backgrounds
transparent, their display width modest (72 pixels), and their text labels live.

Chapter ornaments carry that artwork into the text: a gold thread for starting
and supervising work, circuit traces for architecture and parallelism, and
spectral curves for measurement and verification. All ornaments have empty alt
text in the README because they are decorative. They represent neither actual
measurements nor runtime activity. Essential labels remain native text or live
inside the explanatory diagrams.

Keep the main reading path short: an overview, two launch commands, the source
architecture, GPU scheduling, runtime control and the measurement itself.
Details summaries pair an action with a short description of what expands.
Retain the explicit named anchors when changing chapter titles so existing
section links continue to work.

Colored chapter markers carry the palette through the body without replacing
the native heading text. Setup steps reuse the navigation illustrations.
GitHub's native note/tip callouts distinguish preparation, launch defaults and
research scope. Signal and resource cards keep their labels, values and code
as native text; small illustrations provide a second visual cue. These cues
are not buttons or live controls.

Use GitHub-compatible HTML rather than page-level CSS, scripts or background
attributes. The host controls the page canvas and typography; atmosphere belongs
in self-contained artwork, with native content connecting the figures. Tables
and summaries should remain readable in both light and dark themes.

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

Two detail loops use the same tools and eight-second timing:

```bash
python docs/assets/render_detail_motion.py
```

The renderer imports the path-sampling helpers from `render_run_atlas.py`.
GPU pulse routes are invisible, named paths inside `gpu-orchestration.svg`.
Pulses illustrate thread dispatch and collection, not a benchmark, an exact
execution order or available capacity. Model C stays pending throughout.
Spectrum highlights follow named matrix-entry rectangles, singular-value bars
and eigenvalue points in `weight-to-spectrum.svg`. The correspondence applies
to this diagonal example; it is not a general entrywise decomposition rule.
Numbers and labels never change or disappear. Gentle outlines and small pulses
carry the motion; entire panels do not flash.

Every animated figure has a reduced-motion `picture` source and an adjacent
static-view link. A compact index below the navigation links to all three static
views. Preserve these alternatives even if the renderer or animation changes.
The new GIFs use fixed palettes and incremental frames; their combined size is
approximately 310 KB. Only final GIFs are retained after generation.

## Maintenance

Keep the architecture visible in the root README, with the source map and
detailed routing reference available on expansion. Its 1200 × 490 layout keeps
planning, resource ownership and execution side by side at the same scale as
the other figures. Preserve the supervision rail, ready/held distinction and
cleanup return path when updating behavior. The lower-level loading diagram
remains editable Mermaid inside its expandable reference.
Keep opening and closing artwork shallow (currently 164 and 134 pixels at
1200 pixels wide). Decorative SVGs are static, editable vector paths, with
no additional animation or regeneration dependencies. Preserve transparent
backgrounds in all page ornaments so they blend with light and dark page canvases.
Do not reintroduce full-canvas fills or rectangular frames around the header
and footer; opaque panels belong to explanatory diagrams that need label contrast.
Update these SVG labels if the corresponding behavior changes. Preview at both
native width and approximately 830 pixels, and check arrowheads, wrapping and
text contrast. Retain the README image descriptions when replacing an asset.
When simplifying the README, preserve detailed controls in expandable sections
or link to the existing operations guide rather than discarding them.
