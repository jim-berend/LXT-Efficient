# lxt-efficient: LXT reduced to the efficient core

**This is a fork. The original project is [LRP-eXplains-Transformers (LXT)](https://github.com/rachtibat/LRP-eXplains-Transformers)**
by Fraunhofer HHI. [PyPI: `lxt`](https://pypi.org/project/lxt/) ·
[docs](https://lxt.readthedocs.io) · [AttnLRP, ICML 2024](https://proceedings.mlr.press/v235/achtibat24a.html).
**If you are looking for LXT itself, meaning the model zoo, the explicit implementation and the
tutorials, use upstream, not this.** All credit for the method and the code belongs there. This fork
adds no functionality.

What it is: LXT cut down to the parts of `lxt.efficient` that do not import `transformers`, namely
the LRP rules, the autograd patches, and the zennit interop. 541 lines across 6 modules instead of
13,814 across 30, one required runtime dependency instead of eight, zero `transformers` imports.

## Why this fork exists

Upstream LXT couples the whole package to a specific `transformers` version, and the coupling is
load-bearing for code that only wants the rules:

- **A single unrelated model module can take down the entire package.**
  `lxt/efficient/models/bert.py` imports `find_pruneable_heads_and_indices` and
  `prune_linear_layer`, which `transformers 5` removed. Because `models/__init__.py` eagerly imports
  every model module and `efficient/__init__.py` imports `core`, which imports `models`, that one
  `ImportError` makes `import lxt.efficient` fail outright, even for consumers that never touch
  BERT. There is no failure isolation: an unsupported architecture is indistinguishable from a
  broken install.
- **The coupling is structural, not accidental.** `bert.py` is a 2033-line modified copy of
  `transformers`' `modeling_bert`, 43% of LXT's source. Carrying a snapshot of another library's
  internals means breakage on that library's releases is expected, not bad luck.
- **Eight runtime dependencies** (including `bitsandbytes` and `open_clip_torch`) for consumers that
  use LXT as a rule library and bring their own patch maps.
- **Upstream has had no commits in roughly a year** and no `transformers 5`-compatible release, so
  waiting is not a strategy.

Deleting the model zoo and the explicit implementation removes the coupling by construction rather
than shimming around it: the remaining modules cannot break on a future `transformers` release
because they never import it. The trade is that you must supply your own patch maps, which is
already the case for anyone using LXT as a rule library.

A fix for the failure-isolation defect (per-model `try/except ImportError`, or moving the imports
inside `get_default_map`) belongs upstream and is worth contributing separately. It is orthogonal
to this reduction.

Forked from upstream commit `0b91031` (LXT 2.1). The import package is still named `lxt`, so
downstream code needs no changes:

```python
from lxt.efficient import monkey_patch, monkey_patch_zennit
from lxt.efficient import patches
from lxt.efficient.patches import divide_gradient, identity_rule_implicit
import lxt.efficient.zennit_patches as zp
```

## Installation

```bash
pip install "lxt-efficient[zennit] @ git+https://github.com/jim-berend/LXT-Efficient@v2.1.0-efficient"
```

`lxt-efficient` and upstream `lxt` both provide the `lxt` import package and cannot be installed
side by side, so remove `lxt` first.

## Dependencies

| | |
| --- | --- |
| Required | `torch` |
| Extra `zennit` | `zennit`, needed for `monkey_patch_zennit`. The import is guarded, so the package remains importable without it (with a warning) |
| Python | `>=3.10` |

Upstream declares `torch`, `transformers`, `accelerate`, `tabulate`, `matplotlib`, `bitsandbytes`,
`open_clip_torch` and `zennit`. All but `torch` and `zennit` were needed only by the removed
subsystems.

## What this fork contains

```
lxt/efficient/patches.py         # autograd patch functions
lxt/efficient/rules.py           # LRP rules as torch.autograd.Function
lxt/efficient/zennit_patches.py  # zennit interop (the zennit import is guarded)
lxt/efficient/core.py            # monkey_patch
```

## What was removed, and why

| Removed | Reason |
| --- | --- |
| `lxt/efficient/models/` | The bundled per-model patch maps. Every `transformers` import in `lxt.efficient` lives here, including `models/bert.py`, whose import of `find_pruneable_heads_and_indices` / `prune_linear_layer` fails on `transformers 5` and, through the eager `models/__init__.py`, takes down all of `lxt.efficient` with it. |
| `lxt/explicit/` | The mathematically explicit LRP implementation. Depends on `transformers.utils.fx`, removed in `transformers 5`. |
| `lxt/utils.py` | matplotlib heatmap helpers, outside the core. |
| `docs/`, `examples/`, `tests/`, `.readthedocs.yaml` | Cover the removed subsystems. |
| `setup.py` | Replaced by `pyproject.toml`. |

Consumers that want LXT's ready-made Llama/Qwen/Gemma3 maps rather than their own can re-add the
relevant module from `lxt/efficient/models/`. Each is 20 to 32 lines and imports only its own
architecture.

## Modifications relative to upstream

Only two source files differ from upstream. `patches.py`, `rules.py` and `zennit_patches.py` are
byte-identical copies.

- `lxt/efficient/core.py`: dropped `from lxt.efficient.models import get_default_map`.
  `monkey_patch(module, patch_map, verbose=False)` now takes `patch_map` as a required argument and
  raises `ValueError` instead of falling back to a bundled default map.
- `lxt/efficient/__init__.py`: added `__all__`.

Packaging: distribution name `lxt-efficient` (upstream: `lxt`), version `2.1.0`, built with
hatchling.

## Staying in sync with upstream

```bash
git fetch upstream && git rebase upstream/main
```

Conflicts are limited to `core.py`, `efficient/__init__.py` and the packaging files.

## License and attribution

LXT is copyright 2024 Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. and the
authors Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain, Thomas
Wiegand, Sebastian Lapuschkin, Wojciech Samek, released under the Clear BSD License. The `LICENSE`
file and all per-file copyright headers are retained unmodified. This is a modified derivative
work, and the modifications are listed above. The names of the original authors are not used to
endorse or promote it. Note that the Clear BSD variant withholds patent rights.

## Citation

Please cite the original work:

```
@InProceedings{pmlr-v235-achtibat24a,
  title = {{A}ttn{LRP}: Attention-Aware Layer-Wise Relevance Propagation for Transformers},
  author = {Achtibat, Reduan and Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and Jain, Aakriti and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {135--168},
  year = {2024},
  editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = {235},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR}
}
```

## Acknowledgements

LXT's code is heavily inspired by [Zennit](https://github.com/chr5tphr/zennit), a tool for LRP
attributions in PyTorch using hooks.
