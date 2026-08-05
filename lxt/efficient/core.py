# Copyright 2024, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. &
# the authors: Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain,
# Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek. All rights reserved.
# 
# This code is based on the following work:
# 
#   'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers. ICML 2024.'
#
# The copyright in this software is being made available under the Clear BSD License.
# No patent rights, trademark rights and/or other Intellectual Property Rights other than
# the copyrights concerning the Software are granted under this license.
# You may obtain a full copy of the License at
#     
#   https://github.com/rachtibat/LRP-eXplains-Transformers/blob/main/LICENSE
#
from warnings import warn


def monkey_patch(module, patch_map, verbose=False):
    """
    This function modifies the module's classes with the provided patch_map by e.g. replacing the forward method.
    This way, Layer-wise Relevance Propagation rules can be applied to the module's layers.

    Differs from upstream LXT only in that ``patch_map`` is required: this fork drops the
    bundled per-model maps (``lxt.efficient.models``), which were the sole source of
    LXT's transformers coupling. Callers supply their own map.

    Parameters:
    -----------
    module: Python module
        The module to be patched.
    patch_map: dict
        A dictionary that maps the target classes to the patch functions.
        The patch functions should take the target class as input and return True if the patching was successful.
    verbose: bool
        If True, prints the patched classes.
    """
    if patch_map is None:
        raise ValueError(
            "patch_map is required: this fork of LXT ships no default per-model patch maps. "
            "Supply a {target_class: patch_fn} mapping."
        )

    for target, patch in patch_map.items():
        success = patch(target)
        if not success:
            warn(f"Failed to patch {target.__name__}. Skipping...")
        elif verbose:
            print(f"Patched {target.__name__}")
