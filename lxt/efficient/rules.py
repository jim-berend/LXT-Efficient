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
import torch.nn.functional as F
from torch.autograd import Function


def identity_rule_implicit(fn, input):
    """
    Implements the identity rule (from Equation 9 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers)
    in a more efficient manner through the Gradient*Input framework.

    Used on element-wise non-linear functions.

    Parameters:
    -----------

    fn: callable
        The function to be called with the input.
        This function must accept a single tensor as input and return a tensor of the same shape.
    input: torch.Tensor
        The input tensor
    """
    
    return identity_rule_implicit_fn.apply(fn, input)


def divide_gradient(input, factor=2):
    """
    Implements the uniform rule (from Equation 7 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers)
    in a more efficient manner through the Gradient*Input framework.

    Used on a tensor after the torch.matmul or the element-wise multiplication operation.

    Parameters:
    -----------
    input: torch.Tensor
        An input tensor
    factor: int
        The factor to divide the gradient/relevance by
    """

    return divide_gradient_fn.apply(input, factor)


def stop_gradient(input):
    """
    Stop the gradient from flowing through the input tensor.
    This rule is used in CP-LRP (from the paper 
    XAI for Transformers: Better Explanations through Conservative Propagation).
    """

    return input.detach()


def layer_norm_identity(input, normalized_shape, weight, bias, eps):
    """
    Implements the identity rule on LayerNorm, i.e. the normalisation statistics are treated
    as constants, **without rewriting the forward pass**.

    The rule only ever intended to change the backward pass, but expressing it as a forward
    rewrite -- `(x - mean) / stop_gradient(std)` -- means giving up the fused `F.layer_norm`
    kernel, because there is no seam inside a fused kernel to put a `detach()` into. The
    forward values then differ from the unpatched model by the fused kernel's rounding, which
    is not small in low precision: measured at 3.3% of a logit row on Gemma-3-12B in bfloat16,
    where the norm is 1152 wide. Here the forward *is* `F.layer_norm`, so forward-value
    preservation holds by construction, and the rule lives in `backward` where it belongs.

    The backward is the analytic gradient of the forward-rewrite formulation, so this is a
    numerics fix and not a rule change. Note in particular that both the `-1/n`
    mean-subtraction term and the frozen `1/std` are needed: with `std` constant,

        d/dx_i sum_j g_j w_j (x_j - mean)/std = (g_i w_i - mean_j(g_j w_j)) / std

    and it is the mean-subtraction term that makes the rule conserve relevance, since it
    turns `sum_i x_i * grad_i` into `<x - mean, g*w>/std == sum_j (y_j - b_j) g_j`. Dropping
    it -- returning just `g*w/std` -- leaves the relevance dependent on `mean(x)`, i.e. on
    exactly the component LayerNorm removes, and is a different rule.

    The statistics are recomputed from a **detached** `input`, which is what makes this equal
    to the forward rewrite at every order rather than only at the first: `stop_gradient(std)`
    puts a constant in the graph, so `grad_input` is linear in the incoming relevance and has
    no path back to `input`, and a double backward -- as used by second-order attribution --
    must see the same. Recomputing from a live `input` would invent a second-order term the
    rule does not have. `backward` is composed of differentiable ops and is deliberately not
    wrapped in `once_differentiable`, so a double backward returns a value instead of raising.

    Being loose about the backward's numerics is fine, and it is why the cheaper recompute is
    preferred over saving the kernel's own `rstd`: LRP deliberately wants a gradient the model
    does not have, so there is no reference gradient to be faithful to. It is the forward that
    has a reference.

    Parameters:
    -----------
    input: torch.Tensor
        The input tensor
    normalized_shape: tuple[int, ...]
        The trailing axes normalised over, i.e. `nn.LayerNorm.normalized_shape`. Honoured in
        full, so a multi-axis LayerNorm is handled.
    weight: torch.Tensor or None
        The affine scale, or None if the module has no affine.
    bias: torch.Tensor or None
        The affine shift, or None. Takes no relevance, so it appears in the forward only.
    eps: float
        The variance epsilon
    """

    return layer_norm_identity_fn.apply(input, normalized_shape, weight, bias, eps)


class identity_rule_implicit_fn(Function):
    """
    Implements the identity rule (from Equation 9 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers)
    in a more efficient manner through the Gradient*Input framework.

    Used on element-wise non-linear functions.

    Parameters:
    -----------

    fn: callable
        The function to be called with the input.
        This function must accept a single tensor as input and return a tensor of the same shape.
    input: torch.Tensor
        The input tensor
    """

    @staticmethod
    def forward(ctx, fn, input, epsilon=1e-10):

        output = fn(input)
        if input.requires_grad:
            ctx.save_for_backward(output/(input + epsilon))
        return output

    @staticmethod
    def backward(ctx, *out_relevance):

        gradient = ctx.saved_tensors[0] * out_relevance[0]

        return None, gradient, None


class divide_gradient_fn(Function):
    """
    Implements the uniform rule (from Equation 7 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers)
    in a more efficient manner through the Gradient*Input framework.

    Used on a tensor after the torch.matmul or the element-wise multiplication operation.

    Parameters:
    -----------
    input: torch.Tensor
        An input tensor
    factor: int
        The factor to divide the gradient/relevance by
    """

    @staticmethod
    def forward(ctx, input, factor=2):
        ctx.factor = factor
        return input

    @staticmethod
    def backward(ctx, *out_relevance):

        return out_relevance[0] / ctx.factor, None


class layer_norm_identity_fn(Function):
    """
    Implements the identity rule on LayerNorm while keeping the fused forward pass.

    See `layer_norm_identity` for why the forward must not be rewritten, why the
    mean-subtraction term is load-bearing, and why the statistics are recomputed detached.
    """

    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps):

        ctx.save_for_backward(input, weight)
        ctx.dims = tuple(range(-len(normalized_shape), 0))
        ctx.eps = eps

        return F.layer_norm(input, normalized_shape, weight, bias, eps)

    @staticmethod
    def backward(ctx, *out_relevance):

        input, weight = ctx.saved_tensors

        # the frozen normalisation statistic, i.e. what stop_gradient(std) stood for
        rstd = (input.detach().var(ctx.dims, keepdim=True, unbiased=False) + ctx.eps).rsqrt()

        relevance = out_relevance[0] if weight is None else out_relevance[0] * weight
        relevance = relevance - relevance.mean(ctx.dims, keepdim=True)

        # the affine parameters take no relevance, and neither does eps
        return relevance * rstd, None, None, None, None
