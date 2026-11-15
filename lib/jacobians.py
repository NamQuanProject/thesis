"""IA3 target-probability Jacobians for the sample-independent, frozen ViT.

The reference differentiates a whole batch once per output row. For this ViT,
outputs have no cross-example dependencies. Mapping a single-example gradient
therefore computes the same Jacobian without repeatedly backpropagating zeros
through the other examples. Floating-point operation order can still differ.
"""
from torch.func import functional_call, grad, jacrev, vmap


def build_ia3_jacobian(net, mode="batch", chunk_size=1):
    if chunk_size < 0:
        raise ValueError("Jacobian chunk size must be nonnegative")
    chunk = chunk_size or None
    if mode == "batch":
        return jacrev(lambda params, x, y: functional_call(net, params, (x, y)),
                      argnums=0, chunk_size=chunk)
    if mode != "per_sample":
        raise ValueError(f"Unknown Jacobian mode: {mode}")
    if net.training:
        raise ValueError("Per-sample Jacobians require the frozen evaluation model")

    def single(params, x, y):
        return functional_call(net, params, (x.unsqueeze(0), y.unsqueeze(0))).squeeze(0)

    return vmap(grad(single, argnums=0), in_dims=(None, 0, 0),
                randomness="error", chunk_size=chunk)
