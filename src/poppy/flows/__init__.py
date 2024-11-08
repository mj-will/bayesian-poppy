def get_flow_wrapper(backend: str = "zuko", flow_matching: bool = False):
    """Get the wrapper for the flow implementation."""
    if backend == "zuko":
        from .torch.flows import ZukoFlow, ZukoFlowMatching

        if flow_matching:
            return ZukoFlowMatching
        else:
            return ZukoFlow
    elif backend == "flowjax":
        from .jax.flows import FlowJax

        if flow_matching:
            raise NotImplementedError(
                "Flow matching not implemented for JAX backend"
            )
        return FlowJax
    else:
        raise ValueError(f"Unknown backend: {backend}")
