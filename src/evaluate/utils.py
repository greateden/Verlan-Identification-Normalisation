import torch

MIN_COMPUTE_CAPABILITY = (3, 5)


def get_device(min_capability=MIN_COMPUTE_CAPABILITY) -> str:
    """Return "cuda" if a sufficiently capable GPU is available, else "cpu".

    A GPU is considered usable only if its compute capability meets or exceeds
    ``min_capability``. When the GPU is too old, we fall back to CPU to avoid
    runtime errors like ``"sm_13" not built"`` from PyTorch.
    """
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major * 10 + minor >= min_capability[0] * 10 + min_capability[1]:
                return "cuda"
            else:
                print(
                    f"[warn] GPU compute capability {major}.{minor} < "
                    f"{min_capability[0]}.{min_capability[1]}; falling back to CPU"
                )
        except Exception:
            # If capability check fails for any reason, fall back to CPU.
            pass
    return "cpu"
