from __future__ import annotations


def relax_huggingface_hub_upper_bound() -> None:
    """Process-local workaround for old Transformers with a newer hub package.

    Some project environments pair Transformers 4.46 with huggingface-hub 1.x.
    Transformers rejects that during import through a metadata version check even
    before any hub API is used. This shim leaves the environment untouched and
    only relaxes the reported version inside the current Python process.
    """
    try:
        import importlib.metadata
    except Exception:
        return

    current_version = importlib.metadata.version
    if getattr(current_version, "_msrs_hf_compat_patched", False):
        return

    def patched_version(distribution_name: str) -> str:
        value = current_version(distribution_name)
        if distribution_name == "huggingface-hub":
            major = value.split(".", 1)[0]
            if major.isdigit() and int(major) >= 1:
                return "0.36.2"
        return value

    setattr(patched_version, "_msrs_hf_compat_patched", True)
    importlib.metadata.version = patched_version
