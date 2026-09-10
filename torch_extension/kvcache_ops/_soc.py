"""SoC selection for building/loading this checkout; no Torch dependency."""
import ctypes


def normalize_soc(name: str) -> str:
    name = name.strip().lower()
    if name == "ascend910c" or name.startswith("ascend910_93"):
        return "ascend910_93"
    if name == "ascend950" or name.startswith(("ascend950pr", "ascend950dt")):
        return "ascend950"
    raise ValueError(f"Unsupported SoC {name!r}; expected Ascend 910C/A3 or 950/A5.")


def detect_soc() -> str:
    try:
        runtime = ctypes.CDLL("libascendcl.so")
        getter = runtime.aclrtGetSocName
        getter.restype, getter.argtypes = ctypes.c_char_p, []
        name = getter()
        if not name:
            raise RuntimeError("aclrtGetSocName returned an empty name")
        return normalize_soc(name.decode())
    except (OSError, AttributeError, RuntimeError) as error:
        raise RuntimeError("Cannot detect SoC; source the CANN environment or set SOC_VERSION for building.") from error


if __name__ == "__main__":
    import os
    print(normalize_soc(os.environ["SOC_VERSION"]) if os.getenv("SOC_VERSION") else detect_soc())
