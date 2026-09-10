"""Reject a generated ACLNN ABI that does not match the nine-input launcher."""
import re
import sys
from pathlib import Path


def check_abi(header: str) -> None:
    match = re.search(r"aclnnKvcacheScatterCopyGetWorkspaceSize\s*\((.*?)\)\s*;", header, re.S)
    if not match or len(re.findall(r"\baclTensor\s*\*", match[1])) != 9:
        raise RuntimeError("Expected nine tensor arguments in the generated in-place ACLNN API; inspect its header.")
    names = re.findall(r"\baclTensor\s*\*\s*(\w+)", match[1])
    names = [n.replace("_", "").lower().replace("optional", "").removesuffix("ref") for n in names]
    expected = ["hbmkv", "dramkv", "hbmkpe", "dramkpe", "hbmblocktable", "dramblocktable",
                "sourcetokenids", "destinationslots", "copycounts"]
    if names != expected:
        raise RuntimeError(f"Generated ACLNN tensor order does not match the launcher: {names}")


def main():
    root = Path(sys.argv[1])
    headers = list(root.glob("vendors/*/op_api/include/aclnn_kvcache_scatter_copy.h"))
    if len(headers) != 1:
        raise RuntimeError("Expected one generated scatter ACLNN header.")
    check_abi(headers[0].read_text())
    metadata = list(root.rglob("binary_info_config.json"))
    if not any("KvcacheScatterCopy" in p.read_text() for p in metadata):
        raise RuntimeError("KvcacheScatterCopy is absent from compiled kernel metadata.")
    print("CHECK build: nine-input reference ABI and kernel metadata OK")


if __name__ == "__main__":
    main()
