"""Check CANN reference definitions and the generated nine-input ACLNN ABI."""
import json
import re
import sys
from pathlib import Path


def check_definition(operators: list) -> None:
    for op in operators:
        inputs = {item["name"]: item for item in op["input_desc"]}
        for output in op["output_desc"]:
            source = inputs.get(output["name"])
            if source is None:
                raise RuntimeError("Scatter outputs must reference caller-owned inputs.")
            if any(item.get("param_type", "required") != "required" for item in (source, output)):
                raise RuntimeError(f"{output['name']}: CANN inout references must be required, not optional.")
            if source["type"] != output["type"] or source["format"] != output["format"]:
                raise RuntimeError(f"{output['name']}: reference input/output types and formats must match.")


def check_abi(header: str) -> None:
    match = re.search(r"aclnnKvcacheScatterCopyGetWorkspaceSize\s*\((.*?)\)\s*;", header, re.S)
    if not match or len(re.findall(r"\baclTensor\s*\*", match[1])) != 9:
        raise RuntimeError("Expected nine tensor arguments in the generated in-place ACLNN API; inspect its header.")
    names = re.findall(r"\baclTensor\s*\*\s*(\w+)", match[1])
    names = [n.replace("_", "").lower() for n in names]
    expected = ["hbmkvref", "dramkv", "hbmkperef", "dramkpe", "hbmblocktable", "dramblocktable",
                "sourcetokenids", "destinationslots", "copycounts"]
    if names != expected:
        raise RuntimeError(f"Generated ACLNN tensor order does not match the launcher: {names}")


def main():
    if sys.argv[1] == "--definition":
        check_definition(json.loads(Path(sys.argv[2]).read_text()))
        return
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
