"""Single-die byte-exact checks and timing for BF16 and packed-C8."""
from dataclasses import replace
import sys
from _common import parse_args, print_table, summarize, workload


def correctness_suite(config, device, graph=False):
    from _case import Case, kvcache_ops, torch
    edge_config = replace(config, batch_size=3, copy_cap=129, copy_count=129, source_len=259, hbm_slots=259)
    case = Case(edge_config, device, counts=[0, 1, 129], edges=True)
    case.check()
    # Non-default stream, with caller-owned allocations created on the original stream.
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        case.reset()
        kvcache_ops.kvcache_scatter_copy(*case.inputs)
    torch.npu.synchronize()
    case.verify()
    if graph:
        original = case.counts_cpu.tolist()
        case.set_counts([0, 0, 0])
        case.check()
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            kvcache_ops.kvcache_scatter_copy(*case.inputs)
        for values in (original, [0, 0, 0], original):
            case.set_counts(values)
            case.reset()
            capture.replay()
            torch.npu.synchronize()
            case.verify()
        del capture
    # Shape/dtype failures must be rejected on the host without a device assert.
    invalid = list(case.inputs)
    invalid[6] = case.metadata[2].squeeze(1)
    try:
        kvcache_ops.kvcache_scatter_copy(*invalid)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("An invalid metadata shape was accepted.")
    invalid = list(case.inputs)
    invalid[2] = None if config.dtype == "bf16" else case.targets[0]
    try:
        kvcache_ops.kvcache_scatter_copy(*invalid)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("An invalid optional-KPE combination was accepted.")
    del case
    # Exercise full caller-provided capacity, including an exact-full destination.
    cap = config.copy_cap
    full = Case(replace(config, batch_size=1, copy_count=cap, source_len=cap, hbm_slots=cap), device)
    full.check()
    full.set_counts([0])
    full.check()
    del full
    sparse = Case(replace(config, batch_size=1, copy_cap=65536, copy_count=1, source_len=128, hbm_slots=128), device)
    sparse.check()
    print(f"CHECK dtype={config.dtype}: byte-exact, caller-owned, guards, counts, stream, capacity"
          + (", graph" if graph else "") + " OK", flush=True)


def main(argv=None):
    args = parse_args(argv)
    from _case import Case, torch
    rows = []
    for dtype in args.dtype:
        config = workload(args, dtype, args.batch_size, args.copy_count)
        correctness_suite(config, args.device, args.graph)
        case = Case(config, args.device)
        case.check()
        rows.append(summarize(config, [case.measure()]))
        del case
        torch.npu.empty_cache()
    print_table(rows)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f"ERROR: {error or 'interrupted'}", file=sys.stderr, flush=True)
        sys.exit(1)
