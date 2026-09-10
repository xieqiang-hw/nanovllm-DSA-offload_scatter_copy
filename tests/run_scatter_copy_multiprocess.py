"""Sweep BF16 and C8 using one spawned process per die, without result files."""
from __future__ import annotations

import multiprocessing as mp
import faulthandler
import os
import sys
import time
import traceback
from multiprocessing.connection import wait

from _common import cases, parse_args, print_table, summarize


def worker(config, device, barrier, connection):
    faulthandler.enable(all_threads=True)
    try:
        from _case import Case
        case = Case(config, f"npu:{device}")
        case.check()
        connection.send({"result": case.measure(barrier)})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def run_workers(config, devices, timeout=600, *, target=worker):
    context = mp.get_context("spawn")
    barrier = context.Barrier(len(devices), timeout=timeout)
    processes, pipes, results = [], {}, {}
    deadline = time.monotonic() + timeout
    try:
        for device in devices:
            receive, send = context.Pipe(duplex=False)
            process = context.Process(target=target, args=(config, device, barrier, send))
            try:
                process.start()
            except BaseException:
                receive.close()
                raise
            finally:
                send.close()
            processes.append(process)
            pipes[receive] = device
        while pipes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Case timed out after {timeout}s; pending dies={list(pipes.values())}")
            for connection in wait(list(pipes), timeout=min(remaining, 0.2)):
                device = pipes.pop(connection)
                try:
                    message = connection.recv()
                except EOFError as error:
                    raise RuntimeError(f"die {device} exited without a result") from error
                finally:
                    connection.close()
                if "error" in message:
                    raise RuntimeError(f"die {device} failed:\n{message['error']}")
                results[device] = message["result"]
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
            if process.exitcode != 0:
                raise RuntimeError(f"Worker did not exit successfully: exitcode={process.exitcode}")
        return [results[device] for device in devices]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in pipes:
            connection.close()


def main(argv=None):
    args = parse_args(argv, multi=True)
    if args.dry_run:
        print(f"visible-devices={args.visible_devices or 'runtime visible set'}; source-pool=private")
        for cards, config in cases(args):
            print(f"CASE cards={cards} {config}")
        return
    if args.visible_devices is not None:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.visible_devices))
    from _case import torch
    available = torch.npu.device_count()
    if max(args.cards) > available:
        raise ValueError(f"Requested {max(args.cards)} dies, only {available} are visible.")
    rows = []
    print("Timing in us; avg_bandwidth in GB/s; card-count counts dies; source-pool=private", flush=True)
    for cards, config in cases(args):
        print(f"RUN dtype={config.dtype} cards={cards} batch={config.batch_size} copy-count={config.copy_count}", flush=True)
        rows.append(summarize(config, run_workers(config, list(range(cards)), args.timeout)))
    print_table(rows)


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f"ERROR: {error or 'interrupted'}", file=sys.stderr, flush=True)
        sys.exit(1)
