#!/usr/bin/env python3
"""Measure computational cost of compiled release policy."""

import time
import json
import pickle
import platform
import numpy as np
import sys

from paths import COMPUTE_COST

# Import environments and compile function
from arena.experiments.settlement.gate import envs_for
from arena.experiments.settlement.outage import compile_outage
from arena.experiments.settlement.core import best_action
from arena.experiments.settlement.simulate import Channel

def measure_compile_time(env, env_name, env_type, n_runs=5):
    """Measure wall time to compile policy (median of n_runs)."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if env_type == "outage":
            policy = compile_outage(env, env_name)
        else:
            # For chain environments, compile means computing best_action
            # which is the decision table
            policy = best_action(env.f, 1.0, np.linspace(0, 1, 501), env.params(rho=1.0))[0]
        end = time.perf_counter()
        times.append(end - start)
    return np.median(times), np.array(times)

def measure_table_size(policy, env_type):
    """Measure table structure, dimensions, number of entries, memory size."""
    if env_type == "outage":
        # CompiledOutagePolicy has:
        # - v_grid: ndarray of shape (n_v,)
        # - tables: list of n_v tables, each of shape (i, l, r, pi)
        n_v = len(policy.v_grid)
        # Get shape of first table
        table_shape = policy.tables[0].shape
        total_entries = sum(t.size for t in policy.tables)

        # Memory: v_grid + all tables
        v_grid_bytes = policy.v_grid.nbytes
        tables_bytes = sum(t.nbytes for t in policy.tables)
        total_bytes = v_grid_bytes + tables_bytes

        # Pickled size
        pickled = pickle.dumps(policy)
        pickled_bytes = len(pickled)

        return {
            "n_v": n_v,
            "table_shape_per_v": list(table_shape),
            "total_entries": int(total_entries),
            "v_grid_bytes": int(v_grid_bytes),
            "tables_bytes": int(tables_bytes),
            "total_bytes": int(total_bytes),
            "pickled_bytes": int(pickled_bytes),
        }
    else:
        # Chain: best_action returns (lab, V, debug_info)
        # lab has shape (N+1, len(pi_grid))
        n_stages = policy.shape[0]
        n_pi = policy.shape[1] if len(policy.shape) > 1 else 1
        total_entries = policy.size
        total_bytes = policy.nbytes
        pickled = pickle.dumps(policy)
        pickled_bytes = len(pickled)

        return {
            "shape": list(policy.shape),
            "total_entries": int(total_entries),
            "total_bytes": int(total_bytes),
            "pickled_bytes": int(pickled_bytes),
        }

def measure_lookup_latency(policy, env_type, n_lookups=100000):
    """Measure latency of single decision lookup (median and 99th percentile)."""
    np.random.seed(42)

    if env_type == "outage":
        # Policy call signature: policy(i, l, r, v, pi)
        # Bounds: i in [0..N+1], l in [0..H], r in {0,1}, v and pi arbitrary
        n_v = len(policy.v_grid)
        max_i = 18  # N+1 for E-outage (N=17)
        max_l = 31  # H+1 (H=30)

        # Generate random lookups
        i_vals = np.random.randint(0, max_i, n_lookups)
        l_vals = np.random.randint(0, max_l, n_lookups)
        r_vals = np.random.randint(0, 2, n_lookups)
        v_vals = np.random.uniform(0.5, 2000, n_lookups)
        pi_vals = np.random.uniform(0, 1, n_lookups)

        # Warm up
        for _ in range(100):
            policy(0, 15, 0, 1.0, 0.5)

        # Measure
        times = []
        for j in range(n_lookups):
            start = time.perf_counter()
            result = policy(int(i_vals[j]), int(l_vals[j]), int(r_vals[j]), float(v_vals[j]), float(pi_vals[j]))
            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)
        median_time = np.median(times)
        p99_time = np.percentile(times, 99)
        throughput = n_lookups / np.sum(times)  # lookups per second

        return {
            "median_us": median_time * 1e6,
            "p99_us": p99_time * 1e6,
            "throughput_per_s": throughput,
        }
    else:
        # For chain policy, it's just indexing into a numpy array
        # policy[i, pi_bin] returns the action
        n_stages = policy.shape[0]
        n_pi = policy.shape[1] if len(policy.shape) > 1 else 1

        # Generate random lookups
        i_vals = np.random.randint(0, n_stages, n_lookups)
        pi_vals = np.random.randint(0, n_pi, n_lookups)

        # Warm up
        for _ in range(100):
            _ = policy[0, 0]

        # Measure
        times = []
        for j in range(n_lookups):
            start = time.perf_counter()
            result = policy[int(i_vals[j]), int(pi_vals[j])]
            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times)
        median_time = np.median(times)
        p99_time = np.percentile(times, 99)
        throughput = n_lookups / np.sum(times)

        return {
            "median_us": median_time * 1e6,
            "p99_us": p99_time * 1e6,
            "throughput_per_s": throughput,
        }

def get_system_info():
    """Get CPU model, Python and numpy versions."""
    import subprocess

    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        lscpu_out = result.stdout
    except Exception as e:
        lscpu_out = f"Error: {e}"

    return {
        "cpu_model_from_lscpu": lscpu_out.split('\n')[0:5],  # First few lines
        "python_version": platform.python_version(),
        "python_version_info": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy_version": np.__version__,
    }

def main():
    # Get environments
    envs = envs_for("mid")

    results = {}

    # Measure for each environment
    for env_name in ["E-fast", "E-slow", "E-outage", "E-fitted"]:
        if env_name not in envs:
            print(f"Skipping {env_name} (not found)")
            continue

        env_type, env, rho = envs[env_name]
        print(f"\nMeasuring {env_name} ({env_type})...")

        # Compile time
        print(f"  Measuring compile time...")
        compile_time_med, compile_times = measure_compile_time(env, env_name, env_type)
        print(f"    Median: {compile_time_med*1000:.3f} ms")

        # Compile once for table size and lookup measurements
        if env_type == "outage":
            policy = compile_outage(env, env_name)
        else:
            # For chain, compile once
            policy = best_action(env.f, 1.0, np.linspace(0, 1, 501), env.params(rho=rho))[0]

        # Table size
        print(f"  Measuring table size...")
        table_info = measure_table_size(policy, env_type)
        print(f"    Total bytes: {table_info.get('total_bytes', 'N/A')}")

        # Lookup latency
        print(f"  Measuring lookup latency (100k lookups)...")
        lookup_info = measure_lookup_latency(policy, env_type)
        print(f"    Median: {lookup_info['median_us']:.3f} µs")
        print(f"    P99: {lookup_info['p99_us']:.3f} µs")
        print(f"    Throughput: {lookup_info['throughput_per_s']:.0f} lookups/s")

        results[env_name] = {
            "type": env_type,
            "compile": {
                "median_ms": compile_time_med * 1000,
                "all_runs_ms": [t * 1000 for t in compile_times],
            },
            "table": table_info,
            "lookup": lookup_info,
        }

    # Add system info
    results["system_info"] = get_system_info()

    # Write results
    COMPUTE_COST.mkdir(parents=True, exist_ok=True)
    with open(COMPUTE_COST / "result.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results written to result.json")

    return results

if __name__ == "__main__":
    main()
