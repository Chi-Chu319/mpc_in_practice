import pandas as pd
import matplotlib.pyplot as plt
from .file_utils import * 

def render_strong_scale_time_complete(n):
    df = pd.read_csv(cuda_dir(f'strong_scale_complete_{n}/strong_scale_complete_{n}.csv'))

    x = n / df["num_vertex_local"]
    # 8, 16, 32, ..., 128, 256

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, df['total'], label='total time', marker='o')
    plt.plot(x, df['cpu'], label='cpu time', marker='o')

    # kernel time
    plt.plot(x, df['min_from_cluster_kernel'], label='min_from_cluster_kernel time', marker='o')
    plt.plot(x, df['min_to_cluster_kernel'], label='min_to_cluster_kernel time', marker='o')

    # mem copy
    plt.plot(x, df['DtoH'], label='DtoH time', marker='o')
    plt.plot(x, df['HtoD'], label='HtoD time', marker='o')

    plt.xlabel('number of thread')
    plt.ylabel('time (milli-second)')
    plt.title(f'Strong scaling time complete {n}')
    plt.xscale('log')

    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    
def render_strong_scale_time_sparse(n, avg_rank):
    df = pd.read_csv(cuda_dir(f'strong_scale_sparse_{n}_{avg_rank}/strong_scale_sparse_{n}_{avg_rank}.csv'))

    x = n / df["num_vertex_local"]
    # 8, 16, 32, ..., 128, 256

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, df['total'], label='total time', marker='o')
    plt.plot(x, df['cpu'], label='cpu time', marker='o')

    # kernel time
    plt.plot(x, df['min_from_cluster_kernel'], label='min_from_cluster_kernel time', marker='o')
    plt.plot(x, df['min_to_cluster_kernel'], label='min_to_cluster_kernel time', marker='o')

    # mem copy
    plt.plot(x, df['DtoH'], label='DtoH time', marker='o')
    plt.plot(x, df['HtoD'], label='HtoD time', marker='o')

    plt.xlabel('number of thread')
    plt.ylabel('time (milli-second)')
    plt.title(f'Strong scaling time sparse {n} {avg_rank}')
    plt.xscale('log')

    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    
def render_strong_scale_composition_complete(n):
    df = pd.read_csv(cuda_dir(f'strong_scale_complete_{n}/strong_scale_complete_{n}.csv'))

    width = 0.5

    fig, ax = plt.subplots()

    x = list(range(len(df["num_vertex_local"])))
    # 8, 16, 32, ..., 128, 256

    plt.xticks(
        ticks=list(range(len(df["num_vertex_local"]))),
        labels=df["num_vertex_local"].astype(str),
        rotation=90
    )
    plt.xlabel("number of vertex local")

    kernel1_fractions = df['min_to_cluster_kernel'] / df['total']
    kernel2_fractions = df['min_from_cluster_kernel'] / df['total']
    cpu_fractions = df['cpu'] / df['total']
    d_to_h_fractions = df['DtoH'] / df['total']
    h_to_d_fractions = df['HtoD'] / df['total']

    p = ax.bar(x, kernel1_fractions, width, label="kernel1")
    p = ax.bar(x, kernel2_fractions, width, label="kernel2", bottom=kernel1_fractions)
    p = ax.bar(x, cpu_fractions, width, label="cpu", bottom=kernel1_fractions + kernel2_fractions)
    p = ax.bar(x, d_to_h_fractions, width, label="DtoH", bottom=kernel1_fractions + kernel2_fractions + cpu_fractions)
    p = ax.bar(x, h_to_d_fractions, width, label="HtoD", bottom=kernel1_fractions + kernel2_fractions + cpu_fractions + d_to_h_fractions)

    ax.set_title(f'strong scaling {n} complete Time composition')
    ax.legend(loc="upper right")

    plt.show()
    
def render_strong_scale_composition_sparse(n, avg_rank):
    df = pd.read_csv(cuda_dir(f'strong_scale_sparse_{n}_{avg_rank}/strong_scale_sparse_{n}_{avg_rank}.csv'))

    width = 0.5

    fig, ax = plt.subplots()

    x = list(range(len(df["num_vertex_local"])))
    # 8, 16, 32, ..., 128, 256

    plt.xticks(
        ticks=list(range(len(df["num_vertex_local"]))),
        labels=df["num_vertex_local"].astype(str),
        rotation=90
    )
    plt.xlabel("number of vertex local")

    kernel1_fractions = df['min_to_cluster_kernel'] / df['total']
    kernel2_fractions = df['min_from_cluster_kernel'] / df['total']
    cpu_fractions = df['cpu'] / df['total']
    d_to_h_fractions = df['DtoH'] / df['total']
    h_to_d_fractions = df['HtoD'] / df['total']

    p = ax.bar(x, kernel1_fractions, width, label="kernel1")
    p = ax.bar(x, kernel2_fractions, width, label="kernel2", bottom=kernel1_fractions)
    p = ax.bar(x, cpu_fractions, width, label="cpu", bottom=kernel1_fractions + kernel2_fractions)
    p = ax.bar(x, d_to_h_fractions, width, label="DtoH", bottom=kernel1_fractions + kernel2_fractions + cpu_fractions)
    p = ax.bar(x, h_to_d_fractions, width, label="HtoD", bottom=kernel1_fractions + kernel2_fractions + cpu_fractions + d_to_h_fractions)

    ax.set_title(f"strong scaling {n} {avg_rank} Time composition")
    ax.legend(loc="upper right")

    plt.show()

def get_speed_up(t_arr):
    initial = t_arr[0]
    speedup = 1
    
    result = [speedup]
    
    for i in range(1, len(t_arr)):
        result.append((initial / t_arr[i]))
        
    return result

def render_strong_scale_complete_speedup(n, key):
    df = pd.read_csv(cuda_dir(f'strong_scale_complete_{n}/strong_scale_complete_{n}.csv'))

    x = n / df["num_vertex_local"]
    # 8, 16, 32, ..., 128, 256

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, get_speed_up(list(df[key])[::-1])[::-1], label=f'{key} time speed up', marker='o')
    
    plt.xlabel('number of thread')
    plt.ylabel('speed up')
    plt.title(f'Strong scaling complete {n} speed up')
    plt.xscale('log')

    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    
def render_strong_scale_sparse_speedup(n, avg_rank, key):
    df = pd.read_csv(cuda_dir(f'strong_scale_sparse_{n}_{avg_rank}/strong_scale_sparse_{n}_{avg_rank}.csv'))

    x = n / df["num_vertex_local"]
    # 8, 16, 32, ..., 128, 256

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, get_speed_up(list(df[key])[::-1])[::-1], label=f'{key} time speed up', marker='o')
    
    plt.xlabel('number of thread')
    plt.ylabel('speed up')
    plt.title(f'Strong scaling {n} {avg_rank} speed up')
    plt.xscale('log')

    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    
def render_weak_scale_complete_speedup(n):
    df = pd.read_csv(cuda_dir(f'weak_scale_complete{n}/weak_scale_complete{n}.csv'))

    x = df["num_threads"]
    # 8, 16, 32, ..., 128, 256
    
    units = df["num_threads"] * df["num_vertex_local"]
    
    plt.figure(figsize=(10, 6))
    
    parallel_part = (df["min_from_cluster_kernel"] + df["min_to_cluster_kernel"]) / df["total"]
    sequential_part = 1 - parallel_part
    
    speedup = sequential_part + parallel_part * units
    
    plt.plot(x, speedup, label=f'speed up', marker='o')

    plt.xlabel('number of threads')
    plt.ylabel('speed up')
    plt.title(f'Weak scaling speed up')
    plt.xscale('log')

    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()