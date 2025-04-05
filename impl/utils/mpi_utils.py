import pandas as pd
import matplotlib.pyplot as plt
from .file_utils import *

def get_strong_scaling_speed_up(t_arr):
    initial = t_arr[0]
    speedup = 1
    
    result = [speedup]
    
    for i in range(1, len(t_arr)):
        result.append((initial / t_arr[i]))
        
    return result

def get_weak_scaling_speedup(df):
    t_parallel = df["t_total"] - df["t_rank0"] - df["t_mpi"]

    num_threads = df["num_threads"]
    t_computation = num_threads * t_parallel + df["t_rank0"] + df["t_mpi"]  
    
    speedup = []
    
    for i in t_computation:
        speedup.append(i / t_computation[0])
    
    return speedup