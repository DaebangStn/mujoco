import time
import numpy as np
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import subprocess
from etils import epath
import mujoco
from mujoco import mjx

# Set up paths
MODEL_ROOT_PATH = epath.Path(epath.resource_path('mujoco')) / 'mjx/test_data/humanoid'
NUM_STEPS = 10000

# Create a benchmark function for standard MuJoCo on CPU
def benchmark_mujoco_cpu(steps=NUM_STEPS):
    # Load the humanoid model
    mj_model = mujoco.MjModel.from_xml_path(
        (MODEL_ROOT_PATH / 'humanoid.xml').as_posix())
    mj_data = mujoco.MjData(mj_model)
    
    # Randomize initial state
    np.random.seed(0)
    mj_data.qpos = mj_model.qpos0 + np.random.uniform(
        -0.01, 0.01, size=mj_model.nq)
    mj_data.qvel = np.random.uniform(
        -0.01, 0.01, size=mj_model.nv)
    
    # Benchmark
    start_time = time.time()
    for _ in range(steps):
        mujoco.mj_step(mj_model, mj_data)
    end_time = time.time()
    
    total_time = end_time - start_time
    steps_per_second = steps / total_time
    
    return {
        'platform': 'MuJoCo CPU',
        'batch_size': 1,
        'total_time': total_time,
        'steps': steps,
        'steps_per_second': steps_per_second,
        'environments_per_second': steps_per_second
    }

# Create a benchmark function for MJX on CPU
def benchmark_mjx_cpu(batch_size=1, steps=NUM_STEPS):
    # Force JAX to use CPU
    with jax.default_device(jax.devices('cpu')[0]):
        # Load the humanoid model
        mj_model = mujoco.MjModel.from_xml_path(
            (MODEL_ROOT_PATH / 'humanoid.xml').as_posix())
        mj_data = mujoco.MjData(mj_model)
        
        # Create MJX model
        mjx_model = mjx.put_model(mj_model)
        
        try:
            # Create batch of initial states
            batch_states = []
            rng = jax.random.PRNGKey(0)
            
            for i in range(batch_size):
                rng, key = jax.random.split(rng)
                qpos = mj_model.qpos0 + jax.random.uniform(
                    key, (mj_model.nq,), minval=-0.01, maxval=0.01)
                rng, key = jax.random.split(rng)
                qvel = jax.random.uniform(
                    key, (mj_model.nv,), minval=-0.01, maxval=0.01)
                mjx_data = mjx.put_data(mj_model, mj_data)
                mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
                batch_states.append(mjx_data)
            
            # Stack the states into a batch
            batch = jax.tree_util.tree_map(lambda *xs: jp.stack(xs), *batch_states)
            
            # Measure compilation time separately
            compilation_start = time.time()
            
            # Compile the step function
            jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
            
            # Warm-up JIT compilation
            batch = jit_step(mjx_model, batch)
            batch.qpos.block_until_ready()  # Ensure compilation is complete
            compilation_end = time.time()
            compilation_time = compilation_end - compilation_start
            print(f"    Compilation time: {compilation_time:.4f} seconds")
                        
            # Benchmark
            start_time = time.time()
            for _ in range(steps):
                batch = jit_step(mjx_model, batch)
                batch.qpos.block_until_ready()
            end_time = time.time()
            
            total_time = end_time - start_time
            print(f"    Total time: {total_time:.4f} seconds (compilation ratio: {compilation_time / total_time:.2f})")
            
            steps_per_second = steps * batch_size / total_time
            
            return {
                'platform': 'MJX CPU',
                'batch_size': batch_size,
                'total_time': total_time,
                'steps': steps,
                'steps_per_second': steps_per_second,
                'environments_per_second': steps_per_second
            }
        except Exception as e:
            print(f"Error during MJX CPU benchmark: {e}")
            return {
                'platform': 'MJX CPU',
                'batch_size': batch_size,
                'total_time': float('nan'),
                'steps': steps,
                'steps_per_second': 0,
                'environments_per_second': 0
            }

# Create a benchmark function for MJX on GPU
def benchmark_mjx_gpu(batch_size, steps=NUM_STEPS):
    # Load the humanoid model
    mj_model = mujoco.MjModel.from_xml_path(
        (MODEL_ROOT_PATH / 'humanoid.xml').as_posix())
    mj_data = mujoco.MjData(mj_model)
    
    # Create MJX model
    mjx_model = mjx.put_model(mj_model)
    
    # Create a single initial state first to test if we can even do that
    try:
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        qpos = mj_model.qpos0 + jax.random.uniform(
            key, (mj_model.nq,), minval=-0.01, maxval=0.01)
        rng, key = jax.random.split(rng)
        qvel = jax.random.uniform(
            key, (mj_model.nv,), minval=-0.01, maxval=0.01)
        mjx_data = mjx.put_data(mj_model, mj_data)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
    except Exception as e:
        print(f"Error creating initial state: {e}")
        return {
            'platform': 'MJX GPU',
            'batch_size': batch_size,
            'total_time': float('nan'),
            'compilation_time': float('nan'),
            'execution_time': float('nan'),
            'steps': steps,
            'steps_per_second': 0,
            'environments_per_second': 0
        }
    
    # If we can create one state, try to create the batch
    try:
        # Create batch of initial states - one at a time
        batch_states = []
        rng = jax.random.PRNGKey(0)
        
        for i in range(batch_size):
            jax.clear_caches()  # Clear caches before each state creation
            rng, key = jax.random.split(rng)
            qpos = mj_model.qpos0 + jax.random.uniform(
                key, (mj_model.nq,), minval=-0.01, maxval=0.01)
            rng, key = jax.random.split(rng)
            qvel = jax.random.uniform(
                key, (mj_model.nv,), minval=-0.01, maxval=0.01)
            mjx_data = mjx.put_data(mj_model, mj_data)
            mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
            batch_states.append(mjx_data)
            
            # Force garbage collection
            if i > 0 and i % 10 == 0:
                import gc
                gc.collect()
        
        # Stack the states into a batch
        batch = jax.tree_util.tree_map(lambda *xs: jp.stack(xs), *batch_states)
        
        # Measure compilation time separately
        compilation_start = time.time()
                
        # Compile the step function
        jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        try:
            # Warm-up JIT compilation
            batch = jit_step(mjx_model, batch)
            batch.qpos.block_until_ready()  # Ensure compilation is complete
        except Exception as e:
            print(f"Error during warm-up: {e}")
            print(f"Batch size {batch_size} is too large for available GPU memory.")
            return {
                'platform': 'MJX GPU',
                'batch_size': batch_size,
                'total_time': float('nan'),
                'compilation_time': float('nan'),
                'execution_time': float('nan'),
                'steps': steps,
                'steps_per_second': 0,
                'environments_per_second': 0
            }
        compilation_end = time.time()
        compilation_time = compilation_end - compilation_start
        print(f"    Compilation time: {compilation_time:.4f} seconds")
        
        # Benchmark execution only (post-compilation)
        execution_start = time.time()
        for _ in range(steps):
            batch = jit_step(mjx_model, batch)
            batch.qpos.block_until_ready()  # Force execution to complete
        execution_end = time.time()
        
        execution_time = execution_end - execution_start
        total_time = compilation_time + execution_time
        print(f"    Total time: {total_time:.4f} seconds (compilation ratio: {compilation_time / total_time:.2f})")
        steps_per_second = steps * batch_size / execution_time  # Use execution time only
        
        return {
            'platform': 'MJX GPU',
            'batch_size': batch_size,
            'total_time': total_time,
            'compilation_time': compilation_time,
            'execution_time': execution_time,
            'steps': steps,
            'steps_per_second': steps_per_second,
            'environments_per_second': steps_per_second
        }
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return {
            'platform': 'MJX GPU',
            'batch_size': batch_size,
            'total_time': float('nan'),
            'compilation_time': float('nan'),
            'execution_time': float('nan'),
            'steps': steps,
            'steps_per_second': 0,
            'environments_per_second': 0
        }

def run_benchmarks():
    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    print('Setting environment variable to use GPU rendering:')
    import os
    os.environ['MUJOCO_GL'] = 'egl'
    
    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags
    
    # Set XLA to use CUDA data directory for caching
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/tmp/xla_cache'
    os.environ['JAX_ENABLE_X64'] = 'True'  # Consistent precision
    
    print("\nNote: You may see 'Results do not match the reference' warnings during benchmarking.")
    print("These are part of JAX's GPU kernel autotuning process and indicate small numerical")
    print("differences between different kernel implementations. This autotuning is important")
    print("for achieving optimal performance on your specific GPU hardware.")
    
    # Check if GPU is available
    if subprocess.run('nvidia-smi').returncode:
        raise RuntimeError(
            'Cannot communicate with GPU. '
            'Make sure you are using a GPU runtime.')
    
    # Run MuJoCo CPU benchmark
    print("\nRunning MuJoCo CPU benchmark...")
    mujoco_cpu_result = benchmark_mujoco_cpu()
    print(f"  Steps per second: {mujoco_cpu_result['steps_per_second']:.2f}")
    
    # Run MJX CPU benchmarks with different batch sizes
    print("\nRunning MJX CPU benchmarks...")
    mjx_cpu_results = []
    cpu_batch_sizes = [
        # 1, 2, 4, 
        8
        ]
    
    for batch_size in cpu_batch_sizes:
        print(f"  Batch size: {batch_size}")
        jax.clear_caches()
        import gc
        gc.collect()
        
        result = benchmark_mjx_cpu(batch_size)
        mjx_cpu_results.append(result)
        
        if not np.isnan(result['total_time']):
            print(f"    Steps per second: {result['steps_per_second']:.2f}")
            print(f"    Environments per second: {result['environments_per_second']:.2f}")
        else:
            print(f"    Benchmark failed")
            break
    
    # Run MJX GPU benchmarks with different batch sizes
    print("\nRunning MJX GPU benchmarks...")
    mjx_gpu_results = []
    gpu_batch_sizes = [32]
    
    for batch_size in gpu_batch_sizes:
        print(f"  Batch size: {batch_size}")
        jax.clear_caches()
        import gc
        gc.collect()
        
        result = benchmark_mjx_gpu(batch_size)
        mjx_gpu_results.append(result)
        
        if not np.isnan(result['total_time']):
            print(f"    Steps per second: {result['steps_per_second']:.2f}")
            print(f"    Environments per second: {result['environments_per_second']:.2f}")
        else:
            print(f"    Benchmark failed")
            break
    
    # Combine all results
    all_results = [mujoco_cpu_result] + mjx_cpu_results + mjx_gpu_results
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Group by platform
    platforms = ['MuJoCo CPU', 'MJX CPU', 'MJX GPU']
    colors = ['blue', 'green', 'red']
    
    for platform, color in zip(platforms, colors):
        platform_results = [r for r in all_results if r['platform'] == platform]
        if platform_results:
            plt.plot([r['batch_size'] for r in platform_results], 
                     [r['environments_per_second'] for r in platform_results], 
                     'o-', linewidth=2, label=platform, color=color)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Batch Size')
    plt.ylabel('Environments Per Second')
    plt.title('Physics Engine Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('physics_engine_comparison.png')
    plt.close()
    
    # Create a summary table with compilation time
    print("\nBenchmark Results Summary (with compilation time):")
    print("-" * 120)
    print(f"{'Platform':^15} | {'Batch Size':^10} | {'Compilation (s)':^15} | {'Execution (s)':^15} | {'Total (s)':^15} | {'Steps/Sec':^12} | {'Envs/Sec':^12}")
    print("-" * 120)
    
    for r in all_results:
        if not np.isnan(r.get('total_time', float('nan'))):
            comp_time = r.get('compilation_time', 0)
            exec_time = r.get('execution_time', r.get('total_time', 0))
            total_time = r.get('total_time', exec_time)
            print(f"{r['platform']:^15} | {r['batch_size']:^10} | {comp_time:^15.4f} | {exec_time:^15.4f} | {total_time:^15.4f} | {r['steps_per_second']:^12.2f} | {r['environments_per_second']:^12.2f}")
    
    # Calculate speedups
    mujoco_cpu_speed = mujoco_cpu_result['environments_per_second']
    
    print("\nSpeedup Relative to MuJoCo CPU:")
    print("-" * 70)
    print(f"{'Platform':^15} | {'Batch Size':^10} | {'Speedup':^10}")
    print("-" * 70)
    
    for r in all_results:
        if not np.isnan(r['total_time']):
            speedup = r['environments_per_second'] / mujoco_cpu_speed
            print(f"{r['platform']:^15} | {r['batch_size']:^10} | {speedup:^10.2f}x")
    
    # Print conclusions
#     print("""
# Performance Comparison Conclusions:

# 1. Single Environment Performance: 
#    - MuJoCo CPU provides a baseline for single environment simulation.
#    - MJX on CPU for a single environment may have overhead from JAX compilation.
#    - MJX on GPU for a single environment shows the overhead of GPU data transfer.

# 2. Batch Processing Advantage:
#    - MJX on GPU shows dramatic performance improvements with increasing batch size.
#    - MJX on CPU also benefits from vectorization but with more modest gains.

# 3. Scaling Characteristics:
#    - MJX GPU demonstrates near-linear scaling with batch size up to memory limits.
#    - The crossover point where MJX GPU outperforms CPU implementations occurs at 
#      relatively small batch sizes.

# 4. Practical Applications:
#    - For single-environment simulation, traditional MuJoCo may be sufficient.
#    - For RL training and other parallel simulation workloads, MJX on GPU offers 
#      substantial performance benefits.
# """)

if __name__ == "__main__":
    run_benchmarks()