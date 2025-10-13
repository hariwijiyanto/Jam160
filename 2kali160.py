import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import sys
import argparse
import struct
import random
import threading
from queue import Queue

def int_to_bigint_np(val):
    """Convert integer to BigInt numpy array"""
    bigint_arr = np.zeros(8, dtype=np.uint32)
    for j in range(8):
        bigint_arr[j] = (val >> (32 * j)) & 0xFFFFFFFF
    return bigint_arr

def bigint_np_to_int(bigint_arr):
    """Convert BigInt numpy array to integer"""
    val = 0
    for j in range(8):
        val |= int(bigint_arr[j]) << (32 * j)
    return val

def load_target_hashes(filename):
    """Load target hash160 from file (hexadecimal format)"""
    targets_bin = bytearray()
    targets_hex = []

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if len(line) == 40:  # Hash160 is 40 hex chars (20 bytes)
                try:
                    hash_bytes = bytes.fromhex(line)
                    if len(hash_bytes) == 20:
                        targets_bin.extend(hash_bytes)
                        targets_hex.append(line.lower())
                        print(f"[+] Loaded target hash: {line}")
                    else:
                        print(f"[!] Warning: Invalid hash length on line {line_num}: {line}")
                except ValueError:
                    print(f"[!] Warning: Invalid hex format on line {line_num}: {line}")
            elif line and not line.startswith('#') and len(line) > 0:
                print(f"[!] Warning: Invalid format on line {line_num} (expected 40 hex chars): {line}")

    if len(targets_hex) == 0:
        print(f"[!] Tidak ada hash160 valid di {filename}.")
        return None, []

    print(f"[*] Total {len(targets_hex)} target hash160 loaded")
    return targets_bin, targets_hex

def init_secp256k1_constants(mod, device_id=0):
    """Initialize secp256k1 curve constants in GPU constant memory"""
    # Set device context
    cuda.Context.synchronize()
    
    # Prime modulus p
    p_data = np.array([
        0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    ], dtype=np.uint32)
    const_p_gpu = mod.get_global("const_p")[0]
    cuda.memcpy_htod(const_p_gpu, p_data)

    # Curve order n
    n_data = np.array([
        0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
        0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    ], dtype=np.uint32)
    const_n_gpu = mod.get_global("const_n")[0]
    cuda.memcpy_htod(const_n_gpu, n_data)

    # Base point G in Jacobian coordinates
    g_x = np.array([
        0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
        0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
    ], dtype=np.uint32)
    g_y = np.array([
        0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
        0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
    ], dtype=np.uint32)
    g_z = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
    g_infinity = np.array([False], dtype=np.bool_)

    # Create structured array for ECPointJac
    ecpoint_jac_dtype = np.dtype([
        ('X', np.uint32, 8),
        ('Y', np.uint32, 8),
        ('Z', np.uint32, 8),
        ('infinity', np.bool_)
    ])
    g_jac = np.zeros(1, dtype=ecpoint_jac_dtype)
    g_jac['X'], g_jac['Y'], g_jac['Z'], g_jac['infinity'] = g_x, g_y, g_z, g_infinity

    const_G_gpu = mod.get_global("const_G_jacobian")[0]
    cuda.memcpy_htod(const_G_gpu, g_jac)

def run_precomputation(mod, device_id=0):
    """Run GPU precomputation kernel"""
    precompute_kernel = mod.get_function("precompute_G_table_kernel")
    precompute_kernel(block=(1, 1, 1))
    cuda.Context.synchronize()
    print(f"[*] GPU {device_id}: Precomputation table selesai.")

class GPUWorker(threading.Thread):
    def __init__(self, device_id, start_key, range_min, range_max, keys_per_launch, 
                 target_bin, target_hex_list, kernel_mode, result_queue, stop_event):
        threading.Thread.__init__(self)
        self.device_id = device_id
        self.start_key = start_key
        self.range_min = range_min
        self.range_max = range_max
        self.keys_per_launch = keys_per_launch
        self.target_bin = target_bin
        self.target_hex_list = target_hex_list
        self.kernel_mode = kernel_mode
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.total_iterations = 0
        self.start_time = time.time()
        
    def run(self):
        try:
            # Set device context for this thread
            cuda.init()
            device = cuda.Device(self.device_id)
            context = device.make_context()
            
            print(f"[*] GPU {self.device_id}: Initializing...")
            
            # Load kernel code
            with open('kernel160.cu', 'r') as f:
                kernel160_code = f.read()

            mod = SourceModule(kernel160_code, no_extern_c=False, options=['-std=c++11', '-arch=sm_75'])
            
            # Initialize constants
            init_secp256k1_constants(mod, self.device_id)
            
            # Run precomputation
            run_precomputation(mod, self.device_id)
            
            # Setup kernel
            if self.kernel_mode == 'optimized':
                find_hash_kernel = mod.get_function("find_hash_kernel_optimized")
                print(f"[*] GPU {self.device_id}: Menggunakan kernel optimized dengan precomputation")
            else:
                find_hash_kernel = mod.get_function("find_hash_kernel")
                print(f"[*] GPU {self.device_id}: Menggunakan kernel standard")
            
            # Allocate GPU memory
            d_target_hashes = cuda.mem_alloc(len(self.target_bin))
            cuda.memcpy_htod(d_target_hashes, np.frombuffer(self.target_bin, dtype=np.uint8))
            
            d_result = cuda.mem_alloc(32)
            d_found_flag = cuda.mem_alloc(4)
            cuda.memset_d32(d_result, 0, 8)
            cuda.memset_d32(d_found_flag, 0, 1)
            
            # Calculate work distribution for this GPU
            range_size = self.range_max - self.range_min + 1
            half_range = range_size // 2
            
            if self.device_id == 0:
                my_range_min = self.range_min
                my_range_max = self.range_min + half_range - 1
            else:
                my_range_min = self.range_min + half_range
                my_range_max = self.range_max
            
            print(f"[*] GPU {self.device_id}: Range {hex(my_range_min)} - {hex(my_range_max)}")
            
            step_np = int_to_bigint_np(1)
            start_scalar_np = int_to_bigint_np(self.start_key)
            
            iteration_offset = 0
            my_range_size = my_range_max - my_range_min + 1
            
            found_flag_host = np.zeros(1, dtype=np.int32)
            
            while iteration_offset < my_range_size and not self.stop_event.is_set():
                iterations_left = my_range_size - iteration_offset
                iterations_this_launch = min(self.keys_per_launch, iterations_left)
                
                if iterations_this_launch <= 0:
                    break
                
                block_size = 256
                grid_size = (iterations_this_launch + block_size - 1) // block_size
                
                # Jalankan kernel
                find_hash_kernel(
                    cuda.In(start_scalar_np),
                    np.uint64(iterations_this_launch),
                    cuda.In(step_np),
                    d_target_hashes,
                    np.int32(len(self.target_hex_list)),
                    d_result,
                    d_found_flag,
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                
                cuda.Context.synchronize()
                
                self.total_iterations += iterations_this_launch
                iteration_offset += iterations_this_launch
                
                cuda.memcpy_dtoh(found_flag_host, d_found_flag)
                
                # Check if found
                if found_flag_host[0] == 1:
                    result_buffer = np.zeros(8, dtype=np.uint32)
                    cuda.memcpy_dtoh(result_buffer, d_result)
                    privkey_int = bigint_np_to_int(result_buffer)
                    
                    print(f"\n[+] GPU {self.device_id}: HASH160 DITEMUKAN!")
                    print(f"    Private Key: {hex(privkey_int)}")
                    
                    # Send result to main thread
                    self.result_queue.put({
                        'device_id': self.device_id,
                        'private_key': privkey_int,
                        'iterations': self.total_iterations,
                        'time': time.time() - self.start_time
                    })
                    
                    self.stop_event.set()
                    break
                
                # Progress reporting
                elapsed = time.time() - self.start_time
                speed = self.total_iterations / elapsed if elapsed > 0 else 0
                progress = 100 * iteration_offset / my_range_size
                
                progress_str = (f"GPU{self.device_id}: {iteration_offset:,}/{my_range_size:,} "
                              f"({progress:.1f}%) | {speed:,.0f} it/s")
                sys.stdout.write(f'\r{progress_str.ljust(80)}')
                sys.stdout.flush()
            
            if not self.stop_event.is_set():
                print(f"\n[*] GPU {self.device_id}: Pencarian selesai, tidak ditemukan.")
            
            # Cleanup
            context.pop()
            
        except Exception as e:
            print(f"\n[!] GPU {self.device_id}: Error - {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='CUDA Hash160 Search Multi-GPU')
    parser.add_argument('--start', type=lambda x: int(x, 0), required=True, help='Skalar awal perkalian')
    parser.add_argument('--range-min', type=lambda x: int(x, 0), required=True, help='Batas bawah range pengali')
    parser.add_argument('--range-max', type=lambda x: int(x, 0), required=True, help='Batas atas range pengali')
    parser.add_argument('--file', required=True, help='File target hash160 (hexadecimal, 40 karakter)')
    parser.add_argument('--keys-per-launch', type=int, default=2**20, help='Jumlah iterasi per batch GPU')
    parser.add_argument('--kernel-mode', choices=['optimized', 'standard'], default='optimized', help='Mode kernel yang digunakan')
    parser.add_argument('--gpus', type=int, default=2, help='Jumlah GPU yang digunakan')

    args = parser.parse_args()

    # Validasi range
    if args.range_min >= args.range_max:
        print("[!] ERROR: range-min harus lebih kecil dari range-max")
        sys.exit(1)

    range_size = args.range_max - args.range_min + 1
    print(f"[*] Range size: {range_size:,} kemungkinan multiplier")

    # Load target hashes
    print(f"[*] Memuat target hash160 dari {args.file}...")
    target_bin, target_hex_list = load_target_hashes(args.file)
    if target_bin is None:
        sys.exit(1)

    num_targets = len(target_hex_list)
    print(f"[*] Loaded {num_targets} target hash160")

    # Check available GPUs
    num_gpus = cuda.Device.count()
    print(f"[*] Found {num_gpus} GPU(s)")
    
    if args.gpus > num_gpus:
        print(f"[!] Warning: Requested {args.gpus} GPUs but only {num_gpus} available")
        args.gpus = num_gpus

    print(f"\n[*] Memulai pencarian multi-GPU:")
    print(f"    Start scalar: {hex(args.start)}")
    print(f"    Range pengali: {hex(args.range_min)} - {hex(args.range_max)}")
    print(f"    Range size: {range_size:,} kemungkinan")
    print(f"    Target hash160: {num_targets} hashes")
    print(f"    Kernel mode: {args.kernel_mode}")
    print(f"    GPU count: {args.gpus}")

    # Create shared objects
    result_queue = Queue()
    stop_event = threading.Event()
    
    # Start GPU workers
    workers = []
    for gpu_id in range(args.gpus):
        worker = GPUWorker(
            device_id=gpu_id,
            start_key=args.start,
            range_min=args.range_min,
            range_max=args.range_max,
            keys_per_launch=args.keys_per_launch,
            target_bin=target_bin,
            target_hex_list=target_hex_list,
            kernel_mode=args.kernel_mode,
            result_queue=result_queue,
            stop_event=stop_event
        )
        workers.append(worker)
        worker.start()
        time.sleep(1)  # Stagger startup

    # Wait for results
    try:
        while any(worker.is_alive() for worker in workers) and not stop_event.is_set():
            time.sleep(1)
            
            # Check for results
            if not result_queue.empty():
                result = result_queue.get()
                break
        
        # If no result found, wait for all threads to complete
        if not stop_event.is_set():
            for worker in workers:
                worker.join()
                
        # Process results
        if not result_queue.empty():
            result = result_queue.get()
            print(f"\n\n[+] PENCARIAN BERHASIL!")
            print(f"    Ditemukan oleh GPU {result['device_id']}")
            print(f"    Private Key: {hex(result['private_key'])}")
            print(f"    Iterations: {result['iterations']:,}")
            print(f"    Waktu: {result['time']:.2f} detik")
            
            # Save result
            with open("found_hash160_multi_gpu.txt", "w") as f:
                f.write(f"Private Key: {hex(result['private_key'])}\n")
                f.write(f"Found by GPU: {result['device_id']}\n")
                f.write(f"Total iterations: {result['iterations']}\n")
                f.write(f"Search time: {result['time']:.2f} seconds\n")
                f.write(f"Range: {hex(args.range_min)} - {hex(args.range_max)}\n")
                f.write(f"Start scalar: {hex(args.start)}\n")
                f.write(f"Target hashes:\n")
                for hash_hex in target_hex_list:
                    f.write(f"  {hash_hex}\n")
        else:
            print(f"\n\n[+] Pencarian selesai. Tidak ditemukan.")
            total_iterations = sum(worker.total_iterations for worker in workers)
            total_time = time.time() - workers[0].start_time if workers else 0
            print(f"    Total iterasi: {total_iterations:,}")
            print(f"    Waktu pencarian: {total_time:.2f} detik")
            
    except KeyboardInterrupt:
        print(f"\n\n[!] Dihentikan oleh pengguna.")
        stop_event.set()
        for worker in workers:
            worker.join()
        
        total_iterations = sum(worker.total_iterations for worker in workers)
        print(f"    Total iterasi: {total_iterations:,}")

if __name__ == '__main__':
    main()
