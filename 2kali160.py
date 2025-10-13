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
from concurrent.futures import ThreadPoolExecutor

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

def init_secp256k1_constants(mod, device_id):
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

def run_precomputation(mod):
    """Run GPU precomputation kernel"""
    precompute_kernel = mod.get_function("precompute_G_table_kernel")
    precompute_kernel(block=(1, 1, 1))
    cuda.Context.synchronize()

class GPUSearcher:
    def __init__(self, device_id, kernel_code, target_bin, target_hex_list, args):
        self.device_id = device_id
        self.kernel_code = kernel_code
        self.target_bin = target_bin
        self.target_hex_list = target_hex_list
        self.args = args
        self.found = False
        self.result = None
        self.total_iterations = 0
        self.start_time = time.time()
        
    def run_search(self):
        """Run search on this specific GPU"""
        try:
            # Set device context
            device = cuda.Device(self.device_id)
            context = device.make_context()
            
            print(f"[GPU {self.device_id}] Initializing...")
            
            # Compile kernel for this device
            mod = SourceModule(self.kernel_code, no_extern_c=False, 
                             options=['-std=c++11', f'-arch=sm_{device.compute_capability()[0]}{device.compute_capability()[1]}'])
            
            # Initialize constants
            init_secp256k1_constants(mod, self.device_id)
            
            print(f"[GPU {self.device_id}] Running precomputation...")
            run_precomputation(mod)
            
            # Select kernel based on mode
            if self.args.kernel_mode == 'optimized':
                find_hash_kernel = mod.get_function("find_hash_kernel_optimized")
                print(f"[GPU {self.device_id}] Using optimized kernel with precomputation")
            else:
                find_hash_kernel = mod.get_function("find_hash_kernel")
                print(f"[GPU {self.device_id}] Using standard kernel")
            
            # Initialize GPU memory
            d_target_hashes = cuda.mem_alloc(len(self.target_bin))
            cuda.memcpy_htod(d_target_hashes, np.frombuffer(self.target_bin, dtype=np.uint8))
            
            # Allocate result memory
            d_result = cuda.mem_alloc(32)
            d_found_flag = cuda.mem_alloc(4)
            cuda.memset_d32(d_result, 0, 8)
            cuda.memset_d32(d_found_flag, 0, 1)
            
            num_targets = len(self.target_bin) // 20
            range_size = self.args.range_max - self.args.range_min + 1
            
            # Calculate work distribution for this GPU
            gpu_count = self.args.gpu_count
            gpu_range_size = range_size // gpu_count
            gpu_range_min = self.args.range_min + (self.device_id * gpu_range_size)
            gpu_range_max = gpu_range_min + gpu_range_size - 1
            
            # Handle remainder for last GPU
            if self.device_id == gpu_count - 1:
                gpu_range_max = self.args.range_max
            
            range_min_np = int_to_bigint_np(gpu_range_min)
            range_max_np = int_to_bigint_np(gpu_range_max)
            step_np = int_to_bigint_np(1)
            
            print(f"[GPU {self.device_id}] Range: {hex(gpu_range_min)} - {hex(gpu_range_max)}")
            print(f"[GPU {self.device_id}] Range size: {gpu_range_max - gpu_range_min + 1:,}")
            
            current_start = self.args.start
            start_scalars_tried = 0
            found_flag_host = np.zeros(1, dtype=np.int32)
            
            while (current_start >= 1 and 
                   start_scalars_tried < self.args.max_start_scalars and 
                   not self.found and 
                   not any(searcher.found for searcher in self.other_searchers if searcher != self)):
                
                # Reset found flag for new start scalar
                cuda.memset_d32(d_found_flag, 0, 1)
                found_flag_host[0] = 0
                
                start_scalar_np = int_to_bigint_np(current_start)
                iteration_offset = 0
                total_iterations_current = 0
                
                # Loop for current range
                while (iteration_offset < gpu_range_size and 
                       found_flag_host[0] == 0 and 
                       not any(searcher.found for searcher in self.other_searchers if searcher != self)):
                    
                    iterations_left = gpu_range_size - iteration_offset
                    iterations_this_launch = min(self.args.keys_per_launch, iterations_left)
                    
                    if iterations_this_launch <= 0:
                        break
                    
                    block_size = 256
                    grid_size = (iterations_this_launch + block_size - 1) // block_size
                    
                    # Run hash search kernel
                    find_hash_kernel(
                        cuda.In(start_scalar_np),
                        np.uint64(iterations_this_launch),
                        cuda.In(step_np),
                        d_target_hashes,
                        np.int32(num_targets),
                        d_result,
                        d_found_flag,
                        block=(block_size, 1, 1),
                        grid=(grid_size, 1)
                    )
                    
                    cuda.Context.synchronize()
                    
                    total_iterations_current += iterations_this_launch
                    self.total_iterations += iterations_this_launch
                    iteration_offset += iterations_this_launch
                    
                    cuda.memcpy_dtoh(found_flag_host, d_found_flag)
                    
                    elapsed = time.time() - self.start_time
                    speed = self.total_iterations / elapsed if elapsed > 0 else 0
                    progress_current = 100 * iteration_offset / gpu_range_size
                    
                    progress_str = (f"[GPU {self.device_id}] Start: {hex(current_start)} | "
                                  f"Progress: {iteration_offset:,}/{gpu_range_size:,} ({progress_current:.1f}%) | "
                                  f"Speed: {speed:,.0f} it/s | "
                                  f"Running: {elapsed:.0f}s")
                    sys.stdout.write('\r' + progress_str.ljust(120))
                    sys.stdout.flush()
                
                # Check results for this start scalar
                if found_flag_host[0] == 1:
                    self.found = True
                    sys.stdout.write('\n')
                    
                    # Read result from GPU
                    result_buffer = np.zeros(8, dtype=np.uint32)
                    cuda.memcpy_dtoh(result_buffer, d_result)
                    
                    # Extract private key
                    found_privkey_np = result_buffer
                    privkey_int = bigint_np_to_int(found_privkey_np)
                    
                    self.result = {
                        'private_key': privkey_int,
                        'start_scalar': current_start,
                        'total_iterations': self.total_iterations,
                        'gpu_id': self.device_id,
                        'search_time': time.time() - self.start_time
                    }
                    
                    print(f"\n[GPU {self.device_id}] HASH160 DITEMUKAN!")
                    print(f"    Private Key: {hex(privkey_int)}")
                    print(f"    Start scalar: {hex(current_start)}")
                    print(f"    Total iterasi: {self.total_iterations:,}")
                    print(f"    Waktu pencarian: {self.result['search_time']:.2f} detik")
                    
                else:
                    # Not found in this start scalar, continue to next
                    current_start += 1
                    start_scalars_tried += 1
            
            # Cleanup
            if d_target_hashes:
                d_target_hashes.free()
            if d_result:
                d_result.free()
            if d_found_flag:
                d_found_flag.free()
                
            context.pop()
            
        except Exception as e:
            print(f"\n[GPU {self.device_id}] Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='CUDA Hash160 Search dengan Multi-GPU Support')
    parser.add_argument('--start', type=lambda x: int(x, 0), required=True, help='Skalar awal perkalian')
    parser.add_argument('--range-min', type=lambda x: int(x, 0), required=True, help='Batas bawah range pengali')
    parser.add_argument('--range-max', type=lambda x: int(x, 0), required=True, help='Batas atas range pengali')
    parser.add_argument('--file', required=True, help='File target hash160 (hexadecimal, 40 karakter)')
    parser.add_argument('--keys-per-launch', type=int, default=2**20, help='Jumlah iterasi per batch GPU')
    parser.add_argument('--max-start-scalars', type=int, default=2**60, help='Maksimal jumlah start scalar yang akan dicoba')
    parser.add_argument('--kernel-mode', choices=['optimized', 'standard'], default='optimized', help='Mode kernel yang digunakan')
    parser.add_argument('--gpus', type=str, default='all', help='GPU devices to use (e.g., "0,1,2" or "all")')

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

    num_targets = len(target_bin) // 20
    print(f"[*] Loaded {num_targets} target hash160")

    # Determine which GPUs to use
    if args.gpus == 'all':
        gpu_count = cuda.Device.count()
        gpu_ids = list(range(gpu_count))
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        gpu_count = len(gpu_ids)
    
    args.gpu_count = gpu_count
    
    print(f"[*] Menggunakan {gpu_count} GPU: {gpu_ids}")

    # Load kernel code
    print("[*] Memuat dan mengkompilasi kernel CUDA...")
    try:
        with open('kernel160.cu', 'r') as f:
            kernel160_code = f.read()
    except FileNotFoundError:
        print("[!] FATAL: File 'kernel160.cu' tidak ditemukan.")
        sys.exit(1)

    # Create GPU searchers
    searchers = []
    for device_id in gpu_ids:
        searcher = GPUSearcher(device_id, kernel160_code, target_bin, target_hex_list, args)
        searchers.append(searcher)
    
    # Set cross-references for coordination
    for searcher in searchers:
        searcher.other_searchers = [s for s in searchers if s != searcher]

    print(f"\n[*] Memulai pencarian multi-GPU hash160:")
    print(f"    Start scalar awal: {hex(args.start)}")
    print(f"    Range pengali: {hex(args.range_min)} - {hex(args.range_max)}")
    print(f"    Range size: {range_size:,} kemungkinan")
    print(f"    Target hash160: {num_targets} hashes")
    print(f"    Kernel mode: {args.kernel_mode}")
    print(f"    GPU devices: {gpu_ids}")

    # Run searches in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=gpu_count) as executor:
        futures = [executor.submit(searcher.run_search) for searcher in searchers]
        
        # Wait for completion
        for future in futures:
            future.result()

    # Check results
    found_searchers = [s for s in searchers if s.found]
    
    if found_searchers:
        searcher = found_searchers[0]
        result = searcher.result
        
        print(f"\n[+] HASH160 DITEMUKAN OLEH GPU {searcher.device_id}!")
        print(f"    Private Key: {hex(result['private_key'])}")
        print(f"    Start scalar: {hex(result['start_scalar'])}")
        print(f"    Total iterasi: {result['total_iterations']:,}")
        print(f"    GPU: {result['gpu_id']}")
        print(f"    Waktu pencarian: {result['search_time']:.2f} detik")

        # Save results
        with open("found_hash160.txt", "w") as f:
            f.write(f"Private Key: {hex(result['private_key'])}\n")
            f.write(f"Start scalar: {hex(result['start_scalar'])}\n")
            f.write(f"Total iterations: {result['total_iterations']}\n")
            f.write(f"GPU: {result['gpu_id']}\n")
            f.write(f"Search time: {result['search_time']:.2f} seconds\n")
            f.write(f"Range: {hex(args.range_min)} - {hex(args.range_max)}\n")
            f.write(f"Target hashes:\n")
            for hash_hex in target_hex_list:
                f.write(f"  {hash_hex}\n")
    else:
        total_iterations_all = sum(s.total_iterations for s in searchers)
        total_time = time.time() - start_time
        
        print(f"\n\n[+] Pencarian selesai. Tidak ditemukan.")
        print(f"    Total GPU digunakan: {gpu_count}")
        print(f"    Total iterasi: {total_iterations_all:,}")
        print(f"    Waktu pencarian: {total_time:.2f} detik")
        print(f"    Kecepatan rata-rata: {total_iterations_all/total_time:,.0f} it/s")

if __name__ == '__main__':
    main()
