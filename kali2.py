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

def init_secp256k1_constants_for_gpu(mod, gpu_id):
    """Initialize secp256k1 curve constants in GPU constant memory for specific GPU"""
    cuda.Context.set_current(mod.get_context())
    
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

def run_precomputation_for_gpu(mod, gpu_id):
    """Run GPU precomputation kernel for specific GPU"""
    cuda.Context.set_current(mod.get_context())
    precompute_kernel = mod.get_function("precompute_G_table_kernel")
    precompute_kernel(block=(1, 1, 1))
    cuda.Context.synchronize()
    print(f"[*] GPU {gpu_id}: Precomputation table selesai.")

def parse_range(range_str):
    """Parse range string in format 'start-end'"""
    range_parts = range_str.split('-')
    if len(range_parts) != 2:
        raise ValueError("Format range tidak valid. Gunakan format: start-end (contoh: 0x3d94cd60-0x3d94cd69)")
    
    range_min = int(range_parts[0], 0)
    range_max = int(range_parts[1], 0)
    
    if range_min >= range_max:
        raise ValueError("Batas bawah range harus lebih kecil dari batas atas")
    
    return range_min, range_max

class GPUWorker:
    def __init__(self, gpu_id, kernel_code, target_bin, num_targets):
        self.gpu_id = gpu_id
        self.found = False
        self.privkey_int = 0
        self.total_iterations = 0
        self.context = None
        self.mod = None
        self.find_hash_kernel = None
        self.d_target_hashes = None
        self.d_result = None
        self.d_found_flag = None
        
        # Set GPU context
        self.device = cuda.Device(gpu_id)
        self.context = self.device.make_context()
        
        try:
            # Compile kernel
            self.mod = SourceModule(kernel_code, no_extern_c=False, options=['-std=c++11', '-arch=sm_75'])
            
            # Initialize constants
            init_secp256k1_constants_for_gpu(self.mod, gpu_id)
            
            # Run precomputation
            run_precomputation_for_gpu(self.mod, gpu_id)
            
            # Get kernel function
            self.find_hash_kernel = self.mod.get_function("find_hash_kernel_optimized")
            
            # Initialize GPU memory
            self.d_target_hashes = cuda.mem_alloc(len(target_bin))
            cuda.memcpy_htod(self.d_target_hashes, np.frombuffer(target_bin, dtype=np.uint8))
            
            self.d_result = cuda.mem_alloc(32)
            self.d_found_flag = cuda.mem_alloc(4)
            cuda.memset_d32(self.d_result, 0, 8)
            cuda.memset_d32(self.d_found_flag, 0, 1)
            
            self.num_targets = num_targets
            
            print(f"[*] GPU {gpu_id}: Inisialisasi selesai")
            
        except Exception as e:
            print(f"[!] GPU {gpu_id}: Error during initialization: {e}")
            if self.context:
                self.context.pop()
            raise
    
    def search_range(self, start_scalar_np, range_min_np, range_start, range_end, keys_per_launch):
        """Search in specific range on this GPU"""
        if self.found:
            return True
            
        cuda.Context.set_current(self.context)
        
        range_size = range_end - range_start
        iteration_offset = 0
        found_flag_host = np.zeros(1, dtype=np.int32)
        
        while iteration_offset < range_size and not self.found:
            iterations_left = range_size - iteration_offset
            iterations_this_launch = min(keys_per_launch, iterations_left)
            
            if iterations_this_launch <= 0:
                break
            
            block_size = 256
            grid_size = (iterations_this_launch + block_size - 1) // block_size
            
            # Reset found flag for this launch
            cuda.memset_d32(self.d_found_flag, 0, 1)
            
            # Run kernel
            self.find_hash_kernel(
                cuda.In(start_scalar_np),
                np.uint64(iterations_this_launch),
                cuda.In(range_min_np),
                np.uint64(range_start + iteration_offset),
                self.d_target_hashes,
                np.int32(self.num_targets),
                self.d_result,
                self.d_found_flag,
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
            
            self.context.synchronize()
            
            iteration_offset += iterations_this_launch
            self.total_iterations += iterations_this_launch
            
            # Check if found
            cuda.memcpy_dtoh(found_flag_host, self.d_found_flag)
            if found_flag_host[0] == 1:
                # Read result
                result_buffer = np.zeros(8, dtype=np.uint32)
                cuda.memcpy_dtoh(result_buffer, self.d_result)
                self.privkey_int = bigint_np_to_int(result_buffer)
                self.found = True
                return True
        
        return False
    
    def cleanup(self):
        """Clean up GPU context"""
        if self.context:
            self.context.pop()

def worker_thread(gpu_worker, start_scalar_np, range_min_np, range_start, range_end, keys_per_launch, result_queue):
    """Worker thread function for GPU"""
    try:
        found = gpu_worker.search_range(start_scalar_np, range_min_np, range_start, range_end, keys_per_launch)
        if found:
            result_queue.put((gpu_worker.gpu_id, gpu_worker.privkey_int, gpu_worker.total_iterations))
        else:
            result_queue.put((gpu_worker.gpu_id, None, gpu_worker.total_iterations))
    except Exception as e:
        print(f"[!] GPU {gpu_worker.gpu_id}: Error in worker thread: {e}")
        result_queue.put((gpu_worker.gpu_id, None, gpu_worker.total_iterations))

def main():
    parser = argparse.ArgumentParser(description='CUDA Hash160 Search dengan Multi-GPU Support')
    parser.add_argument('--start', type=lambda x: int(x, 0), required=True, help='Skalar awal perkalian')
    parser.add_argument('--range', type=str, required=True, help='Range pengali (format: start-end, contoh: 0x3d94cd60-0x3d94cd69)')
    parser.add_argument('--file', required=True, help='File target hash160 (hexadecimal, 40 karakter)')
    parser.add_argument('--keys-per-launch', type=int, default=2**18, help='Jumlah iterasi per batch GPU')
    parser.add_argument('--gpus', type=str, default='all', help='GPU IDs to use (contoh: 0,1,2 atau "all")')

    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpus.lower() == 'all':
        num_gpus = cuda.Device.count()
        gpu_ids = list(range(num_gpus))
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        num_gpus = len(gpu_ids)
    
    if num_gpus == 0:
        print("[!] ERROR: Tidak ada GPU yang tersedia")
        sys.exit(1)
    
    print(f"[*] Menggunakan {num_gpus} GPU: {gpu_ids}")

    # Parse range
    try:
        range_min, range_max = parse_range(args.range)
    except ValueError as e:
        print(f"[!] ERROR: {e}")
        sys.exit(1)

    range_size = range_max - range_min + 1
    print(f"[*] Range size: {range_size:,} kemungkinan multiplier")

    # Load target hashes
    print(f"[*] Memuat target hash160 dari {args.file}...")
    target_bin, target_hex_list = load_target_hashes(args.file)
    if target_bin is None:
        sys.exit(1)

    num_targets = len(target_bin) // 20
    print(f"[*] Loaded {num_targets} target hash160")

    # Load kernel code
    print("[*] Memuat kernel CUDA...")
    try:
        with open('kernel160.cu', 'r') as f:
            kernel_code = f.read()
    except FileNotFoundError:
        print("[!] FATAL: File 'kernel160.cu' tidak ditemukan.")
        sys.exit(1)

    # Initialize GPU workers
    gpu_workers = []
    try:
        for gpu_id in gpu_ids:
            print(f"[*] Menginisialisasi GPU {gpu_id}...")
            worker = GPUWorker(gpu_id, kernel_code, target_bin, num_targets)
            gpu_workers.append(worker)
    except Exception as e:
        print(f"[!] Error inisialisasi GPU: {e}")
        for worker in gpu_workers:
            worker.cleanup()
        sys.exit(1)

    # MAIN MULTI-GPU SEARCH LOOP
    total_iterations_all = 0
    start_time = time.time()
    current_start = args.start
    found = False
    range_min_np = int_to_bigint_np(range_min)

    print(f"\n[*] Memulai pencarian multi-GPU hash160:")
    print(f"    Start scalar awal: {hex(args.start)}")
    print(f"    Range pengali: {hex(range_min)} - {hex(range_max)}")
    print(f"    Range size: {range_size:,} kemungkinan per start scalar")
    print(f"    Target hash160: {num_targets} hashes")
    print(f"    GPU yang digunakan: {gpu_ids}")

    try:
        while current_start >= 1 and not found:
            print(f"\n[*] Mencoba dengan start scalar: {hex(current_start)}")
            
            start_scalar_np = int_to_bigint_np(current_start)
            
            # Divide range among GPUs
            chunk_size = range_size // num_gpus
            ranges = []
            for i in range(num_gpus):
                range_start = range_min + i * chunk_size
                range_end = range_start + chunk_size - 1 if i < num_gpus - 1 else range_max
                ranges.append((range_start, range_end))
            
            # Reset worker states
            for worker in gpu_workers:
                worker.found = False
                worker.total_iterations = 0
            
            # Launch worker threads
            threads = []
            result_queue = Queue()
            
            for i, worker in enumerate(gpu_workers):
                range_start, range_end = ranges[i]
                thread = threading.Thread(
                    target=worker_thread,
                    args=(worker, start_scalar_np, range_min_np, range_start, range_end, args.keys_per_launch, result_queue)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads and collect results
            found_gpu_id = None
            found_privkey = None
            gpu_iterations = {}
            
            for thread in threads:
                thread.join()
            
            # Process results
            while not result_queue.empty():
                gpu_id, privkey, iterations = result_queue.get()
                gpu_iterations[gpu_id] = iterations
                total_iterations_all += iterations
                
                if privkey is not None:
                    found_gpu_id = gpu_id
                    found_privkey = privkey
                    found = True
            
            # Display progress
            elapsed = time.time() - start_time
            speed = total_iterations_all / elapsed if elapsed > 0 else 0
            
            progress_str = (f"[+] Start: {hex(current_start)} | "
                          f"Total iterasi: {total_iterations_all:,} | "
                          f"Speed: {speed:,.0f} it/s | "
                          f"Running: {elapsed:.0f}s")
            sys.stdout.write('\r' + progress_str.ljust(120))
            sys.stdout.flush()
            
            # Check if found
            if found:
                sys.stdout.write('\n')
                print(f"\n[+] HASH160 DITEMUKAN di GPU {found_gpu_id}!")
                print(f"    Private Key: {hex(found_privkey)}")
                print(f"    Start scalar: {hex(current_start)}")
                print(f"    Total iterasi: {total_iterations_all:,}")
                print(f"    Waktu pencarian: {time.time() - start_time:.2f} detik")
                
                # Verifikasi dengan perhitungan ulang
                n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
                expected_privkey = (current_start) % n
                if expected_privkey == found_privkey:
                    print(f"    [VERIFIED] Private key valid")
                else:
                    print(f"    [WARNING] Private key tidak sesuai")
                
                # Simpan hasil
                with open("found_hash160.txt", "w") as f:
                    f.write(f"Private Key: {hex(found_privkey)}\n")
                    f.write(f"Start scalar: {hex(current_start)}\n")
                    f.write(f"Total iterations: {total_iterations_all}\n")
                    f.write(f"Search time: {time.time() - start_time:.2f} seconds\n")
                    f.write(f"Range: {hex(range_min)} - {hex(range_max)}\n")
                    f.write(f"Target hashes:\n")
                    for hash_hex in target_hex_list:
                        f.write(f"  {hash_hex}\n")
            else:
                # Tidak ditemukan di start scalar ini, lanjut ke berikutnya
                print(f"\n[*] Tidak ditemukan di start scalar {hex(current_start)}, melanjutkan ke {hex(current_start-1)}")
                current_start -= 1

        # Handle final results
        if not found:
            print(f"\n\n[+] Pencarian selesai. Tidak ditemukan.")
            print(f"    Total iterasi: {total_iterations_all:,}")
            print(f"    Start scalar terakhir: {hex(current_start)}")
            print(f"    Waktu pencarian: {time.time() - start_time:.2f} detik")

    except KeyboardInterrupt:
        print(f"\n\n[!] Dihentikan oleh pengguna.")
        print(f"    Start scalar saat ini: {hex(current_start)}")
        print(f"    Total iterasi: {total_iterations_all:,}")
        print(f"    Waktu pencarian: {time.time() - start_time:.2f} detik")
    except Exception as e:
        print(f"\n\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        for worker in gpu_workers:
            worker.cleanup()

if __name__ == '__main__':
    main()
