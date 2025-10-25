import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import sys
import argparse
import struct
import random

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

def init_secp256k1_constants(mod):
    """Initialize secp256k1 curve constants in GPU constant memory"""
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
    print("[*] Precomputation table selesai.")

def main():
    parser = argparse.ArgumentParser(description='CUDA Hash160 Search dengan Sequential Scalar Multiplication')
    parser.add_argument('--start', type=lambda x: int(x, 0), required=True, help='Skalar awal perkalian')
    parser.add_argument('--range-min', type=lambda x: int(x, 0), required=True, help='Batas bawah range pengali')
    parser.add_argument('--range-max', type=lambda x: int(x, 0), required=True, help='Batas atas range pengali')
    parser.add_argument('--file', required=True, help='File target hash160 (hexadecimal, 40 karakter)')
    parser.add_argument('--keys-per-launch', type=int, default=2**20, help='Jumlah iterasi per batch GPU')

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

    print("[*] Memuat dan mengkompilasi kernel CUDA...")
    try:
        # Baca kernel160.cu
        with open('kernel160.cu', 'r') as f:
            kernel160_code = f.read()

        full_cuda_code = kernel160_code

    except FileNotFoundError:
        print("[!] FATAL: File 'kernel160.cu' tidak ditemukan.")
        sys.exit(1)

    mod = SourceModule(full_cuda_code, no_extern_c=False, options=['-std=c++11', '-arch=sm_75'])
    init_secp256k1_constants(mod)

    print("[*] Menjalankan precomputation...")
    run_precomputation(mod)

    # Gunakan kernel optimized
    find_hash_kernel = mod.get_function("find_hash_kernel_optimized")
    print("[*] Menggunakan kernel optimized dengan precomputation")

    # Inisialisasi memory GPU
    d_target_hashes = cuda.mem_alloc(len(target_bin))
    cuda.memcpy_htod(d_target_hashes, np.frombuffer(target_bin, dtype=np.uint8))

    # Allocate 32 bytes untuk result (private key saja)
    d_result = cuda.mem_alloc(32)
    d_found_flag = cuda.mem_alloc(4)
    cuda.memset_d32(d_result, 0, 8)
    cuda.memset_d32(d_found_flag, 0, 1)

    # MAIN LOOP UNTUK PENCARIAN SEQUENTIAL DENGAN AUTO-LOOP START SCALAR
    total_iterations_all = 0
    start_time = time.time()
    found_flag_host = np.zeros(1, dtype=np.int32)

    range_min_np = int_to_bigint_np(args.range_min)
    range_max_np = int_to_bigint_np(args.range_max)

    current_start = args.start
    step_np = int_to_bigint_np(1)
    found = False

    print(f"\n[*] Memulai pencarian sequential hash160:")
    print(f"    Start scalar awal: {hex(args.start)}")
    print(f"    Range pengali: {hex(args.range_min)} - {hex(args.range_max)}")
    print(f"    Range size: {range_size:,} kemungkinan per start scalar")
    print(f"    Target hash160: {num_targets} hashes")

    try:
        while current_start >= 1 and not found:
            # Reset found flag untuk start scalar baru
            cuda.memset_d32(d_found_flag, 0, 1)
            found_flag_host[0] = 0

            start_scalar_np = int_to_bigint_np(current_start)
            iteration_offset = 0
            total_iterations_current = 0

            print(f"\n[*] Mencoba dengan start scalar: {hex(current_start)}")

            # Loop untuk range saat ini
            while iteration_offset < range_size and found_flag_host[0] == 0:
                iterations_left = range_size - iteration_offset
                iterations_this_launch = min(args.keys_per_launch, iterations_left)

                if iterations_this_launch <= 0:
                    break

                block_size = 256
                grid_size = (iterations_this_launch + block_size - 1) // block_size

                # Jalankan kernel pencarian hash160
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
                total_iterations_all += iterations_this_launch
                iteration_offset += iterations_this_launch

                cuda.memcpy_dtoh(found_flag_host, d_found_flag)

                elapsed = time.time() - start_time
                speed = total_iterations_all / elapsed if elapsed > 0 else 0
                progress_current = 100 * iteration_offset / range_size

                progress_str = (f"[+] Start: {hex(current_start)} | "
                              f"Progress: {iteration_offset:,}/{range_size:,} ({progress_current:.1f}%) | "
                              f"Speed: {speed:,.0f} it/s | "
                              f"Running: {elapsed:.0f}s")
                sys.stdout.write('\r' + progress_str.ljust(120))
                sys.stdout.flush()

            # Cek hasil untuk start scalar ini
            if found_flag_host[0] == 1:
                found = True
                sys.stdout.write('\n')

                # Baca hasil dari GPU
                result_buffer = np.zeros(8, dtype=np.uint32)
                cuda.memcpy_dtoh(result_buffer, d_result)

                # Extract private key
                found_privkey_np = result_buffer
                privkey_int = bigint_np_to_int(found_privkey_np)

                print(f"\n[+] HASH160 DITEMUKAN!")
                print(f"    Private Key: {hex(privkey_int)}")
                print(f"    Start scalar: {hex(current_start)}")
                print(f"    Total iterasi: {total_iterations_all:,}")
                print(f"    Waktu pencarian: {time.time() - start_time:.2f} detik")

                # Verifikasi dengan perhitungan ulang
                n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
                expected_privkey = (current_start) % n
                if expected_privkey == privkey_int:
                    print(f"    [VERIFIED] Private key valid")
                else:
                    print(f"    [WARNING] Private key tidak sesuai")

                # Simpan hasil
                with open("found_hash160.txt", "w") as f:
                    f.write(f"Private Key: {hex(privkey_int)}\n")
                    f.write(f"Start scalar: {hex(current_start)}\n")
                    f.write(f"Total iterations: {total_iterations_all}\n")
                    f.write(f"Search time: {time.time() - start_time:.2f} seconds\n")
                    f.write(f"Range: {hex(args.range_min)} - {hex(args.range_max)}\n")
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

if __name__ == '__main__':
    main()
