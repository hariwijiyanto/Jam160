#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#define BIGINT_WORDS 8

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

struct ECPoint {
    BigInt x, y;
    bool infinity;
};

struct ECPointJac {
    BigInt X, Y, Z;
    bool infinity;
};

struct CompressedPubKey {
    uint8_t data[33];
};

struct DPEntry {
    uint64_t fp;
    uint64_t reduction_factor;
};

__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ BigInt const_n;
__device__ ECPointJac const_G_table[256];
__constant__ ECPointJac const_target_jacobian;

// Fungsi utilitas BigInt
__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__host__ __device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}

__host__ __device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__host__ __device__ __forceinline__ bool is_zero(const BigInt *a) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        if (a->data[i]) return false;
    }
    return true;
}

__host__ __device__ __forceinline__ int get_bit(const BigInt *a, int i) {
    int word_idx = i >> 5;
    int bit_idx = i & 31;
    if (word_idx >= BIGINT_WORDS) return 0;
    return (a->data[word_idx] >> bit_idx) & 1;
}

__host__ __device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t sum = (uint64_t)a->data[i] + b->data[i] + carry;
        res->data[i] = (uint32_t)sum;
        carry = (sum >> 32);
    }
}

__host__ __device__ __forceinline__ void ptx_u256Sub(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t borrow = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t tmp = (uint64_t)a->data[i] - borrow;
        borrow = tmp < b->data[i] ? 1u : 0u;
        res->data[i] = (uint32_t)(tmp - b->data[i]);
    }
}

__host__ __device__ __forceinline__ void bigint_mul_uint32(BigInt *res, const BigInt *a, uint32_t b_val) {
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t product = (uint64_t)a->data[i] * b_val + carry;
        res->data[i] = (uint32_t)product;
        carry = product >> 32;
    }
}

__device__ __forceinline__ void multiply_bigint_by_const(const BigInt *a, uint32_t c, uint32_t result[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t prod = (uint64_t)a->data[i] * c + carry;
        result[i] = (uint32_t)prod;
        carry = prod >> 32;
    }
    result[8] = (uint32_t)carry;
}

__device__ __forceinline__ void shift_left_word(const BigInt *a, uint32_t result[9]) {
    result[0] = 0;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        result[i+1] = a->data[i];
    }
}

__device__ __forceinline__ void add_9word(uint32_t r[9], const uint32_t addend[9]) {
    uint64_t carry = 0;
    for (int i = 0; i < 9; i++) {
        uint64_t sum = (uint64_t)r[i] + addend[i] + carry;
        r[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__device__ __forceinline__ void convert_9word_to_bigint(const uint32_t r[9], BigInt *res) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        res->data[i] = r[i];
    }
}

__device__ __forceinline__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    uint32_t prod[2 * BIGINT_WORDS] = {0};
    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < BIGINT_WORDS; j++) {
            uint64_t tmp = (uint64_t)prod[i + j] + (uint64_t)a->data[i] * b->data[j] + carry;
            prod[i + j] = (uint32_t)tmp;
            carry = tmp >> 32;
        }
        prod[i + BIGINT_WORDS] += (uint32_t)carry;
    }

    BigInt L, H;
    for (int i = 0; i < BIGINT_WORDS; i++) {
        L.data[i] = prod[i];
        H.data[i] = prod[i + BIGINT_WORDS];
    }

    uint32_t Rext[9] = {0};
    for (int i = 0; i < BIGINT_WORDS; i++) Rext[i] = L.data[i];
    Rext[8] = 0;

    uint32_t H977[9] = {0};
    multiply_bigint_by_const(&H, 977, H977);
    add_9word(Rext, H977);

    uint32_t Hshift[9] = {0};
    shift_left_word(&H, Hshift);
    add_9word(Rext, Hshift);

    if (Rext[8]) {
        uint32_t extra[9] = {0};
        BigInt extraBI;
        init_bigint(&extraBI, Rext[8]);
        Rext[8] = 0;

        uint32_t extra977[9] = {0}, extraShift[9] = {0};
        multiply_bigint_by_const(&extraBI, 977, extra977);
        shift_left_word(&extraBI, extraShift);

        for (int i = 0; i < 9; i++) extra[i] = extra977[i];
        add_9word(extra, extraShift);
        add_9word(Rext, extra);
    }

    BigInt R_temp;
    convert_9word_to_bigint(Rext, &R_temp);

    if (Rext[8] || compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }
    if (compare_bigint(&R_temp, &const_p) >= 0) {
        ptx_u256Sub(&R_temp, &R_temp, &const_p);
    }

    copy_bigint(res, &R_temp);
}

__device__ __forceinline__ void sub_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt temp;
    if (compare_bigint(a, b) < 0) {
         BigInt sum;
         ptx_u256Add(&sum, a, &const_p);
         ptx_u256Sub(&temp, &sum, b);
    } else {
         ptx_u256Sub(&temp, a, b);
    }
    copy_bigint(res, &temp);
}

__device__ __forceinline__ void scalar_mod_n(BigInt *res, const BigInt *a) {
    if (compare_bigint(a, &const_n) >= 0) {
        ptx_u256Sub(res, a, &const_n);
    } else {
        copy_bigint(res, a);
    }
}

__device__ __forceinline__ void add_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt sum_ab;
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
         uint64_t word_sum = (uint64_t)a->data[i] + b->data[i] + carry;
         sum_ab.data[i] = (uint32_t)word_sum;
         carry = word_sum >> 32;
    }
    if (carry || compare_bigint(&sum_ab, &const_p) >= 0) {
        ptx_u256Sub(res, &sum_ab, &const_p);
    } else {
        copy_bigint(res, &sum_ab);
    }
}

__device__ void modexp(BigInt *res, const BigInt *base, const BigInt *exp) {
    BigInt result;
    init_bigint(&result, 1);
    BigInt b;
    copy_bigint(&b, base);
    for (int i = 0; i < 256; i++) {
         if (get_bit(exp, i)) {
              mul_mod_device(&result, &result, &b);
         }
         mul_mod_device(&b, &b, &b);
    }
    copy_bigint(res, &result);
}

__device__ void mod_inverse(BigInt *res, const BigInt *a) {
    if (is_zero(a)) {
        init_bigint(res, 0);
        return;
    }
    BigInt p_minus_2, two;
    init_bigint(&two, 2);
    ptx_u256Sub(&p_minus_2, &const_p, &two);
    modexp(res, a, &p_minus_2);
}

__device__ __forceinline__ void point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
}

__device__ __forceinline__ void point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    copy_bigint(&dest->X, &src->X);
    copy_bigint(&dest->Y, &src->Y);
    copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}

__device__ __forceinline__ void negate_point_jac(ECPointJac *R, const ECPointJac *P) {
    point_copy_jac(R, P);
    sub_mod_device(&R->Y, &const_p, &P->Y);
}

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P) {
    if (P->infinity || is_zero(&P->Y)) {
        point_set_infinity_jac(R);
        return;
    }
    BigInt A, B, C, D, X3, Y3, Z3, temp, temp2;
    mul_mod_device(&A, &P->Y, &P->Y);
    mul_mod_device(&temp, &P->X, &A);
    init_bigint(&temp2, 4);
    mul_mod_device(&B, &temp, &temp2);
    mul_mod_device(&temp, &A, &A);
    init_bigint(&temp2, 8);
    mul_mod_device(&C, &temp, &temp2);
    mul_mod_device(&temp, &P->X, &P->X);
    init_bigint(&temp2, 3);
    mul_mod_device(&D, &temp, &temp2);
    BigInt D2, two, twoB;
    mul_mod_device(&D2, &D, &D);
    init_bigint(&two, 2);
    mul_mod_device(&twoB, &B, &two);
    sub_mod_device(&X3, &D2, &twoB);
    sub_mod_device(&temp, &B, &X3);
    mul_mod_device(&temp, &D, &temp);
    sub_mod_device(&Y3, &temp, &C);
    init_bigint(&temp, 2);
    mul_mod_device(&temp, &temp, &P->Y);
    mul_mod_device(&Z3, &temp, &P->Z);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    if (P->infinity) { point_copy_jac(R, Q); return; }
    if (Q->infinity) { point_copy_jac(R, P); return; }

    BigInt Z1Z1, Z2Z2, U1, U2, S1, S2, H, R_big, H2, H3, U1H2, X3, Y3, Z3, temp;
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&Z2Z2, &Q->Z, &Q->Z);
    mul_mod_device(&U1, &P->X, &Z2Z2);
    mul_mod_device(&U2, &Q->X, &Z1Z1);
    BigInt Z2_cubed, Z1_cubed;
    mul_mod_device(&temp, &Z2Z2, &Q->Z); copy_bigint(&Z2_cubed, &temp);
    mul_mod_device(&temp, &Z1Z1, &P->Z); copy_bigint(&Z1_cubed, &temp);
    mul_mod_device(&S1, &P->Y, &Z2_cubed);
    mul_mod_device(&S2, &Q->Y, &Z1_cubed);

    if (compare_bigint(&U1, &U2) == 0) {
        if (compare_bigint(&S1, &S2) != 0) {
            point_set_infinity_jac(R);
            return;
        } else {
            double_point_jac(R, P);
            return;
        }
    }
    sub_mod_device(&H, &U2, &U1);
    sub_mod_device(&R_big, &S2, &S1);
    mul_mod_device(&H2, &H, &H);
    mul_mod_device(&H3, &H2, &H);
    mul_mod_device(&U1H2, &U1, &H2);
    BigInt R2, two, twoU1H2;
    mul_mod_device(&R2, &R_big, &R_big);
    init_bigint(&two, 2);
    mul_mod_device(&twoU1H2, &U1H2, &two);
    sub_mod_device(&temp, &R2, &H3);
    sub_mod_device(&X3, &temp, &twoU1H2);
    sub_mod_device(&temp, &U1H2, &X3);
    mul_mod_device(&temp, &R_big, &temp);
    mul_mod_device(&Y3, &S1, &H3);
    sub_mod_device(&Y3, &temp, &Y3);
    mul_mod_device(&temp, &P->Z, &Q->Z);
    mul_mod_device(&Z3, &temp, &H);
    copy_bigint(&R->X, &X3);
    copy_bigint(&R->Y, &Y3);
    copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void jacobian_to_affine(ECPoint *R, const ECPointJac *P) {
    if (P->infinity) {
        R->infinity = true;
        init_bigint(&R->x, 0);
        init_bigint(&R->y, 0);
        return;
    }
    BigInt Zinv, Zinv2, Zinv3;
    mod_inverse(&Zinv, &P->Z);
    mul_mod_device(&Zinv2, &Zinv, &Zinv);
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);
    mul_mod_device(&R->x, &P->X, &Zinv2);
    mul_mod_device(&R->y, &P->Y, &Zinv3);
    R->infinity = false;
}

// PRECOMPUTATION FUNCTIONS
__global__ void precompute_G_table_kernel() {
    int idx = threadIdx.x;
    if (idx == 0) {
        ECPointJac current = const_G_jacobian;
        point_copy_jac(&const_G_table[0], &current);

        for (int i = 1; i < 256; i++) {
            double_point_jac(&current, &current);
            point_copy_jac(&const_G_table[i], &current);
        }
    }
}

__device__ void scalar_multiply_jac_precomputed(ECPointJac *result, const BigInt *scalar) {
    point_set_infinity_jac(result);

    for (int i = 0; i < 256; i++) {
        if (get_bit(scalar, i)) {
            add_point_jac(result, result, &const_G_table[i]);
        }
    }
}

// FUNGSI KOMPRESI PUBKEY
__device__ void compress_pubkey(CompressedPubKey *compressed, const ECPoint *pubkey) {
    if (pubkey->infinity) {
        memset(compressed->data, 0, 33);
        return;
    }

    uint8_t prefix = (pubkey->y.data[0] & 1) ? 0x03 : 0x02;
    compressed->data[0] = prefix;

    for (int i = 0; i < BIGINT_WORDS; i++) {
        uint32_t word = pubkey->x.data[BIGINT_WORDS - 1 - i];
        compressed->data[1 + i*4] = (word >> 24) & 0xFF;
        compressed->data[2 + i*4] = (word >> 16) & 0xFF;
        compressed->data[3 + i*4] = (word >> 8) & 0xFF;
        compressed->data[4 + i*4] = word & 0xFF;
    }
}

// FUNGSI BLOOM FILTER DAN DP TABLE
__device__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ bool test_bit(const uint32_t* bloom_filter, uint64_t index) {
    uint64_t word_index = index / 32;
    uint32_t bit_index = index % 32;
    return (bloom_filter[word_index] >> bit_index) & 1;
}

__device__ bool check_bloom_filter(const uint32_t* bloom_filter, uint64_t bloom_size, uint64_t fp) {
    uint64_t h1 = splitmix64(fp);
    uint64_t h2 = splitmix64(h1);

    for (int j = 0; j < 4; j++) {
        uint64_t index = (h1 + j * h2) % bloom_size;
        if (!test_bit(bloom_filter, index)) {
            return false;
        }
    }
    return true;
}

__device__ uint64_t find_dp_factor_binary(const DPEntry* dp_table, uint32_t dp_table_size, uint64_t fp) {
    int left = 0;
    int right = dp_table_size - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        uint64_t mid_fp = dp_table[mid].fp;

        if (mid_fp == fp) {
            return dp_table[mid].reduction_factor;
        } else if (mid_fp < fp) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return 0;
}

// KERNEL UTAMA UNTUK PENCARIAN SEQUENTIAL
__device__ inline bool device_memcmp(const void *s1, const void *s2, size_t n) {
    const uint8_t *p1 = (const uint8_t *)s1;
    const uint8_t *p2 = (const uint8_t *)s2;
    for (size_t i = 0; i < n; ++i) {
        if (p1[i] != p2[i]) return false;
    }
    return true;
}

// KERNEL SEQUENTIAL DENGAN DP
extern "C"
__global__ void find_pubkey_kernel_sequential(
    const BigInt* start_scalar, const BigInt* step_size, uint64_t max_iterations,
    const uint8_t* d_target_pubkeys, int num_targets,
    BigInt* d_result, int* d_found_flag,
    const uint32_t* d_bloom_filter, uint64_t bloom_size,
    const DPEntry* d_dp_table, uint32_t dp_table_size
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_iterations || *d_found_flag) return;

    // 1. HITUNG SKALAR UNTUK ITERASI INI
    BigInt current_scalar;
    BigInt step_times_idx;
    bigint_mul_uint32(&step_times_idx, step_size, (uint32_t)idx);
    ptx_u256Add(&current_scalar, start_scalar, &step_times_idx);
    scalar_mod_n(&current_scalar, &current_scalar);

    // 2. HITUNG PUBLIC KEY: current_scalar * G
    ECPointJac result_jac;
    scalar_multiply_jac_precomputed(&result_jac, &current_scalar);

    // 3. KONVERSI KE AFFINE
    ECPoint public_key;
    jacobian_to_affine(&public_key, &result_jac);
    if (public_key.infinity) return;

    // 4. CEK DP TERLEBIH DAHULU
    uint64_t low64_x = public_key.x.data[0] | ((uint64_t)public_key.x.data[1] << 32);
    uint64_t y_parity = (public_key.y.data[0] & 1) ? 1 : 0;
    uint64_t fp_val = splitmix64(low64_x ^ y_parity);

    if (check_bloom_filter(d_bloom_filter, bloom_size, fp_val)) {
        uint64_t reduction_factor = find_dp_factor_binary(d_dp_table, dp_table_size, fp_val);
        if (reduction_factor > 0) {
            // HITUNG PRIVATE KEY TARGET
            BigInt target_priv;
            BigInt reduction_factor_bi;
            init_bigint(&reduction_factor_bi, 0);
            for (int j = 0; j < 2; j++) {
                reduction_factor_bi.data[j] = (uint32_t)(reduction_factor >> (32 * j));
            }

            ptx_u256Add(&target_priv, &current_scalar, &reduction_factor_bi);
            scalar_mod_n(&target_priv, &target_priv);

            // VERIFIKASI
            ECPointJac verification_jac;
            scalar_multiply_jac_precomputed(&verification_jac, &target_priv);
            ECPoint verification_affine;
            jacobian_to_affine(&verification_affine, &verification_jac);

            CompressedPubKey compressed_verification;
            compress_pubkey(&compressed_verification, &verification_affine);

            for (int i = 0; i < num_targets; i++) {
                if (device_memcmp(compressed_verification.data, &d_target_pubkeys[i * 33], 33)) {
                    if (atomicCAS(d_found_flag, 0, 1) == 0) {
                        copy_bigint(d_result, &target_priv);
                    }
                    return;
                }
            }
        }
    }

    // 5. CEK LANGSUNG DENGAN TARGET
    CompressedPubKey compressed_pubkey;
    compress_pubkey(&compressed_pubkey, &public_key);

    for (int i = 0; i < num_targets; i++) {
        if (device_memcmp(compressed_pubkey.data, &d_target_pubkeys[i * 33], 33)) {
            if (atomicCAS(d_found_flag, 0, 1) == 0) {
                copy_bigint(d_result, &current_scalar);
            }
            return;
        }
    }
}

// KERNEL UNTUK GENERASI DP DI GPU
extern "C"
__global__ void generate_dp_table_kernel(
    DPEntry* d_dp_table,
    uint32_t* d_bloom_filter,
    uint32_t dp_table_size,
    uint64_t bloom_size,
    uint64_t reduction_step
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dp_table_size) {
        return;
    }

    // 1. HITUNG REDUCTION FACTOR UNTUK THREAD INI
    uint64_t reduction_factor_val = (uint64_t)(idx + 1) * reduction_step;
    BigInt reduction_factor_bi;
    init_bigint(&reduction_factor_bi, 0);
    for (int j = 0; j < 2; j++) {
        reduction_factor_bi.data[j] = (uint32_t)(reduction_factor_val >> (32 * j));
    }

    // 2. HITUNG reduction_factor * G
    ECPointJac reduction_g_jac;
    scalar_multiply_jac_precomputed(&reduction_g_jac, &reduction_factor_bi);

    // 3. NEGASI TITIK UNTUK PENGURANGAN
    ECPointJac negative_reduction_g_jac;
    negate_point_jac(&negative_reduction_g_jac, &reduction_g_jac);

    // 4. HITUNG TITIK DP (Target + (-reduction_factor * G))
    ECPointJac dp_jac;
    add_point_jac(&dp_jac, &const_target_jacobian, &negative_reduction_g_jac);

    // 5. KONVERSI KE AFFINE UNTUK MENGHITUNG FINGERPRINT
    ECPoint dp_affine;
    jacobian_to_affine(&dp_affine, &dp_jac);

    // 6. HITUNG FINGERPRINT
    uint64_t low64_x = dp_affine.x.data[0] | ((uint64_t)dp_affine.x.data[1] << 32);
    uint64_t y_parity = (dp_affine.y.data[0] & 1) ? 1 : 0;
    uint64_t fp = splitmix64(low64_x ^ y_parity);

    // 7. TULIS FINGERPRINT DAN REDUCTION FACTOR KE DP TABLE
    d_dp_table[idx].fp = fp;
    d_dp_table[idx].reduction_factor = reduction_factor_val;

    // 8. TULIS KE BLOOM FILTER SECARA ATOMIK
    uint64_t h1 = splitmix64(fp);
    uint64_t h2 = splitmix64(h1);

    for (int j = 0; j < 4; j++) {
        uint64_t index = (h1 + j * h2) % bloom_size;
        uint64_t word_index = index / 32;
        uint32_t bit_index = index % 32;
        atomicOr(&d_bloom_filter[word_index], (1 << bit_index));
    }
}
