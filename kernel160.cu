#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// --- Kode dari secp256k1.cuh ---
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

__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ BigInt const_n;

// Precomputation table untuk 2^i * G (for i=0 to 255)
__device__ ECPointJac const_G_table[256];

__host__ __device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__host__ __device__ __forceinline__ void init_bigint_from_u64(BigInt *a, uint64_t val) {
    a->data[0] = (uint32_t)(val & 0xFFFFFFFF);
    a->data[1] = (uint32_t)(val >> 32);
    for (int i = 2; i < BIGINT_WORDS; i++) {
        a->data[i] = 0;
    }
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

__host__ __device__ __forceinline__ void add_scalar_to_bigint(BigInt *res, const BigInt *a, unsigned long long scalar) {
    uint64_t carry = scalar;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t sum = (uint64_t)a->data[i] + (carry & 0xFFFFFFFFULL);
        res->data[i] = (uint32_t)sum;
        carry = (sum >> 32) + (carry >> 32);
    }
}

__host__ __device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    uint64_t carry = 0;
    for (int i = 0; i < BIGINT_WORDS; ++i) {
        uint64_t sum = (uint64_t)a->data[i] + b->data[i] + carry;
        res->data[i] = (uint32_t)sum;
        carry = (sum >> 32);
    }
}

__host__ __device__ __forceinline__ void bigint_add(BigInt *res, const BigInt *a, const BigInt *b) {
    ptx_u256Add(res, a, b);
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

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P);
__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q);

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

__device__ void scalar_multiply_jac_device(ECPointJac *result, const ECPointJac *point, const BigInt *scalar) {
    ECPointJac res;
    point_set_infinity_jac(&res);

    int highest_bit = BIGINT_WORDS * 32 - 1;
    for (; highest_bit >= 0; highest_bit--) {
        if (get_bit(scalar, highest_bit)) break;
    }

    if (highest_bit < 0) {
        point_copy_jac(result, &res);
        return;
    }

    ECPointJac p_copy;
    point_copy_jac(&p_copy, point);

    for (int i = highest_bit; i >= 0; i--) {
        double_point_jac(&res, &res);
        if (get_bit(scalar, i)) {
            add_point_jac(&res, &res, &p_copy);
        }
    }
    point_copy_jac(result, &res);
}

// ===================================================================
// PRECOMPUTATION FUNCTIONS
// ===================================================================
extern "C"
__global__ void precompute_G_table_kernel() {
    int idx = threadIdx.x;
    if (idx == 0) {
        ECPointJac current = const_G_jacobian;
        point_copy_jac(&const_G_table[0], &current);  // Simpan G (2^0 * G)

        // Precompute 2^i * G untuk i=1 sampai 255
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

// ===================================================================
// KERNEL UTAMA UNTUK PENCARIAN HASH160
// ===================================================================

// --- Kode dari GPUHash.h ---

__device__ __constant__ uint32_t K[] =
{
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
};

__device__ __constant__ uint32_t I[] = {
  0x6a09e667ul,
  0xbb67ae85ul,
  0x3c6ef372ul,
  0xa54ff53aul,
  0x510e527ful,
  0x9b05688cul,
  0x1f83d9abul,
  0x5be0cd19ul,
};

#define ROR(x,n) ((x>>n)|(x<<(32-n)))
#define S0(x) (ROR(x,2) ^ ROR(x,13) ^ ROR(x,22))
#define S1(x) (ROR(x,6) ^ ROR(x,11) ^ ROR(x,25))
#define s0(x) (ROR(x,7) ^ ROR(x,18) ^ (x >> 3))
#define s1(x) (ROR(x,17) ^ ROR(x,19) ^ (x >> 10))

#define Maj(x,y,z) ((x & y) | (z & (x | y)))
#define Ch(x,y,z) (z ^ (x & (y ^ z)))

// SHA-256 inner round
#define S2Round(a, b, c, d, e, f, g, h, k, w) \
    t1 = h + S1(e) + Ch(e,f,g) + k + (w); \
    t2 = S0(a) + Maj(a,b,c); \
    d += t1; \
    h = t1 + t2;

// WMIX
#define WMIX() { \
w[0] += s1(w[14]) + w[9] + s0(w[1]);\
w[1] += s1(w[15]) + w[10] + s0(w[2]);\
w[2] += s1(w[0]) + w[11] + s0(w[3]);\
w[3] += s1(w[1]) + w[12] + s0(w[4]);\
w[4] += s1(w[2]) + w[13] + s0(w[5]);\
w[5] += s1(w[3]) + w[14] + s0(w[6]);\
w[6] += s1(w[4]) + w[15] + s0(w[7]);\
w[7] += s1(w[5]) + w[0] + s0(w[8]);\
w[8] += s1(w[6]) + w[1] + s0(w[9]);\
w[9] += s1(w[7]) + w[2] + s0(w[10]);\
w[10] += s1(w[8]) + w[3] + s0(w[11]);\
w[11] += s1(w[9]) + w[4] + s0(w[12]);\
w[12] += s1(w[10]) + w[5] + s0(w[13]);\
w[13] += s1(w[11]) + w[6] + s0(w[14]);\
w[14] += s1(w[12]) + w[7] + s0(w[15]);\
w[15] += s1(w[13]) + w[8] + s0(w[0]);\
}

// ROUND
#define SHA256_RND(k) {\
S2Round(a, b, c, d, e, f, g, h, K[k], w[0]);\
S2Round(h, a, b, c, d, e, f, g, K[k + 1], w[1]);\
S2Round(g, h, a, b, c, d, e, f, K[k + 2], w[2]);\
S2Round(f, g, h, a, b, c, d, e, K[k + 3], w[3]);\
S2Round(e, f, g, h, a, b, c, d, K[k + 4], w[4]);\
S2Round(d, e, f, g, h, a, b, c, K[k + 5], w[5]);\
S2Round(c, d, e, f, g, h, a, b, K[k + 6], w[6]);\
S2Round(b, c, d, e, f, g, h, a, K[k + 7], w[7]);\
S2Round(a, b, c, d, e, f, g, h, K[k + 8], w[8]);\
S2Round(h, a, b, c, d, e, f, g, K[k + 9], w[9]);\
S2Round(g, h, a, b, c, d, e, f, K[k + 10], w[10]);\
S2Round(f, g, h, a, b, c, d, e, K[k + 11], w[11]);\
S2Round(e, f, g, h, a, b, c, d, K[k + 12], w[12]);\
S2Round(d, e, f, g, h, a, b, c, K[k + 13], w[13]);\
S2Round(c, d, e, f, g, h, a, b, K[k + 14], w[14]);\
S2Round(b, c, d, e, f, g, h, a, K[k + 15], w[15]);\
}

#define bswap32(v) __byte_perm(v, 0, 0x0123)

// Initialise state
__device__ void SHA256Initialize(uint32_t s[8]) {
#pragma unroll 8
  for (int i = 0; i < 8; i++)
    s[i] = I[i];
}

#define DEF(x,y) uint32_t x = s[y]

// Perform SHA-256 transformations, process 64-byte chunks
__device__ void SHA256Transform(uint32_t s[8],uint32_t* w) {

  uint32_t t1;
  uint32_t t2;

  DEF(a, 0);
  DEF(b, 1);
  DEF(c, 2);
  DEF(d, 3);
  DEF(e, 4);
  DEF(f, 5);
  DEF(g, 6);
  DEF(h, 7);

  SHA256_RND(0);
  WMIX();
  SHA256_RND(16);
  WMIX();
  SHA256_RND(32);
  WMIX();
  SHA256_RND(48);

  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;

}

// ---------------------------------------------------------------------------------
// RIPEMD160
// ---------------------------------------------------------------------------------
__device__ __constant__ uint64_t ripemd160_sizedesc_32 = 32 << 3;

__device__ void RIPEMD160Initialize(uint32_t s[5]) {

  s[0] = 0x67452301ul;
  s[1] = 0xEFCDAB89ul;
  s[2] = 0x98BADCFEul;
  s[3] = 0x10325476ul;
  s[4] = 0xC3D2E1F0ul;

}

#define ROL(x,n) ((x>>(32-n))|(x<<n))
#define f1(x, y, z) (x ^ y ^ z)
#define f2(x, y, z) ((x & y) | (~x & z))
#define f3(x, y, z) ((x | ~y) ^ z)
#define f4(x, y, z) ((x & z) | (~z & y))
#define f5(x, y, z) (x ^ (y | ~z))

#define RPRound(a,b,c,d,e,f,x,k,r) \
  u = a + f + x + k; \
  a = ROL(u, r) + e; \
  c = ROL(c, 10);

#define R11(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)
#define R21(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x5A827999ul, r)
#define R31(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1ul, r)
#define R41(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDCul, r)
#define R51(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4Eul, r)
#define R12(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6ul, r)
#define R22(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124ul, r)
#define R32(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3ul, r)
#define R42(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9ul, r)
#define R52(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)

/** Perform a RIPEMD-160 transformation, processing a 64-byte chunk. */
__device__ void RIPEMD160Transform(uint32_t s[5],uint32_t* w) {

  uint32_t u;
  uint32_t a1 = s[0], b1 = s[1], c1 = s[2], d1 = s[3], e1 = s[4];
  uint32_t a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;

  R11(a1, b1, c1, d1, e1, w[0], 11);
  R12(a2, b2, c2, d2, e2, w[5], 8);
  R11(e1, a1, b1, c1, d1, w[1], 14);
  R12(e2, a2, b2, c2, d2, w[14], 9);
  R11(d1, e1, a1, b1, c1, w[2], 15);
  R12(d2, e2, a2, b2, c2, w[7], 9);
  R11(c1, d1, e1, a1, b1, w[3], 12);
  R12(c2, d2, e2, a2, b2, w[0], 11);
  R11(b1, c1, d1, e1, a1, w[4], 5);
  R12(b2, c2, d2, e2, a2, w[9], 13);
  R11(a1, b1, c1, d1, e1, w[5], 8);
  R12(a2, b2, c2, d2, e2, w[2], 15);
  R11(e1, a1, b1, c1, d1, w[6], 7);
  R12(e2, a2, b2, c2, d2, w[11], 15);
  R11(d1, e1, a1, b1, c1, w[7], 9);
  R12(d2, e2, a2, b2, c2, w[4], 5);
  R11(c1, d1, e1, a1, b1, w[8], 11);
  R12(c2, d2, e2, a2, b2, w[13], 7);
  R11(b1, c1, d1, e1, a1, w[9], 13);
  R12(b2, c2, d2, e2, a2, w[6], 7);
  R11(a1, b1, c1, d1, e1, w[10], 14);
  R12(a2, b2, c2, d2, e2, w[15], 8);
  R11(e1, a1, b1, c1, d1, w[11], 15);
  R12(e2, a2, b2, c2, d2, w[8], 11);
  R11(d1, e1, a1, b1, c1, w[12], 6);
  R12(d2, e2, a2, b2, c2, w[1], 14);
  R11(c1, d1, e1, a1, b1, w[13], 7);
  R12(c2, d2, e2, a2, b2, w[10], 14);
  R11(b1, c1, d1, e1, a1, w[14], 9);
  R12(b2, c2, d2, e2, a2, w[3], 12);
  R11(a1, b1, c1, d1, e1, w[15], 8);
  R12(a2, b2, c2, d2, e2, w[12], 6);

  R21(e1, a1, b1, c1, d1, w[7], 7);
  R22(e2, a2, b2, c2, d2, w[6], 9);
  R21(d1, e1, a1, b1, c1, w[4], 6);
  R22(d2, e2, a2, b2, c2, w[11], 13);
  R21(c1, d1, e1, a1, b1, w[13], 8);
  R22(c2, d2, e2, a2, b2, w[3], 15);
  R21(b1, c1, d1, e1, a1, w[1], 13);
  R22(b2, c2, d2, e2, a2, w[7], 7);
  R21(a1, b1, c1, d1, e1, w[10], 11);
  R22(a2, b2, c2, d2, e2, w[0], 12);
  R21(e1, a1, b1, c1, d1, w[6], 9);
  R22(e2, a2, b2, c2, d2, w[13], 8);
  R21(d1, e1, a1, b1, c1, w[15], 7);
  R22(d2, e2, a2, b2, c2, w[5], 9);
  R21(c1, d1, e1, a1, b1, w[3], 15);
  R22(c2, d2, e2, a2, b2, w[10], 11);
  R21(b1, c1, d1, e1, a1, w[12], 7);
  R22(b2, c2, d2, e2, a2, w[14], 7);
  R21(a1, b1, c1, d1, e1, w[0], 12);
  R22(a2, b2, c2, d2, e2, w[15], 7);
  R21(e1, a1, b1, c1, d1, w[9], 15);
  R22(e2, a2, b2, c2, d2, w[8], 12);
  R21(d1, e1, a1, b1, c1, w[5], 9);
  R22(d2, e2, a2, b2, c2, w[12], 7);
  R21(c1, d1, e1, a1, b1, w[2], 11);
  R22(c2, d2, e2, a2, b2, w[4], 6);
  R21(b1, c1, d1, e1, a1, w[14], 7);
  R22(b2, c2, d2, e2, a2, w[9], 15);
  R21(a1, b1, c1, d1, e1, w[11], 13);
  R22(a2, b2, c2, d2, e2, w[1], 13);
  R21(e1, a1, b1, c1, d1, w[8], 12);
  R22(e2, a2, b2, c2, d2, w[2], 11);

  R31(d1, e1, a1, b1, c1, w[3], 11);
  R32(d2, e2, a2, b2, c2, w[15], 9);
  R31(c1, d1, e1, a1, b1, w[10], 13);
  R32(c2, d2, e2, a2, b2, w[5], 7);
  R31(b1, c1, d1, e1, a1, w[14], 6);
  R32(b2, c2, d2, e2, a2, w[1], 15);
  R31(a1, b1, c1, d1, e1, w[4], 7);
  R32(a2, b2, c2, d2, e2, w[3], 11);
  R31(e1, a1, b1, c1, d1, w[9], 14);
  R32(e2, a2, b2, c2, d2, w[7], 8);
  R31(d1, e1, a1, b1, c1, w[15], 9);
  R32(d2, e2, a2, b2, c2, w[14], 6);
  R31(c1, d1, e1, a1, b1, w[8], 13);
  R32(c2, d2, e2, a2, b2, w[6], 6);
  R31(b1, c1, d1, e1, a1, w[1], 15);
  R32(b2, c2, d2, e2, a2, w[9], 14);
  R31(a1, b1, c1, d1, e1, w[2], 14);
  R32(a2, b2, c2, d2, e2, w[11], 12);
  R31(e1, a1, b1, c1, d1, w[7], 8);
  R32(e2, a2, b2, c2, d2, w[8], 13);
  R31(d1, e1, a1, b1, c1, w[0], 13);
  R32(d2, e2, a2, b2, c2, w[12], 5);
  R31(c1, d1, e1, a1, b1, w[6], 6);
  R32(c2, d2, e2, a2, b2, w[2], 14);
  R31(b1, c1, d1, e1, a1, w[13], 5);
  R32(b2, c2, d2, e2, a2, w[10], 13);
  R31(a1, b1, c1, d1, e1, w[11], 12);
  R32(a2, b2, c2, d2, e2, w[0], 13);
  R31(e1, a1, b1, c1, d1, w[5], 7);
  R32(e2, a2, b2, c2, d2, w[4], 7);
  R31(d1, e1, a1, b1, c1, w[12], 5);
  R32(d2, e2, a2, b2, c2, w[13], 5);

  R41(c1, d1, e1, a1, b1, w[1], 11);
  R42(c2, d2, e2, a2, b2, w[8], 15);
  R41(b1, c1, d1, e1, a1, w[9], 12);
  R42(b2, c2, d2, e2, a2, w[6], 5);
  R41(a1, b1, c1, d1, e1, w[11], 14);
  R42(a2, b2, c2, d2, e2, w[4], 8);
  R41(e1, a1, b1, c1, d1, w[10], 15);
  R42(e2, a2, b2, c2, d2, w[1], 11);
  R41(d1, e1, a1, b1, c1, w[0], 14);
  R42(d2, e2, a2, b2, c2, w[3], 14);
  R41(c1, d1, e1, a1, b1, w[8], 15);
  R42(c2, d2, e2, a2, b2, w[11], 14);
  R41(b1, c1, d1, e1, a1, w[12], 9);
  R42(b2, c2, d2, e2, a2, w[15], 6);
  R41(a1, b1, c1, d1, e1, w[4], 8);
  R42(a2, b2, c2, d2, e2, w[0], 14);
  R41(e1, a1, b1, c1, d1, w[13], 9);
  R42(e2, a2, b2, c2, d2, w[5], 6);
  R41(d1, e1, a1, b1, c1, w[3], 14);
  R42(d2, e2, a2, b2, c2, w[12], 9);
  R41(c1, d1, e1, a1, b1, w[7], 5);
  R42(c2, d2, e2, a2, b2, w[2], 12);
  R41(b1, c1, d1, e1, a1, w[15], 6);
  R42(b2, c2, d2, e2, a2, w[13], 9);
  R41(a1, b1, c1, d1, e1, w[14], 8);
  R42(a2, b2, c2, d2, e2, w[9], 12);
  R41(e1, a1, b1, c1, d1, w[5], 6);
  R42(e2, a2, b2, c2, d2, w[7], 5);
  R41(d1, e1, a1, b1, c1, w[6], 5);
  R42(d2, e2, a2, b2, c2, w[10], 15);
  R41(c1, d1, e1, a1, b1, w[2], 12);
  R42(c2, d2, e2, a2, b2, w[14], 8);

  R51(b1, c1, d1, e1, a1, w[4], 9);
  R52(b2, c2, d2, e2, a2, w[12], 8);
  R51(a1, b1, c1, d1, e1, w[0], 15);
  R52(a2, b2, c2, d2, e2, w[15], 5);
  R51(e1, a1, b1, c1, d1, w[5], 5);
  R52(e2, a2, b2, c2, d2, w[10], 12);
  R51(d1, e1, a1, b1, c1, w[9], 11);
  R52(d2, e2, a2, b2, c2, w[4], 9);
  R51(c1, d1, e1, a1, b1, w[7], 6);
  R52(c2, d2, e2, a2, b2, w[1], 12);
  R51(b1, c1, d1, e1, a1, w[12], 8);
  R52(b2, c2, d2, e2, a2, w[5], 5);
  R51(a1, b1, c1, d1, e1, w[2], 13);
  R52(a2, b2, c2, d2, e2, w[8], 14);
  R51(e1, a1, b1, c1, d1, w[10], 12);
  R52(e2, a2, b2, c2, d2, w[7], 6);
  R51(d1, e1, a1, b1, c1, w[14], 5);
  R52(d2, e2, a2, b2, c2, w[6], 8);
  R51(c1, d1, e1, a1, b1, w[1], 12);
  R52(c2, d2, e2, a2, b2, w[2], 13);
  R51(b1, c1, d1, e1, a1, w[3], 13);
  R52(b2, c2, d2, e2, a2, w[13], 6);
  R51(a1, b1, c1, d1, e1, w[8], 14);
  R52(a2, b2, c2, d2, e2, w[14], 5);
  R51(e1, a1, b1, c1, d1, w[11], 11);
  R52(e2, a2, b2, c2, d2, w[0], 15);
  R51(d1, e1, a1, b1, c1, w[6], 8);
  R52(d2, e2, a2, b2, c2, w[3], 13);
  R51(c1, d1, e1, a1, b1, w[15], 5);
  R52(c2, d2, e2, a2, b2, w[9], 11);
  R51(b1, c1, d1, e1, a1, w[13], 6);
  R52(b2, c2, d2, e2, a2, w[11], 11);

  uint32_t t = s[0];
  s[0] = s[1] + c1 + d2;
  s[1] = s[2] + d1 + e2;
  s[2] = s[3] + e1 + a2;
  s[3] = s[4] + a1 + b2;
  s[4] = t + b1 + c2;
}

// ---------------------------------------------------------------------------------
// Key encoding
// ---------------------------------------------------------------------------------

__device__ __noinline__ void _GetHash160Comp(uint64_t *x, uint8_t isOdd, uint8_t *hash) {

  uint32_t *x32 = (uint32_t *)(x);
  uint32_t publicKeyBytes[16];
  uint32_t s[16];

  // Compressed public key
  publicKeyBytes[0] = __byte_perm(x32[7], 0x2 + isOdd, 0x4321);
  publicKeyBytes[1] = __byte_perm(x32[7], x32[6], 0x0765);
  publicKeyBytes[2] = __byte_perm(x32[6], x32[5], 0x0765);
  publicKeyBytes[3] = __byte_perm(x32[5], x32[4], 0x0765);
  publicKeyBytes[4] = __byte_perm(x32[4], x32[3], 0x0765);
  publicKeyBytes[5] = __byte_perm(x32[3], x32[2], 0x0765);
  publicKeyBytes[6] = __byte_perm(x32[2], x32[1], 0x0765);
  publicKeyBytes[7] = __byte_perm(x32[1], x32[0], 0x0765);
  publicKeyBytes[8] = __byte_perm(x32[0], 0x80, 0x0456);
  publicKeyBytes[9] = 0;
  publicKeyBytes[10] = 0;
  publicKeyBytes[11] = 0;
  publicKeyBytes[12] = 0;
  publicKeyBytes[13] = 0;
  publicKeyBytes[14] = 0;
  publicKeyBytes[15] = 0x108;

  SHA256Initialize(s);
  SHA256Transform(s, publicKeyBytes);

#pragma unroll 8
  for (int i = 0; i < 8; i++)
    s[i] = bswap32(s[i]);

  *(uint64_t *)(s + 8) = 0x80ULL;
  *(uint64_t *)(s + 10) = 0ULL;
  *(uint64_t *)(s + 12) = 0ULL;
  *(uint64_t *)(s + 14) = ripemd160_sizedesc_32;

  RIPEMD160Initialize((uint32_t *)hash);
  RIPEMD160Transform((uint32_t *)hash, s);

}

__device__ __noinline__ void _GetHash160CompSym(uint64_t *x, uint8_t *h1, uint8_t *h2) {

  uint32_t *x32 = (uint32_t *)(x);
  uint32_t publicKeyBytes[16];
  uint32_t publicKeyBytes2[16];
  uint32_t s[16];

  // Compressed public key

  // Even
  publicKeyBytes[0] = __byte_perm(x32[7], 0x2, 0x4321);
  publicKeyBytes[1] = __byte_perm(x32[7], x32[6], 0x0765);
  publicKeyBytes[2] = __byte_perm(x32[6], x32[5], 0x0765);
  publicKeyBytes[3] = __byte_perm(x32[5], x32[4], 0x0765);
  publicKeyBytes[4] = __byte_perm(x32[4], x32[3], 0x0765);
  publicKeyBytes[5] = __byte_perm(x32[3], x32[2], 0x0765);
  publicKeyBytes[6] = __byte_perm(x32[2], x32[1], 0x0765);
  publicKeyBytes[7] = __byte_perm(x32[1], x32[0], 0x0765);
  publicKeyBytes[8] = __byte_perm(x32[0], 0x80, 0x0456);
  publicKeyBytes[9] = 0;
  publicKeyBytes[10] = 0;
  publicKeyBytes[11] = 0;
  publicKeyBytes[12] = 0;
  publicKeyBytes[13] = 0;
  publicKeyBytes[14] = 0;
  publicKeyBytes[15] = 0x108;

  // Odd
  publicKeyBytes2[0] = __byte_perm(x32[7], 0x3, 0x4321);
  publicKeyBytes2[1] = publicKeyBytes[1];
  *(uint64_t *)(&publicKeyBytes2[2]) = *(uint64_t *)(&publicKeyBytes[2]);
  *(uint64_t *)(&publicKeyBytes2[4]) = *(uint64_t *)(&publicKeyBytes[4]);
  *(uint64_t *)(&publicKeyBytes2[6]) = *(uint64_t *)(&publicKeyBytes[6]);
  *(uint64_t *)(&publicKeyBytes2[8]) = *(uint64_t *)(&publicKeyBytes[8]);
  *(uint64_t *)(&publicKeyBytes2[10]) = *(uint64_t *)(&publicKeyBytes[10]);
  *(uint64_t *)(&publicKeyBytes2[12]) = *(uint64_t *)(&publicKeyBytes[12]);
  *(uint64_t *)(&publicKeyBytes2[14]) = *(uint64_t *)(&publicKeyBytes[14]);

  SHA256Initialize(s);
  SHA256Transform(s, publicKeyBytes);

#pragma unroll 8
  for (int i = 0; i < 8; i++)
    s[i] = bswap32(s[i]);

  *(uint64_t *)(s + 8) = 0x80ULL;
  *(uint64_t *)(s + 10) = 0ULL;
  *(uint64_t *)(s + 12) = 0ULL;
  *(uint64_t *)(s + 14) = ripemd160_sizedesc_32;

  RIPEMD160Initialize((uint32_t *)h1);
  RIPEMD160Transform((uint32_t *)h1, s);

  SHA256Initialize(s);
  SHA256Transform(s, publicKeyBytes2);

#pragma unroll 8
  for (int i = 0; i < 8; i++)
    s[i] = bswap32(s[i]);

  RIPEMD160Initialize((uint32_t *)h2);
  RIPEMD160Transform((uint32_t *)h2, s);

}

__device__ __noinline__ void _GetHash160(uint64_t *x, uint64_t *y, uint8_t *hash) {

  uint32_t *x32 = (uint32_t *)(x);
  uint32_t *y32 = (uint32_t *)(y);
  uint32_t publicKeyBytes[32];
  uint32_t s[16];

  // Uncompressed public key
  publicKeyBytes[0] = __byte_perm(x32[7], 0x04, 0x4321);
  publicKeyBytes[1] = __byte_perm(x32[7], x32[6], 0x0765);
  publicKeyBytes[2] = __byte_perm(x32[6], x32[5], 0x0765);
  publicKeyBytes[3] = __byte_perm(x32[5], x32[4], 0x0765);
  publicKeyBytes[4] = __byte_perm(x32[4], x32[3], 0x0765);
  publicKeyBytes[5] = __byte_perm(x32[3], x32[2], 0x0765);
  publicKeyBytes[6] = __byte_perm(x32[2], x32[1], 0x0765);
  publicKeyBytes[7] = __byte_perm(x32[1], x32[0], 0x0765);
  publicKeyBytes[8] = __byte_perm(x32[0], y32[7], 0x0765);
  publicKeyBytes[9] = __byte_perm(y32[7], y32[6], 0x0765);
  publicKeyBytes[10] = __byte_perm(y32[6], y32[5], 0x0765);
  publicKeyBytes[11] = __byte_perm(y32[5], y32[4], 0x0765);
  publicKeyBytes[12] = __byte_perm(y32[4], y32[3], 0x0765);
  publicKeyBytes[13] = __byte_perm(y32[3], y32[2], 0x0765);
  publicKeyBytes[14] = __byte_perm(y32[2], y32[1], 0x0765);
  publicKeyBytes[15] = __byte_perm(y32[1], y32[0], 0x0765);
  publicKeyBytes[16] = __byte_perm(y32[0], 0x80, 0x0456);
  publicKeyBytes[17] = 0;
  publicKeyBytes[18] = 0;
  publicKeyBytes[19] = 0;
  publicKeyBytes[20] = 0;
  publicKeyBytes[21] = 0;
  publicKeyBytes[22] = 0;
  publicKeyBytes[23] = 0;
  publicKeyBytes[24] = 0;
  publicKeyBytes[25] = 0;
  publicKeyBytes[26] = 0;
  publicKeyBytes[27] = 0;
  publicKeyBytes[28] = 0;
  publicKeyBytes[29] = 0;
  publicKeyBytes[30] = 0;
  publicKeyBytes[31] = 0x208;

  SHA256Initialize(s);
  SHA256Transform(s, publicKeyBytes);
  SHA256Transform(s, publicKeyBytes + 16);

#pragma unroll 8
  for (int i = 0; i < 8; i++)
    s[i] = bswap32(s[i]);

  *(uint64_t *)(s + 8) = 0x80ULL;
  *(uint64_t *)(s + 10) = 0ULL;
  *(uint64_t *)(s + 12) = 0ULL;
  *(uint64_t *)(s + 14) = ripemd160_sizedesc_32;

  RIPEMD160Initialize((uint32_t *)hash);
  RIPEMD160Transform((uint32_t *)hash, s);

}

__device__ inline bool device_memcmp(const void *s1, const void *s2, size_t n) {
    const uint8_t *p1 = (const uint8_t *)s1;
    const uint8_t *p2 = (const uint8_t *)s2;
    for (size_t i = 0; i < n; ++i) {
        if (p1[i] != p2[i]) return false;
    }
    return true;
}

// ===================================================================
// KERNEL UTAMA YANG DIOPTIMISASI UNTUK PENCARIAN HASH160
// ===================================================================

extern "C"
__global__ void find_hash_kernel_optimized(
    const BigInt* start_key,
    unsigned long long keys_per_launch,
    const BigInt* step,
    const uint8_t* d_targets,
    int num_targets,
    BigInt* d_result,
    int* d_found_flag
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= keys_per_launch || *d_found_flag) return;

    // 1. HITUNG PRIVATE KEY: start_key + step * idx
    BigInt current_priv;
    BigInt priv_idx_mul_step;

    // Kalikan step dengan idx
    bigint_mul_uint32(&priv_idx_mul_step, step, (uint32_t)idx);

    // Tambahkan ke start_key
    ptx_u256Add(&current_priv, start_key, &priv_idx_mul_step);

    // Modulo n (curve order)
    scalar_mod_n(&current_priv, &current_priv);

    // 2. PUBLIC KEY DENGAN PRECOMPUTED MULTIPLICATION
    ECPointJac result_jac;
    scalar_multiply_jac_precomputed(&result_jac, &current_priv);

    // 3. KONVERSI KE AFFINE COORDINATES
    ECPoint public_key;
    jacobian_to_affine(&public_key, &result_jac);

    if (public_key.infinity) return;

    // 4. HASH160 COMPUTATION
    uint8_t final_hash160[20];
    uint8_t is_odd = public_key.y.data[0] & 1;
    _GetHash160Comp((uint64_t*)public_key.x.data, is_odd, final_hash160);

    // 5. PERBANDINGAN DENGAN TARGET HASH160
    for (int i = 0; i < num_targets; i++) {
        if (device_memcmp(final_hash160, &d_targets[i * 20], 20)) {
            if (atomicCAS(d_found_flag, 0, 1) == 0) {
                copy_bigint(d_result, &current_priv);
            }
            return;
        }
    }
}

extern "C"
__global__ void find_hash_kernel(
    const BigInt* start_key,
    unsigned long long keys_per_launch,
    const BigInt* step,
    const uint8_t* d_targets,
    int num_targets,
    BigInt* d_result,
    int* d_found_flag
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= keys_per_launch || *d_found_flag) return;

    // 1. HITUNG PRIVATE KEY: start_key + step * idx
    BigInt priv_idx_mul_step;
    bigint_mul_uint32(&priv_idx_mul_step, step, (uint32_t)idx);

    BigInt current_priv;
    ptx_u256Add(&current_priv, start_key, &priv_idx_mul_step);
    scalar_mod_n(&current_priv, &current_priv);

    // 2. PRIVATE KEY -> PUBLIC KEY (tanpa precomputation)
    ECPointJac result_jac;
    scalar_multiply_jac_device(&result_jac, &const_G_jacobian, &current_priv);

    // 3. KONVERSI KE AFFINE
    ECPoint public_key;
    jacobian_to_affine(&public_key, &result_jac);
    if (public_key.infinity) return;

    // 4. HASH160 COMPUTATION
    uint8_t final_hash160[20];
    uint8_t is_odd = public_key.y.data[0] & 1;
    _GetHash160Comp((uint64_t*)public_key.x.data, is_odd, final_hash160);

    // 5. PERBANDINGAN DENGAN TARGET HASH160
    for (int i = 0; i < num_targets; i++) {
        if (device_memcmp(final_hash160, &d_targets[i * 20], 20)) {
            if (atomicCAS(d_found_flag, 0, 1) == 0) {
                copy_bigint(d_result, &current_priv);
            }
            return;
        }
    }
}

// ===================================================================
// KERNEL TAMBAHAN UNTUK KOMPATIBILITAS
// ===================================================================

__global__ void private_to_public_key_batch_kernel(
    const BigInt *d_private_keys,
    ECPoint *d_public_keys,
    int num_keys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    BigInt priv;
    copy_bigint(&priv, &d_private_keys[idx]);
    scalar_mod_n(&priv, &priv);

    ECPointJac result_jac;
    scalar_multiply_jac_device(&result_jac, &const_G_jacobian, &priv);

    ECPoint public_key;
    jacobian_to_affine(&public_key, &result_jac);

    d_public_keys[idx] = public_key;
}
