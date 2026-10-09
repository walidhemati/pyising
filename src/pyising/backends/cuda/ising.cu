// Hand-written CUDA C reference kernels, compiled at run time with NVRTC (CuPy RawModule).
// Compile-time constants: LH (= Lx/2), LY, LZ, NDIM, HEATBATH, RNG_PER_SITE, NW (= LH/64),
// LAZY.  Philox4x32-10 as in Salmon et al. (SC'11) / Random123.

typedef unsigned int u32;
typedef unsigned long long u64;

struct u32x4 { u32 x, y, z, w; };

__device__ __forceinline__ u32x4 philox4x32_10(u32 c0, u32 c1, u32 c2, u32 c3, u32 k0, u32 k1)
{
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const u32 hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        const u32 hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
        k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    u32x4 out = {c0, c1, c2, c3};
    return out;
}

#ifdef LH
// ---------------------------------------------------------------- int8 checkerboard ---------
__device__ __forceinline__ signed char update(int s, int h, u32 u, const u32* __restrict__ table)
{
#if HEATBATH
    return (u < __ldg(table + (h + 2 * NDIM) / 2)) ? 1 : -1;
#else
    const int x = s * h;
    return (x <= 0 || u < __ldg(table + (x + 2 * NDIM) / 2)) ? -s : s;
#endif
}

extern "C" __global__ void int8_half(signed char* __restrict__ own,
                                     const signed char* __restrict__ other,
                                     const u32* __restrict__ table, long long n_quads,
                                     u32 t, u32 color, u32 k0, u32 k1)
{
    const long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_quads) return;
    const long long n = q * 4;
    const long long row = n / LH;
    const int j = (int)(n - row * LH);
    const int y = (int)(row % LY);
    const int z = (int)(row / LY);
    const int p = (y + z + (int)color) & 1;
    const long long rbase = row * LH;

    const char4 s = *reinterpret_cast<const char4*>(own + n);
    const char4 c = *reinterpret_cast<const char4*>(other + n);
    const long long yu = (long long)(z * LY + (y + 1 == LY ? 0 : y + 1)) * LH + j;
    const long long yd = (long long)(z * LY + (y == 0 ? LY - 1 : y - 1)) * LH + j;
    const char4 a = *reinterpret_cast<const char4*>(other + yu);
    const char4 b = *reinterpret_cast<const char4*>(other + yd);
    int h0 = c.x + a.x + b.x, h1 = c.y + a.y + b.y, h2 = c.z + a.z + b.z, h3 = c.w + a.w + b.w;
#if NDIM == 3
    const long long zu = (long long)((z + 1 == LZ ? 0 : z + 1) * LY + y) * LH + j;
    const long long zd = (long long)((z == 0 ? LZ - 1 : z - 1) * LY + y) * LH + j;
    const char4 e = *reinterpret_cast<const char4*>(other + zu);
    const char4 f = *reinterpret_cast<const char4*>(other + zd);
    h0 += e.x + f.x; h1 += e.y + f.y; h2 += e.z + f.z; h3 += e.w + f.w;
#endif
    if (p) {  // horizontal neighbour is j + 1
        const int jn = (j + 4 == LH) ? 0 : j + 4;
        h0 += c.y; h1 += c.z; h2 += c.w; h3 += other[rbase + jn];
    } else {  // horizontal neighbour is j - 1
        const int jp = (j == 0) ? LH - 1 : j - 1;
        h0 += other[rbase + jp]; h1 += c.x; h2 += c.y; h3 += c.z;
    }
#if RNG_PER_SITE
    // ablation: one Philox call per site (same numbers, 4x the RNG work)
    const u32 u0 = philox4x32_10((u32)q, t, color, 0u, k0, k1).x;
    const u32 u1 = philox4x32_10((u32)q, t, color, 0u, k0, k1 + (u32)(h0 == 99)).y;
    const u32 u2 = philox4x32_10((u32)q, t, color, 0u, k0, k1 + (u32)(h1 == 99)).z;
    const u32 u3 = philox4x32_10((u32)q, t, color, 0u, k0, k1 + (u32)(h2 == 99)).w;
#else
    const u32x4 r = philox4x32_10((u32)q, t, color, 0u, k0, k1);
    const u32 u0 = r.x, u1 = r.y, u2 = r.z, u3 = r.w;
#endif
    char4 o;
    o.x = update(s.x, h0, u0, table);
    o.y = update(s.y, h1, u1, table);
    o.z = update(s.z, h2, u2, table);
    o.w = update(s.w, h3, u3, table);
    *reinterpret_cast<char4*>(own + n) = o;
}
#endif

#ifdef NW
// ------------------------------------------- multi-spin coding: 64 spins per 64-bit word ----
__device__ __forceinline__ void lazy_step(u64 r, int bit, u32 thr4, u32 thr8, u64 a0, u64 a1,
                                          u64& acc, u64& und)
{
    const u64 t4 = ((thr4 >> bit) & 1u) ? ~0ull : 0ull;
    const u64 t8 = ((thr8 >> bit) & 1u) ? ~0ull : 0ull;
    const u64 tm = (a1 & t4) | (a0 & t8);
    acc |= und & ~r & tm;   // first differing bit: u has 0, threshold has 1 -> u < thr
    und &= ~(r ^ tm);
}

extern "C" __global__ void msc_half(u64* __restrict__ own, const u64* __restrict__ other,
                                    long long n_words, u32 t, u32 color, u32 k0, u32 k1,
                                    u32 thr4, u32 thr8, unsigned long long* __restrict__ stats)
{
    const long long w = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n_words) return;
    const int y = (int)(w / NW);
    const int x = (int)(w - (long long)y * NW);
    const long long rbase = (long long)y * NW;
    const int p = (y + (int)color) & 1;
    const u64 o = own[w];
    const u64 c = other[w];
    u64 hz;
    if (p) {
        const u64 nx = other[rbase + (x + 1 == NW ? 0 : x + 1)];
        hz = (c >> 1) | (nx << 63);
    } else {
        const u64 pv = other[rbase + (x == 0 ? NW - 1 : x - 1)];
        hz = (c << 1) | (pv >> 63);
    }
    const u64 up = other[(long long)(y + 1 == LY ? 0 : y + 1) * NW + x];
    const u64 dn = other[(long long)(y == 0 ? LY - 1 : y - 1) * NW + x];
    const u64 x1 = o ^ c, x2 = o ^ hz, x3 = o ^ up, x4 = o ^ dn;
    const u64 any = x1 | x2 | x3 | x4;
    const u64 a0 = ~any;
    const u64 s1 = x1 ^ x2, c1 = x1 & x2, s2 = x3 ^ x4, c2 = x3 & x4;
    const u64 a2 = c1 | c2 | (s1 & s2);
    const u64 a1 = any & ~a2;
    u64 acc = 0, und = a0 | a1;
    int calls = 0;
#if LAZY
    for (int lp = 0; lp < 16 && und; ++lp) {
#else
    for (int lp = 0; lp < 16; ++lp) {
#endif
        const u32x4 r = philox4x32_10((u32)w, t, color, 1u + (u32)lp, k0, k1);
        ++calls;
        lazy_step((u64)r.x | ((u64)r.y << 32), 31 - 2 * lp, thr4, thr8, a0, a1, acc, und);
        lazy_step((u64)r.z | ((u64)r.w << 32), 30 - 2 * lp, thr4, thr8, a0, a1, acc, und);
    }
    own[w] = o ^ (a2 | acc);
    if (stats) atomicAdd(stats, (unsigned long long)calls);
}
#endif
