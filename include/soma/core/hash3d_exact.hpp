/**
 * hash3d_exact.hpp -- exact (and optionally tolerance-based) global numbering
 * of FP64 points.  Header-only, any dimension D >= 1.
 *
 * It numbers points held by one process; across MPI ranks, match_points
 * (soma/parallel/point_match.hpp) gives the same numbering.
 *
 * ===========================================================================
 *  API
 * ===========================================================================
 *   assign_ids_exact<D>(pts)
 *       Two points share an id  <=>  every coordinate is bit-identical
 *       (-0.0 is canonicalised to +0.0; NaN throws std::invalid_argument).
 *       Points that differ by 1 ulp always get different ids.
 *
 *   assign_ids_tol<D>(pts, tol, norm)
 *       Points are merged into the connected components of the graph
 *           edge(a,b)  <=>  norm(a-b)/tol <= 1
 *       (single linkage: chains A~B~C merge even if |A-C| > tol).
 *       Pair criterion (identical in the Python prototype), r_j = |a_j-b_j| / tol:
 *           Norm::Linf : max_j r_j <= 1
 *           Norm::L1   : sum_j r_j <= 1
 *           Norm::L2   : sum_j r_j*r_j <= 1        (summed in axis order)
 *       Dividing by tol first makes the test immune to under/overflow of
 *       squares, so any finite tol > 0 works (1e-320 ... 1e300).
 *       Points with a non-finite coordinate (+-inf) are never merged with
 *       anything except exact duplicates of themselves.
 *
 *   Ids (both modes)
 *       Contiguous in [0, M), deterministic, independent of input order.
 *       ids are ranks of each cluster's smallest exact key (lexicographic,
 *       x-major).  unique_points[id] is that smallest member (the cluster
 *       representative); in exact mode it is the point itself.
 *
 *   hash128<D>(p)
 *       Fixed-width digest for bucketing / MPI routing.  NOT an identity.
 *
 * Complexity (tolerance mode): O(N log N) sort + O(N * k) pair tests, where k
 * is the average number of points in the 3^D neighbouring grid cells.  If tol
 * is so large that most points fall into a few cells the pair tests degrade
 * towards O(N^2) -- inherent to single-linkage clustering by brute search.
 *
 * ===========================================================================
 *  COMPILATION NOTES AND FLAGS
 * ===========================================================================
 *  Integration
 *    - Header-only: add the `include/` directory to the include path and
 *          #include "soma/core/hash3d_exact.hpp"
 *      No libraries, no build step, no dependencies beyond the C++ standard
 *      library.  Every function is a template or `inline`, so the header may
 *      be included from any number of translation units (ODR-safe; checked by
 *      tests/test_odr_*.cpp).  A CMake INTERFACE target `hash3d_exact::hash3d_exact`
 *      is provided for add_subdirectory()/find_package() users.
 *
 *  Language standard
 *    - C++11 or newer.  Tested: g++ 13 and clang++ with -std=c++11, 14, 17, 20.
 *    - Requires IEEE-754 binary64 `double` (checked with static_assert).
 *
 *  Exact mode (assign_ids_exact, exact_key, hash128)
 *    - Uses ONLY integer operations on the bit pattern of each coordinate, so
 *      it is immune to floating-point flags: it stays correct even under
 *      -ffast-math / -Ofast (including flush-to-zero of subnormals) and with
 *      any rounding mode.
 *
 *  Tolerance mode (assign_ids_tol) needs strict IEEE-754 double arithmetic
 *    - Recommended: -O2 or -O3.  -march=native is fine: fused multiply-add
 *      contraction is blocked INSIDE the pair criterion (an inline-asm
 *      barrier on GCC/Clang, a volatile round trip elsewhere), so results do
 *      not depend on whether the compiler would have used FMA.  Without the
 *      barrier, FMA-capable CPUs (x86 with -march=native, Apple Silicon,
 *      POWER, ...) flip decisions on near-tie pairs.
 *    - NEVER use -ffast-math, -Ofast, -funsafe-math-optimizations,
 *      -fassociative-math, -freciprocal-math, -ffinite-math-only or
 *      -fno-signed-zeros.  (-ffast-math additionally links code that sets the
 *      FTZ/DAZ CPU flags, which flush subnormals and break tolerance mode for
 *      tiny tol.)  Other compilers: use the IEEE-strict model, e.g.
 *          MSVC            /fp:precise (default) or /fp:strict, not /fp:fast
 *          Intel icx/icpx  -fp-model=precise   (their default is fast!)
 *          NVIDIA nvc++    -Kieee
 *          Cray/IBM/others: the option that disables unsafe FP optimisations
 *    - 32-bit x86: use SSE2 math (-msse2 -mfpmath=sse); x87 extended precision
 *      is not supported.
 *    - assign_ids_tol runs a cheap one-time self-check (fast-math macros, subnormal
 *      arithmetic, FMA-sensitive tie pairs) and throws std::runtime_error if the build
 *      or CPU state is unsafe.  fp_environment_ok() exposes the same check.
 *    - Define HASH3D_EXACT_PORTABLE_BARRIER to use the volatile-based barrier
 *      instead of inline asm (automatic on non-GNU/Clang compilers).
 *
 *  Warnings: clean with -Wall -Wextra -Wpedantic -Wconversion -Wshadow
 *  -Wsign-conversion on g++ and clang++.  Also clean under ASan + UBSan.
 *
 *  Threading: all functions are reentrant (no global mutable state apart from
 *  the thread-safe one-time FP self-check); a call is single-threaded.
 */
#ifndef HASH3D_EXACT_HPP
#define HASH3D_EXACT_HPP
#pragma once

#define HASH3D_EXACT_VERSION_MAJOR 2
#define HASH3D_EXACT_VERSION_MINOR 0
#define HASH3D_EXACT_VERSION_PATCH 0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8,
              "hash3d_exact requires IEEE-754 binary64 double");

namespace spatial_exact {

template <std::size_t D> using Point = std::array<double, D>;
template <std::size_t D> using Key = std::array<uint64_t, D>;
using Key3 = Key<3>;

enum class Norm { L1, L2, Linf };

// ---------------------------------------------------------------------------
// Exact keys
// ---------------------------------------------------------------------------
namespace detail {
// Bit-pattern key of one coordinate: order-preserving, injective, integer ops only.
// -0.0 -> +0.0.  `is_nan` is set (and the key is meaningless) for NaN inputs.
inline uint64_t key_bits(double v, bool& is_nan) {
    uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    is_nan = ((b & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL) && ((b & 0x000FFFFFFFFFFFFFULL) != 0);
    if (b == 0x8000000000000000ULL) b = 0;  // -0.0 -> +0.0
    return (b >> 63) ? ~b : (b | (uint64_t{1} << 63));
}
inline bool is_finite_bits(double v) {
    uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return (b & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}
}  // namespace detail

inline uint64_t key_of(double v) {
    bool nan = false;
    const uint64_t k = detail::key_bits(v, nan);
    if (nan) throw std::invalid_argument("hash3d_exact: NaN coordinate");
    return k;
}

inline double value_of(uint64_t k) {
    const uint64_t b = (k >> 63) ? (k & ~(uint64_t{1} << 63)) : ~k;
    double v;
    std::memcpy(&v, &b, sizeof v);
    return v;
}

template <std::size_t D>
inline Key<D> exact_key(const Point<D>& p) {
    Key<D> k;
    for (std::size_t j = 0; j < D; ++j) k[j] = key_of(p[j]);
    return k;
}
inline Key3 exact_key(double x, double y, double z) { return exact_key<3>(Point<3>{x, y, z}); }

template <std::size_t D>
inline Point<D> point_of(const Key<D>& k) {
    Point<D> p;
    for (std::size_t j = 0; j < D; ++j) p[j] = value_of(k[j]);
    return p;
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
template <std::size_t D>
struct IdResult {
    std::vector<Point<D>> unique_points;  // M representatives, key-ordered
    std::vector<int64_t> ids;             // N entries in [0, M)
};

// ---------------------------------------------------------------------------
// Exact ids
// ---------------------------------------------------------------------------
template <std::size_t D>
IdResult<D> assign_ids_exact(const std::vector<Point<D>>& pts) {
    static_assert(D >= 1, "dimension must be >= 1");
    struct KI {
        Key<D> k;
        std::size_t i;
    };
    const std::size_t n = pts.size();
    std::vector<KI> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = KI{exact_key<D>(pts[i]), i};
    std::sort(v.begin(), v.end(), [](const KI& a, const KI& b) { return a.k < b.k; });

    IdResult<D> r;
    r.ids.resize(n);
    int64_t id = -1;
    for (std::size_t s = 0; s < n; ++s) {
        if (s == 0 || v[s].k != v[s - 1].k) {
            ++id;
            r.unique_points.push_back(point_of<D>(v[s].k));
        }
        r.ids[v[s].i] = id;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Tolerance ids
// ---------------------------------------------------------------------------
namespace detail {

// Forces `v` through memory so the compiler cannot fuse the multiply that
// produced it with a following add (FMA contraction would change rounding).
#if defined(HASH3D_EXACT_PORTABLE_BARRIER) || !(defined(__GNUC__) || defined(__clang__))
template <class T>
inline void fp_barrier(T& v) {
    volatile T t = v;
    v = t;
}
#else
template <class T>
inline void fp_barrier(T& v) {
    __asm__ volatile("" : "+m"(v));
}
#endif

template <std::size_t D>
inline bool within(const Point<D>& a, const Point<D>& b, double tol, Norm norm) {
    double s = 0.0;
    for (std::size_t j = 0; j < D; ++j) {
        const double r = std::fabs(a[j] - b[j]) / tol;
        switch (norm) {
            case Norm::Linf:
                if (!(r <= 1.0)) return false;
                break;
            case Norm::L1:
                s += r;
                break;
            case Norm::L2: {
                double t = r * r;
                fp_barrier(t);
                s += t;
                break;
            }
        }
    }
    return norm == Norm::Linf ? true : (s <= 1.0);
}

// One-time check that double arithmetic behaves as strict IEEE-754:
//  * subnormals are not flushed (-ffast-math links FTZ/DAZ start-up code),
//  * two L2 tie pairs known to flip under FMA contraction / reassociation
//    produce the unfused answer.
inline bool fp_environment_compute() {
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
    return false;  // translation unit compiled with -ffast-math / /fp:fast: x/tol may become x*(1/tol), etc.
#endif
    // (compare BIT PATTERNS: with DAZ enabled even `==` treats subnormal operands as zero)
    volatile double tiny = 5e-324, two = 2.0;
    volatile double t2 = tiny * two;            // exact result: smallest subnormal times 2 == bit pattern 2
    const double t2v = t2;
    uint64_t t2bits;
    std::memcpy(&t2bits, &t2v, sizeof t2bits);
    if (t2bits != 2u) return false;
    const Point<2> o = {{0.0, 0.0}};
    const Point<2> a = {{0.37070783362460713, 0.9287495367909212}};   // unfused sum > 1, fused <= 1
    const Point<2> b = {{0.18598426076673474, 0.9825527236474648}};   // unfused sum <= 1, fused > 1
    if (within<2>(o, a, 1.0, Norm::L2)) return false;
    if (!within<2>(o, b, 1.0, Norm::L2)) return false;
    return true;
}

// Grid cell of one coordinate.  Cell width is c = 2^k with tol < c <= 2*tol, so
// x/c is exact (power-of-two scaling) and two coordinates within tol have
// quotients differing by < 1, i.e. cell indices differing by at most 1.
// When |x/c| > 2^53 the spacing of doubles exceeds c > tol, so two such
// coordinates within tol are identical (and a "near" coordinate can never be
// within tol of a "far" one).  Far coordinates only need to share a
// deterministic cell id derived from their bits.
inline int64_t cell_of(double x, int k) {
    const double q = std::ldexp(x, -k);
    if (std::fabs(q) <= 9007199254740992.0 /* 2^53 */) return static_cast<int64_t>(std::floor(q));
    return static_cast<int64_t>(key_of(x) >> 2);
}

}  // namespace detail

/// True if the compiler/CPU state gives strict IEEE-754 double arithmetic (see notes above).
inline bool fp_environment_ok() {
    static const bool ok = detail::fp_environment_compute();
    return ok;
}

template <std::size_t D>
IdResult<D> assign_ids_tol(const std::vector<Point<D>>& pts, double tol, Norm norm = Norm::L2) {
    if (!fp_environment_ok())
        throw std::runtime_error(
            "hash3d_exact: unsafe floating-point environment for tolerance mode "
            "(built with -ffast-math/-Ofast, FTZ/DAZ enabled, or unexpected FMA/reassociation); "
            "see the COMPILATION NOTES in hash3d_exact.hpp");
    if (!(tol > 0.0) || !detail::is_finite_bits(tol))
        throw std::invalid_argument("hash3d_exact: tol must be finite and > 0");

    IdResult<D> ex = assign_ids_exact<D>(pts);  // distinct points, key-sorted
    const std::vector<Point<D>>& U = ex.unique_points;
    const std::size_t M = U.size();
    if (M == 0) return ex;

    // ---- grid ------------------------------------------------------------
    int e = 0;
    std::frexp(tol, &e);  // tol = m * 2^e, m in [0.5, 1)
    const int k = e;  // cell width 2^k in (tol, 2 tol]

    using Cell = std::array<int64_t, D>;
    struct Rec {
        Cell c;
        std::size_t idx;
    };
    std::vector<Rec> recs;
    recs.reserve(M);
    for (std::size_t i = 0; i < M; ++i) {
        bool finite = true;
        for (std::size_t j = 0; j < D; ++j) finite = finite && detail::is_finite_bits(U[i][j]);
        if (!finite) continue;  // stays a singleton
        Rec r;
        r.idx = i;
        for (std::size_t j = 0; j < D; ++j) r.c[j] = detail::cell_of(U[i][j], k);
        recs.push_back(r);
    }
    std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) {
        return a.c != b.c ? a.c < b.c : a.idx < b.idx;
    });

    struct CellRun {
        Cell c;
        std::size_t b, e;
    };
    std::vector<CellRun> cells;
    for (std::size_t s = 0; s < recs.size();) {
        std::size_t t = s + 1;
        while (t < recs.size() && recs[t].c == recs[s].c) ++t;
        cells.push_back(CellRun{recs[s].c, s, t});
        s = t;
    }

    // Cells are sorted lexicographically, so the three cells (prefix, c-1..c+1)
    // that share a neighbouring prefix row are adjacent in memory: one binary
    // search per forward prefix offset instead of three.  Forward prefix
    // offsets: o in {-1,0,1}^(D-1), lexicographically > 0.  The row with
    // o == 0 is handled by looking at the next cell (last-axis offset +1).
    std::vector<std::array<int, D>> offs;
    {
        std::size_t total = 1;
        for (std::size_t j = 0; j + 1 < D; ++j) total *= 3;
        for (std::size_t t = 0; t < total && D > 1; ++t) {
            std::array<int, D> o{};
            std::size_t u = t;
            for (std::size_t j = D - 1; j-- > 0;) {
                o[j] = static_cast<int>(u % 3) - 1;
                u /= 3;
            }
            std::size_t f = 0;
            while (f + 1 < D && o[f] == 0) ++f;
            if (f + 1 < D && o[f] > 0) offs.push_back(o);
        }
    }

    // ---- union-find (root == smallest index == smallest exact key) -------
    std::vector<std::size_t> parent(M);
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    auto find = [&](std::size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto try_union = [&](std::size_t i, std::size_t j) {
        const std::size_t ri = find(i), rj = find(j);
        if (ri == rj) return;
        if (!detail::within<D>(U[i], U[j], tol, norm)) return;
        if (ri < rj) parent[rj] = ri; else parent[ri] = rj;
    };

    auto same_prefix = [](const Cell& x, const Cell& y) {
        for (std::size_t j = 0; j + 1 < D; ++j)
            if (x[j] != y[j]) return false;
        return true;
    };
    auto pairs = [&](const CellRun& P, const CellRun& Q) {
        for (std::size_t p = P.b; p < P.e; ++p)
            for (std::size_t q = Q.b; q < Q.e; ++q) try_union(recs[p].idx, recs[q].idx);
    };
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

    for (std::size_t a = 0; a < cells.size(); ++a) {
        const CellRun& A = cells[a];
        for (std::size_t p = A.b; p < A.e; ++p)
            for (std::size_t q = p + 1; q < A.e; ++q) try_union(recs[p].idx, recs[q].idx);

        const int64_t last = A.c[D - 1];
        // same prefix row, last axis + 1  (the neighbouring cell, if present, is next in order)
        if (a + 1 < cells.size() && last != kMax && same_prefix(cells[a + 1].c, A.c) &&
            cells[a + 1].c[D - 1] == last + 1)
            pairs(A, cells[a + 1]);

        // forward prefix rows: last axis in [last-1, last+1]
        const int64_t lo = (last == kMin) ? kMin : last - 1;
        const int64_t hi = (last == kMax) ? kMax : last + 1;
        for (const auto& o : offs) {
            Cell key;
            bool ok = true;
            for (std::size_t j = 0; j + 1 < D && ok; ++j) {
                const int64_t v = A.c[j];
                if (o[j] > 0) { if (v == kMax) ok = false; else key[j] = v + 1; }
                else if (o[j] < 0) { if (v == kMin) ok = false; else key[j] = v - 1; }
                else key[j] = v;
            }
            if (!ok) continue;
            key[D - 1] = lo;
            auto it = std::lower_bound(cells.begin(), cells.end(), key,
                                       [](const CellRun& cr, const Cell& k2) { return cr.c < k2; });
            for (; it != cells.end() && same_prefix(it->c, key) && it->c[D - 1] <= hi; ++it) pairs(A, *it);
        }
    }

    // ---- final ids -------------------------------------------------------
    std::vector<int64_t> comp(M, -1);
    IdResult<D> out;
    int64_t next = 0;
    for (std::size_t i = 0; i < M; ++i)
        if (find(i) == i) {
            comp[i] = next++;
            out.unique_points.push_back(U[i]);
        }
    std::vector<int64_t> uid(M);
    for (std::size_t i = 0; i < M; ++i) uid[i] = comp[find(i)];
    out.ids.resize(ex.ids.size());
    for (std::size_t j = 0; j < ex.ids.size(); ++j) out.ids[j] = uid[static_cast<std::size_t>(ex.ids[j])];
    return out;
}

// ---------------------------------------------------------------------------
// Fixed-width digest (NOT an identity)
// ---------------------------------------------------------------------------
inline uint64_t mix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
inline uint64_t mix64_alt(uint64_t x) {
    x = (x ^ (x >> 33)) * 0xFF51AFD7ED558CCDULL;
    x = (x ^ (x >> 33)) * 0xC4CEB9FE1A85EC53ULL;
    return x ^ (x >> 33);
}

template <std::size_t D>
inline std::array<uint64_t, 2> hash128(const Point<D>& p) {
    const Key<D> k = exact_key<D>(p);
    uint64_t hi = 0x243F6A8885A308D3ULL, lo = 0x13198A2E03707344ULL;
    for (std::size_t j = 0; j < D; ++j) {
        hi = mix64(hi ^ (k[j] + 0x9E3779B97F4A7C15ULL * uint64_t(j + 1)));
        lo = mix64_alt(lo + k[j] * 0xD6E8FEB86659FD93ULL + uint64_t(j));
    }
    const uint64_t h2 = mix64(hi ^ mix64_alt(lo));
    const uint64_t l2 = mix64_alt(lo ^ mix64(hi));
    return {h2, l2};
}
inline std::array<uint64_t, 2> hash128(double x, double y, double z) { return hash128<3>(Point<3>{x, y, z}); }

}  // namespace spatial_exact

#endif  // HASH3D_EXACT_HPP
