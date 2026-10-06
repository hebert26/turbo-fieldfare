#ifndef QWEN_PINNED_SOURCE_SCALAR_EXP
#define QWEN_PINNED_SOURCE_SCALAR_EXP
#include <metal_stdlib>
using namespace metal;
// Exact integer replay of the pinned macOS scalar expf binary64 polynomial.
// System provenance: libsystem_m.dylib UUID B54FBE99-DB0B-32C7-ABDD-C2206DF606A4,
// expf offset 0x5ac0, table offset 0x2db40. Generic constants only.
// Bounded binary64 replay for the captured scalar expf polynomial. Metal has no
// binary64 arithmetic. Integer limbs retain the exact 106-bit product and a
// guard/sticky tail before each nearest-even binary64 rounding. All nonzero
// binary64 operands/results here are normal: finite FP32 |x| < 128 produces
// reduction/polynomial values above 2^-180 and scale/output above 2^-186.
// This is not a general binary64 API. NaN/infinity/extreme inputs are handled
// by qwenSourceScalarExpFloat before reaching these normal-domain helpers.
struct QwenExpWide { ulong hi; ulong lo; };
inline QwenExpWide qwenExpWide(ulong hi, ulong lo) { return {hi, lo}; }
inline bool qwenExpNonzero(QwenExpWide a) { return (a.hi | a.lo) != 0ul; }
inline int qwenExpLeading(QwenExpWide a) {
    return a.hi ? 127 - int(clz(a.hi)) : (a.lo ? 63 - int(clz(a.lo)) : -1);
}
inline QwenExpWide qwenExpLeft(QwenExpWide a, int n) {
    if (n <= 0) return a;
    if (n >= 128) return qwenExpWide(0ul, 0ul);
    if (n >= 64) return qwenExpWide(a.lo << (n - 64), 0ul);
    return qwenExpWide((a.hi << n) | (a.lo >> (64 - n)), a.lo << n);
}
inline QwenExpWide qwenExpRight(QwenExpWide a, int n) {
    if (n <= 0) return a;
    if (n >= 128) return qwenExpWide(0ul, 0ul);
    if (n >= 64) return qwenExpWide(0ul, a.hi >> (n - 64));
    return qwenExpWide(a.hi >> n, (a.lo >> n) | (a.hi << (64 - n)));
}
inline bool qwenExpBelow(QwenExpWide a, int n) {
    if (n <= 0) return false;
    if (n >= 128) return qwenExpNonzero(a);
    if (n == 64) return a.lo != 0ul;
    if (n < 64) return (a.lo & ((1ul << n) - 1ul)) != 0ul;
    return a.lo != 0ul || (a.hi & ((1ul << (n - 64)) - 1ul)) != 0ul;
}
inline bool qwenExpBit(QwenExpWide a, int n) {
    if (n < 0 || n >= 128) return false;
    return n < 64 ? ((a.lo >> n) & 1ul) != 0ul : ((a.hi >> (n - 64)) & 1ul) != 0ul;
}
inline QwenExpWide qwenExpAlign(QwenExpWide a, int n) {
    if (n >= 0) return qwenExpLeft(a, n);
    QwenExpWide result = qwenExpRight(a, -n);
    if (qwenExpBelow(a, -n)) result.lo |= 1ul;
    return result;
}
inline QwenExpWide qwenExpProduct(ulong a, ulong b) {
    const ulong mask = 0xfffffffful;
    ulong al = a & mask, ah = a >> 32, bl = b & mask, bh = b >> 32;
    ulong first = al * bl, middle = ah * bl + (first >> 32);
    ulong lo = (first & mask) | (middle << 32);
    ulong hi = ah * bh + (middle >> 32), cross = al * bh, previous = lo;
    lo += cross << 32;
    hi += (cross >> 32) + ulong(lo < previous);
    return qwenExpWide(hi, lo);
}
inline QwenExpWide qwenExpAddWide(QwenExpWide a, QwenExpWide b) {
    ulong lo = a.lo + b.lo;
    return qwenExpWide(a.hi + b.hi + ulong(lo < a.lo), lo);
}
inline QwenExpWide qwenExpSubtractWide(QwenExpWide a, QwenExpWide b) {
    return qwenExpWide(a.hi - b.hi - ulong(a.lo < b.lo), a.lo - b.lo);
}
inline bool qwenExpLess(QwenExpWide a, QwenExpWide b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
inline ulong qwenExpRoundedMantissa(QwenExpWide a, int shift) {
    if (shift <= 0) return qwenExpLeft(a, -shift).lo;
    ulong result = qwenExpRight(a, shift).lo;
    if (qwenExpBit(a, shift - 1) && (qwenExpBelow(a, shift - 1) || (result & 1ul))) ++result;
    return result;
}
inline ulong qwenExpFromFloat(float value) {
    uint bits = as_type<uint>(value), fraction = bits & 0x007fffffu;
    ulong sign = ulong(bits >> 31) << 63;
    int exponent = int((bits >> 23) & 255u);
    if (exponent) return sign | (ulong(exponent - 127 + 1023) << 52) | (ulong(fraction) << 29);
    if (!fraction) return sign;
    int leading = 31 - int(clz(fraction));
    return sign | (ulong(leading - 149 + 1023) << 52)
        | ((ulong(fraction) << (52 - leading)) & 0x000ffffffffffffful);
}
inline ulong qwenExpNormalFMA(ulong a, ulong b, ulong c) {
    const ulong fractionMask = 0x000ffffffffffffful, magnitudeMask = 0x7ffffffffffffffful;
    if (!(a & magnitudeMask) || !(b & magnitudeMask)) return c;
    ulong am = (a & fractionMask) | (1ul << 52), bm = (b & fractionMask) | (1ul << 52);
    QwenExpWide product = qwenExpProduct(am, bm);
    int productPower = int((a >> 52) & 2047ul) + int((b >> 52) & 2047ul) - 2046 - 104;
    int productLeading = qwenExpLeading(product), cPower = 0;
    bool hasC = (c & magnitudeMask) != 0ul, negative = ((a ^ b) >> 63) != 0ul;
    int base = productPower + productLeading - 126;
    QwenExpWide addend = qwenExpWide(0ul, 0ul);
    if (hasC) {
        cPower = int((c >> 52) & 2047ul) - 1023 - 52;
        base = max(base, cPower + 52 - 126);
        addend = qwenExpAlign(qwenExpWide(0ul, (c & fractionMask) | (1ul << 52)), cPower - base);
    }
    product = qwenExpAlign(product, productPower - base);
    QwenExpWide sum;
    if (!hasC || negative == ((c >> 63) != 0ul)) sum = qwenExpAddWide(product, addend);
    else if (qwenExpLess(product, addend)) { sum = qwenExpSubtractWide(addend, product); negative = (c >> 63) != 0ul; }
    else sum = qwenExpSubtractWide(product, addend);
    int leading = qwenExpLeading(sum);
    if (leading < 0) return 0ul;
    ulong mantissa = qwenExpRoundedMantissa(sum, leading - 52);
    int exponent = base + leading;
    if (mantissa & (1ul << 53)) { mantissa >>= 1; ++exponent; }
    return (ulong(negative) << 63) | (ulong(exponent + 1023) << 52) | (mantissa & fractionMask);
}
inline float qwenExpToFloat(ulong bits) {
    int exponent = int((bits >> 52) & 2047ul) - 1023;
    QwenExpWide mantissa = qwenExpWide(0ul, (bits & 0x000ffffffffffffful) | (1ul << 52));
    if (exponent < -126) return as_type<float>(uint(qwenExpRoundedMantissa(mantissa, -exponent - 97)));
    ulong rounded = qwenExpRoundedMantissa(mantissa, 29);
    if (rounded & (1ul << 24)) { rounded >>= 1; ++exponent; }
    if (exponent > 127) return INFINITY;
    return as_type<float>((uint(exponent + 127) << 23) | (uint(rounded) & 0x007fffffu));
}
constant ulong qwenScalarSystemExpTable[128] = {
    0x3ff0000000000000ul,
    0x3ff0163da9fb3335ul,
    0x3ff02c9a3e778061ul,
    0x3ff04315e86e7f85ul,
    0x3ff059b0d3158574ul,
    0x3ff0706b29ddf6deul,
    0x3ff0874518759bc8ul,
    0x3ff09e3ecac6f383ul,
    0x3ff0b5586cf9890ful,
    0x3ff0cc922b7247f7ul,
    0x3ff0e3ec32d3d1a2ul,
    0x3ff0fb66affed31bul,
    0x3ff11301d0125b51ul,
    0x3ff12abdc06c31ccul,
    0x3ff1429aaea92de0ul,
    0x3ff15a98c8a58e51ul,
    0x3ff172b83c7d517bul,
    0x3ff18af9388c8deaul,
    0x3ff1a35beb6fcb75ul,
    0x3ff1bbe084045cd4ul,
    0x3ff1d4873168b9aaul,
    0x3ff1ed5022fcd91dul,
    0x3ff2063b88628cd6ul,
    0x3ff21f49917ddc96ul,
    0x3ff2387a6e756238ul,
    0x3ff251ce4fb2a63ful,
    0x3ff26b4565e27cddul,
    0x3ff284dfe1f56381ul,
    0x3ff29e9df51fdee1ul,
    0x3ff2b87fd0dad990ul,
    0x3ff2d285a6e4030bul,
    0x3ff2ecafa93e2f56ul,
    0x3ff306fe0a31b715ul,
    0x3ff32170fc4cd831ul,
    0x3ff33c08b26416fful,
    0x3ff356c55f929ff1ul,
    0x3ff371a7373aa9cbul,
    0x3ff38cae6d05d866ul,
    0x3ff3a7db34e59ff7ul,
    0x3ff3c32dc313a8e5ul,
    0x3ff3dea64c123422ul,
    0x3ff3fa4504ac801cul,
    0x3ff4160a21f72e2aul,
    0x3ff431f5d950a897ul,
    0x3ff44e086061892dul,
    0x3ff46a41ed1d0057ul,
    0x3ff486a2b5c13cd0ul,
    0x3ff4a32af0d7d3deul,
    0x3ff4bfdad5362a27ul,
    0x3ff4dcb299fddd0dul,
    0x3ff4f9b2769d2ca7ul,
    0x3ff516daa2cf6642ul,
    0x3ff5342b569d4f82ul,
    0x3ff551a4ca5d920ful,
    0x3ff56f4736b527daul,
    0x3ff58d12d497c7fdul,
    0x3ff5ab07dd485429ul,
    0x3ff5c9268a5946b7ul,
    0x3ff5e76f15ad2148ul,
    0x3ff605e1b976dc09ul,
    0x3ff6247eb03a5585ul,
    0x3ff6434634ccc320ul,
    0x3ff6623882552225ul,
    0x3ff68155d44ca973ul,
    0x3ff6a09e667f3bcdul,
    0x3ff6c012750bdabful,
    0x3ff6dfb23c651a2ful,
    0x3ff6ff7df9519484ul,
    0x3ff71f75e8ec5f74ul,
    0x3ff73f9a48a58174ul,
    0x3ff75feb564267c9ul,
    0x3ff780694fde5d3ful,
    0x3ff7a11473eb0187ul,
    0x3ff7c1ed0130c132ul,
    0x3ff7e2f336cf4e62ul,
    0x3ff80427543e1a12ul,
    0x3ff82589994cce13ul,
    0x3ff8471a4623c7adul,
    0x3ff868d99b4492edul,
    0x3ff88ac7d98a6699ul,
    0x3ff8ace5422aa0dbul,
    0x3ff8cf3216b5448cul,
    0x3ff8f1ae99157736ul,
    0x3ff9145b0b91ffc6ul,
    0x3ff93737b0cdc5e5ul,
    0x3ff95a44cbc8520ful,
    0x3ff97d829fde4e50ul,
    0x3ff9a0f170ca07baul,
    0x3ff9c49182a3f090ul,
    0x3ff9e86319e32323ul,
    0x3ffa0c667b5de565ul,
    0x3ffa309bec4a2d33ul,
    0x3ffa5503b23e255dul,
    0x3ffa799e1330b358ul,
    0x3ffa9e6b5579fdbful,
    0x3ffac36bbfd3f37aul,
    0x3ffae89f995ad3adul,
    0x3ffb0e07298db666ul,
    0x3ffb33a2b84f15fbul,
    0x3ffb59728de5593aul,
    0x3ffb7f76f2fb5e47ul,
    0x3ffba5b030a1064aul,
    0x3ffbcc1e904bc1d2ul,
    0x3ffbf2c25bd71e09ul,
    0x3ffc199bdd85529cul,
    0x3ffc40ab5fffd07aul,
    0x3ffc67f12e57d14bul,
    0x3ffc8f6d9406e7b5ul,
    0x3ffcb720dcef9069ul,
    0x3ffcdf0b555dc3faul,
    0x3ffd072d4a07897cul,
    0x3ffd2f87080d89f2ul,
    0x3ffd5818dcfba487ul,
    0x3ffd80e316c98398ul,
    0x3ffda9e603db3285ul,
    0x3ffdd321f301b460ul,
    0x3ffdfc97337b9b5ful,
    0x3ffe264614f5a129ul,
    0x3ffe502ee78b3ff6ul,
    0x3ffe7a51fbc74c83ul,
    0x3ffea4afa2a490daul,
    0x3ffecf482d8e67f1ul,
    0x3ffefa1bee615a27ul,
    0x3fff252b376bba97ul,
    0x3fff50765b6e4540ul,
    0x3fff7bfdad9cbe14ul,
    0x3fffa7c1819e90d8ul,
    0x3fffd3c22b8f71f1ul,
};

inline float qwenSourceScalarExpFloat(float value) {
#pragma clang fp contract(off)
    if (isnan(value)) return value + value;
    if (value >= 128.0f) return INFINITY;
    if (value <= -128.0f) return 0.0f;
    const ulong inverse = 0x40671547652b82feul, shift = 0x4338000000000000ul;
    const ulong quadratic = 0x3eeebfbdff30d656ul, linear = 0x3f762e4453e10daeul;
    const ulong input = qwenExpFromFloat(value);
    // The system magic shift rounds inverse*x to the nearest-even integer.
    // Its binary64 spacing is one throughout this bounded input domain.
    const ulong shifted = qwenExpNormalFMA(inverse, input, shift);
    const int n = int(long(shifted) - long(shift));
    const ulong r = qwenExpNormalFMA(inverse, input, qwenExpFromFloat(-float(n)));
    const ulong polynomial = qwenExpNormalFMA(quadratic, r, linear);
    const ulong product = qwenExpNormalFMA(polynomial, r, 0ul);
    const ulong table = qwenScalarSystemExpTable[uint(n) & 127u];
    const ulong scale = (table & 0x000ffffffffffffful) | (ulong(1023 + (n >> 7)) << 52);
    return qwenExpToFloat(qwenExpNormalFMA(product, scale, scale));
}
// Correctly rounded positive reciprocal when the FP32 result is subnormal.
// Work in units of 2^-149. Power-of-two rescaling keeps every intermediate
// normal, and the exact FMA residual resolves nearest-even integer rounding.
inline float qwenSourceScalarPositiveReciprocal(float denominator) {
#pragma clang fp contract(off)
    if (!(denominator > 0x1p126f) || isinf(denominator)) return 1.0f / denominator;
    float scaledDenominator = as_type<float>(as_type<uint>(denominator) - (23u << 23));
    const float numerator = 0x1p126f;
    float quotient = rint(numerator / scaledDenominator);
    float residual = fma(-quotient, scaledDenominator, numerator);
    float midpoint = 0.5f * scaledDenominator;
    uint integer = uint(quotient);
    if (residual > midpoint || (residual == midpoint && (integer & 1u))) ++integer;
    if (residual < -midpoint || (residual == -midpoint && (integer & 1u))) --integer;
    return as_type<float>(integer);
}

#endif
