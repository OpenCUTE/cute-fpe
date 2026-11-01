#pragma once

#include <stdlib.h>
#include <math.h>
#include <cstdint>
#include <string>

// #define FLOATDECODE_DEBUG

typedef struct{
    int32_t is_subnormal; // 是否是非规格化数
    int32_t is_nan;
    int32_t is_pinf;
    int32_t is_ninf;
    int32_t is_pzero;
    int32_t is_nzero;
} DecodeException;

typedef struct{
    int32_t sign;
    int32_t exponent;
    int64_t mantissa;
    DecodeException exception; // 用于存储解码异常信息
} FloatDecode;

inline int DecodeExceptionInit(DecodeException* exception) {
    if (!exception) return -1;
    exception->is_subnormal = 0;
    exception->is_nan = 0;
    exception->is_pinf = 0;
    exception->is_ninf = 0;
    exception->is_pzero = 0;
    exception->is_nzero = 0;
    return 0;
}

inline FloatDecode decode(int32_t f, int dtype) {
    // dtype: 0 - FP32/TF32, 1 - FP16, 2 - BF16, 3 - e4m3, 4 - FP4, 5 - e5m2, 6 - e3m2, 7 - e2m3
    int exp_width;
    int mantissa_width; //编码的尾数位宽，不包括隐含的1位
    switch(dtype) {
        case 0: // FP32
            exp_width = 8;
            mantissa_width = 23;
            break;
        case 1: // FP16
            exp_width = 5;
            mantissa_width = 10;
            break;
        case 2: // BF16
            exp_width = 8;
            mantissa_width = 7;
            break;
        case 3: // e4m3
            exp_width = 4;
            mantissa_width = 3;
            break;
        case 4: // FP4
            exp_width = 2;
            mantissa_width = 1;
            break;
        case 5: // e5m2
            exp_width = 5;
            mantissa_width = 2;
            break;
        case 6: //e3m2
            exp_width = 3;
            mantissa_width = 2;
            break;
        case 7: //e2m3
            exp_width = 2;
            mantissa_width = 3;
            break;
        case 8: // s/uint8/4,视为特殊的浮点数(这一块还没有做)
            exp_width = 0;
            mantissa_width = 8;
            break;
        case 9: // ue8m0
            exp_width = 8;
            mantissa_width = 0;
            break;
    }

    FloatDecode result;
    DecodeExceptionInit(&result.exception); // 初始化异常信息

    if (dtype == 8) { // s/uint8/4特殊处理
        result.sign = 0;
        result.mantissa = 0;
        result.exponent = 0;
        return result;
    } else if (dtype == 9) { // ue8m0特殊处理
        result.sign = 0;
        result.mantissa = 1;

        if ((f & 0xff) == 0xff) {
            result.exponent = 0;
            result.exception.is_nan = 1; // Not a number
        } else result.exponent = (f & 0xff) - 127;
        
        return result;
    }

    int32_t bits = f;

    int32_t exp_mask = (1 << exp_width) - 1;
    int32_t exp_bias = (1 << (exp_width - 1)) - 1; // 偏移量
    int32_t mantissa_mask = (1 << mantissa_width) - 1;
    int32_t implicit_bit = 1 << mantissa_width; // 隐含的1位
    int32_t max_exp = exp_mask - exp_bias; // 最大指数值

    result.sign = (bits >> (exp_width + mantissa_width)) & 0x1;
    int32_t TempExpo = ((bits >> mantissa_width) & exp_mask) - exp_bias;
    int32_t TempMantissa = bits & mantissa_mask;

    //此时说明指数位全为0，则为非规格数
    if (TempExpo == -exp_bias) {
        //如果尾数也是0，则说明其为非规格数0
        if(TempMantissa == 0) {
            result.exponent = -exp_bias; // Zero。这里指数不能改为0，否则会出现运算错误
            result.mantissa = 0;
            if(result.sign == 1) {
                result.exception.is_nzero = 1; // Negative zero
            } else {
                result.exception.is_pzero = 1; // Positive zero
            }
        }
        //非规格数（此时的计算方法发生了改变，指数位变为2^(1-b)，而不是2^(0-b)
        else{
            //2^(1-b)
            result.exponent = -exp_bias + 1; // Denormalized exponent is -126
            result.mantissa = TempMantissa; // For denormalized numbers, mantissa is the raw bits
            result.exception.is_subnormal = 1; // If exponent is -127 and mantissa is not zero, it's a subnormal number
        }
    }
    //当指数全为1，尾数为0，此时为无穷,而类型3、4、6、7、8均没有无穷
    else if(TempExpo == max_exp && TempMantissa == 0 && dtype != 3 && dtype != 4 && dtype != 6 && dtype != 7 && dtype != 8) {
        result.exponent = max_exp; // Positive infinity
        result.mantissa = 0;
        if(result.sign == 1) {
            result.exception.is_ninf = 1; // Negative infinity
        } else {
            result.exception.is_pinf = 1; // Positive infinity
        }
    }
    //当指数全为1，尾数不为0的时候，此时为NAN，但是当为e4m3的时候，只有尾数全为0的时候才为NAN
    else if(TempExpo == max_exp && TempMantissa != 0 && (dtype != 3 || TempMantissa == mantissa_mask) && dtype != 4 && dtype != 6 && dtype != 7 && dtype != 8) {
        result.exponent = max_exp; // NaN
        result.mantissa = TempMantissa;
        result.exception.is_nan = 1; // Not a number
        // printf("it is not number\n");
    }
    //否则就是规格化的数
    else {
        result.exponent = TempExpo;
        result.mantissa = TempMantissa | implicit_bit; // For normalized numbers, add the implicit leading 1
    }
    

    #ifdef FLOATDECODE_DEBUG
        printf("decode result: sign=%d, exponent=%d, mantissa=%x\n", result.sign, result.exponent, result.mantissa);
    #endif

    return result;
}

inline FloatDecode decode_fp32(int32_t f) {
    return decode(f, 0); // 0 for FP32
}

inline FloatDecode decode_fp16(int16_t f) {
    int32_t bits = (int32_t)(*(uint16_t*)&f); // Convert int16_t to int32_t for bit manipulation
    return decode(bits, 1); // 1 for FP16
}

inline FloatDecode decode_bf16(int16_t f) {
    int32_t bits = (int32_t)(*(uint16_t*)&f); // Convert int16_t to int32_t for bit manipulation
    return decode(bits, 2); // 2 for BF16
}

inline FloatDecode decode_tf32(int32_t f) {
    FloatDecode decoded = decode(f, 0); // 3 for TF32
    decoded.mantissa >>= 13;
    return decoded;
}

inline FloatDecode decode_e4m3(int8_t f) {
    int32_t bits = (int32_t)(*(uint8_t*)&f); // Convert int8_t to int32_t for bit manipulation
    return decode(bits, 3); // 3 for MXFP8E4M3
}

inline FloatDecode decode_e5m2(int8_t f) {
    int32_t bits = (int32_t)(*(uint8_t*)&f); // Convert int8_t to int32_t for bit manipulation
    return decode(bits, 5); // 5 for MXFP8E5M2
}

inline FloatDecode decode_nvfp4(int8_t f) {
    int32_t bits = (int32_t)(*(uint8_t*)&f) & 0xf; // Convert int8_t to int32_t for bit manipulation
    return decode(bits, 4); // 4 for NVFP4
}


inline FloatDecode decode_e3m2(int8_t f) {
    int32_t bits = (int32_t)(*(uint8_t*)&f) & 0x3f; // Convert int8_t to int32_t for bit manipulation
    return decode(bits, 6); // 6 for e3m2
}

inline FloatDecode decode_e2m3(int8_t f) {
    int32_t bits = (int32_t)(*(uint8_t*)&f) & 0x3f; // Convert int8_t to int32_t for bit manipulation
    return decode(bits, 7); // 7 for e2m3
}

static int32_t fp4map[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

inline int32_t e2m1tofixed(int8_t f) {
    // e2m1到绝对值映射
    // 0bx000 -> 0 0bx001 -> 0.5 0bx010 -> 1 0bx011 -> 1.5 0bx100 -> 2 0bx101 -> 3 0bx110 -> 4 0bx111 -> 6
    // e2m1到定点数
    // 0bx000 -> 0b 0bx001 -> 1b 0bx010 -> 10b 0bx011 -> 11b 0bx100 -> 100b 0bx101 -> 110b 0bx110 -> 1000b 0bx111 -> 1100b
    int32_t bits = (int32_t)(*(uint8_t*)&f) & 0xf;
    return fp4map[bits]; 
}

inline int32_t encode_fp32(FloatDecode decoded) {
    int32_t bits = (decoded.sign << 31) | (((decoded.exponent + 127) & 0xFF) << 23) | (decoded.mantissa & 0x7FFFFF);
    return bits;
}

//先导零：二进制数从最高位开始连续出现的0的个数，从而方便左移
// 计算先导零个数
inline int count_leading_zeros(uint64_t x, int bits) {
    if (x == 0) return bits;

    int count = 0;
    for (int i = bits - 1; i >= 0; i--) {
        if ((x >> i) & 1ULL) break;
        count++;
    }
    return count;
}

// 获取异常码
// 1-正无穷
// 2-负无穷
// 3-NaN
inline int32_t get_exceptioncode(float x){
    if (isnan(x)) {
        return 3; // NaN
    } else if (isinf(x)) {
        return (x > 0) ? 1 : 2; // Positive infinity or Negative infinity
    } else {
        return 0; // No exception
    }
}

inline std::string exception_flag(const DecodeException* exception) {
    if (!exception) return "invalid";
    if (exception->is_subnormal) return "subnormal";
    if (exception->is_nan) return "nan";
    if (exception->is_pinf) return "pinf";
    if (exception->is_ninf) return "ninf";
    if (exception->is_pzero) return "pzero";
    if (exception->is_nzero) return "nzero";
    return "normal";
}
