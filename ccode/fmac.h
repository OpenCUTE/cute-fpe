#include <stdio.h>
#include "FloatDecode.h"
#include <cstdint>
// #define DECODEADD_DEBUG
// #define DECODEMUL_DEBUG
// #define FP16_DECODE_DEBUG // 已无问题，无需使用
// #define FP16MAC
// #define BF16_DECODE_DEBUG
// #define BF16MAC
// #define FP8_DECODE_DEBUG
// #define MUL_MAX_DEBUG
// #define FP4_DECODE_DEBUG


//size是数组元素数，mode是数据类型
int32_t i8mac(int8_t* a, int8_t* b, int32_t *c,int size){
    int32_t result = 0;
    for (int i = 0; i < size; i++) {
        result += (int32_t)a[i] * (int32_t)b[i];
    }
    result += *c; // 加上初始值c
    return result;
}

int32_t i8ui8mac(int8_t* a, int8_t* b, int32_t *c,int size){
    int32_t result = 0;
    for (int i = 0; i < size; i++) {
        result += (int32_t)a[i] * (int32_t)*(uint8_t *)&b[i];
    }
    result += *c; // 加上初始值c
    return result;
}

int32_t ui8i8mac(int8_t* a, int8_t* b, int32_t *c,int size){
    int32_t result = 0;
    for (int i = 0; i < size; i++) {
        result += (int32_t)*(uint8_t *)&a[i] * (int32_t)b[i];
    }
    result += *c; // 加上初始值c
    return result;
}

int32_t ui8ui8mac(int8_t* a, int8_t* b, int32_t *c,int size){
    int32_t result = 0;
    for (int i = 0; i < size; i++) {
        result += (int32_t)*(uint8_t *)&a[i] * (int32_t)*(uint8_t *)&b[i];
    }
    result += *c; // 加上初始值c
    return result;
}

int32_t decodeadd(FloatDecode *x, int size){
    FloatDecode result = {0, -126, 0};
    // 初始化result的异常信息
    DecodeExceptionInit(&result.exception);
    // printf("HELLO__________________\n");

    int32_t has_nan = 0;
    int32_t has_pinf = 0;
    int32_t has_ninf = 0;
    int32_t only_nzero = 1; // 用于检查是否只有负零
    for (int i = 0; i < size; i++) {
        if (x[i].exponent > result.exponent) {
            result.exponent = x[i].exponent;
        }
        has_nan |= x[i].exception.is_nan; // 检查是否有NaN
        has_pinf |= x[i].exception.is_pinf; // 检查是否有正无穷
        has_ninf |= x[i].exception.is_ninf; // 检查是否有负无穷
        only_nzero &= x[i].exception.is_nzero; // 检查是否只有负零
    }

    // 如果存在以上情况直接返回符合要求的值，无需对阶计算
    int32_t exception_res_bits = has_nan || has_pinf && has_ninf ? 0x7FC00000 : // NaN
                          (has_pinf ? 0x7F800000 : // 正无穷
                          (has_ninf ? 0xFF800000 : // 负无穷
                            0x80000000)); // 负零或正常值
    if(has_nan || has_pinf || has_ninf || only_nzero)
        return exception_res_bits;

    #ifdef DECODEADD_DEBUG
        printf("Max exponent: %d\n", result.exponent);
    #endif

    // 对阶处理
    for (int i = 0; i < size; i++) {
        int right_shift = (result.exponent - x[i].exponent > 31) ? 31 : (result.exponent - x[i].exponent);
        int32_t mantissa_shifted = x[i].mantissa >> right_shift;
        #ifdef DECODEADD_DEBUG
            printf("x[%d]: before shift: mantissa=%x\t after shift: mantissa_shifted=%x, right_shift=%d\n", i, x[i].mantissa, mantissa_shifted, right_shift);
        #endif
        if(x[i].sign == 0)
            result.mantissa += mantissa_shifted;
        else
            result.mantissa -= mantissa_shifted;
        #ifdef DECODEADD_DEBUG
        printf("After adding x[%d]: mantissa=%x\n", i, result.mantissa);
        #endif
    }
    if(result.mantissa < 0) {
        result.sign = 1;
        result.mantissa = -result.mantissa; // 取绝对值
    } else {
        result.sign = 0;
    }

    int32_t lz = count_leading_zeros(result.mantissa, 32);
    if (lz == 32)
        return 0; // 如果结果是零，直接返回0
    
    if (lz > 8) {
        int32_t left_shift = lz - 8; // 计算左移的位数
        result.exponent -= left_shift; // 减去左移的位数
        result.mantissa <<= left_shift; // 左移以规格化
    } else {
        result.mantissa >>= (8 - lz); // 右移以规格化
        result.exponent += (8 - lz); // 加上右移的位数
    }

    // 尝试规格化后，现在的result.mantissa只要不是全0，就都可以移位得到一个规格化的格式 1.xxx
    // result.exponent不限范围
    // 处理阶码过小导致结果为0/subnormal/阶码过大导致无穷的情况
    if(result.exponent < -149){
        result.exponent = -127;
        result.mantissa = 0; // 设置为零
    } else if(result.exponent < -126) {
        result.mantissa >>= (-result.exponent - 126); // 将尾数左
        result.exponent = -127; // 设置为非规格化数的阶码 -126? -127?
        result.exception.is_subnormal = 1; // 标记为非规格化数
    } else if(result.exponent > 127) {
        result.exponent = 128; // 设置为正无穷
        result.mantissa = 0; // 尾数为0
    }
    int32_t bits = (result.sign << 31) | ((result.exponent + 127) << 23) | (result.mantissa & 0x7FFFFF);
    return bits; // 返回结果的bits表示
}

// nvfp4的点积计算难以避免负数右移问题，所以在右边使用了3位保留位将负数右移带来的误差减少到一个比特
int32_t decodeadd_with3bit_reserve(FloatDecode *x, int size) {
    FloatDecode result = {0, -126, 0};
    // 初始化result的异常信息
    DecodeExceptionInit(&result.exception);
    // printf("HELLO__________________\n");

    int32_t has_nan = 0;
    int32_t has_pinf = 0;
    int32_t has_ninf = 0;
    int32_t only_nzero = 1; // 用于检查是否只有负零
    for (int i = 0; i < size; i++) {
        if (x[i].exponent > result.exponent) {
            result.exponent = x[i].exponent;
        }
        has_nan |= x[i].exception.is_nan; // 检查是否有NaN
        has_pinf |= x[i].exception.is_pinf; // 检查是否有正无穷
        has_ninf |= x[i].exception.is_ninf; // 检查是否有负无穷
        only_nzero &= x[i].exception.is_nzero; // 检查是否只有负零
    }

    // 如果存在以上情况直接返回符合要求的值，无需对阶计算
    int32_t exception_res_bits = has_nan || has_pinf && has_ninf ? 0x7FC00000 : // NaN
                          (has_pinf ? 0x7F800000 : // 正无穷
                          (has_ninf ? 0xFF800000 : // 负无穷
                            0x80000000)); // 负零或正常值
    if(has_nan || has_pinf || has_ninf || only_nzero)
        return exception_res_bits;

    #ifdef DECODEADD_DEBUG
        printf("Max exponent: %d\n", result.exponent);
    #endif

    int64_t mantissa_sum = 0;

    // 对阶处理
    for (int i = 0; i < size; i++) {
        int right_shift = (result.exponent - x[i].exponent > 31) ? 31 : (result.exponent - x[i].exponent);
        int64_t mantissa_shifted =  ((int64_t)x[i].mantissa << 3) >> right_shift;
        #ifdef DECODEADD_DEBUG
            printf("x[%d]: before shift: mantissa=%x\t after shift: mantissa_shifted=%lx, right_shift=%x\n", i, x[i].mantissa, mantissa_shifted, right_shift);
        #endif
        if(x[i].sign == 0)
            mantissa_sum += mantissa_shifted;
        else
            mantissa_sum -= mantissa_shifted;
        #ifdef DECODEADD_DEBUG
        printf("After adding x[%d]: mantissa=%lx\n", i, mantissa_sum);
        #endif
    }
    result.mantissa = (int32_t)((mantissa_sum >> 3) & 0xFFFFFFFF); // 右移3位还原
    #ifdef DECODEADD_DEBUG
        printf("Final mantissa after restoring 3-bit reserve: %x\n", result.mantissa);
    #endif
    if(result.mantissa < 0) {
        result.sign = 1;
        result.mantissa = -result.mantissa; // 取绝对值
    } else {
        result.sign = 0;
    }

    int32_t lz = count_leading_zeros(result.mantissa, 32);
    if (lz == 32)
        return 0; // 如果结果是零，直接返回0
    
    if (lz > 8) {
        int32_t left_shift = lz - 8; // 计算左移的位数
        result.exponent -= left_shift; // 减去左移的位数
        result.mantissa <<= left_shift; // 左移以规格化
    } else {
        result.mantissa >>= (8 - lz); // 右移以规格化
        result.exponent += (8 - lz); // 加上右移的位数
    }

    // 尝试规格化后，现在的result.mantissa只要不是全0，就都可以移位得到一个规格化的格式 1.xxx
    // result.exponent不限范围
    // 处理阶码过小导致结果为0/subnormal/阶码过大导致无穷的情况
    if(result.exponent < -149){
        result.exponent = -127;
        result.mantissa = 0; // 设置为零
    } else if(result.exponent < -126) {
        result.mantissa >>= (-result.exponent - 126); // 将尾数左
        result.exponent = -127; // 设置为非规格化数的阶码 -126? -127?
        result.exception.is_subnormal = 1; // 标记为非规格化数
    } else if(result.exponent > 127) {
        result.exponent = 128; // 设置为正无穷
        result.mantissa = 0; // 尾数为0
    }
    int32_t bits = (result.sign << 31) | ((result.exponent + 127) << 23) | (result.mantissa & 0x7FFFFF);
    return bits; // 返回结果的bits表示
}

int decodemul(FloatDecode* raw_a, FloatDecode* raw_b, int size, FloatDecode* mul_result) {
    for(int i = 0; i < size; i++){
        // 计算乘积
        mul_result[i].sign = raw_a[i].sign ^ raw_b[i].sign;
        mul_result[i].exponent = raw_a[i].exponent + raw_b[i].exponent; // 加上偏移量
        mul_result[i].mantissa = raw_a[i].mantissa * raw_b[i].mantissa; // 右移23位以适应规格化
        
        #ifdef DECODEMUL_DEBUG
            printf("mul_result[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, mul_result[i].sign, mul_result[i].exponent, mul_result[i].mantissa);
        #endif

        int32_t res_is_nan = 
            (raw_a[i].exception.is_nan || raw_b[i].exception.is_nan) || // 检查是否有NaN
            (raw_a[i].exception.is_pinf && (raw_b[i].exception.is_pzero || raw_b[i].exception.is_nzero)) ||
            (raw_b[i].exception.is_pinf && (raw_a[i].exception.is_pzero || raw_a[i].exception.is_nzero)) ||
            (raw_a[i].exception.is_ninf && (raw_b[i].exception.is_pzero || raw_b[i].exception.is_nzero)) ||
            (raw_b[i].exception.is_ninf && (raw_a[i].exception.is_pzero || raw_a[i].exception.is_nzero)); // 检查是否有无穷大和零的组合

        int32_t res_is_zero =
            !res_is_nan && // 不是nan
            (raw_a[i].exception.is_pzero || raw_b[i].exception.is_pzero ||
            raw_a[i].exception.is_nzero || raw_b[i].exception.is_nzero); // 检查是否有零值

        int32_t res_is_inf = 
            !res_is_nan && // 不是nan
            (raw_a[i].exception.is_pinf || raw_b[i].exception.is_pinf ||
            raw_a[i].exception.is_ninf || raw_b[i].exception.is_ninf); // 检查是否有无穷大

        if(res_is_nan) {
            mul_result[i].exponent = 128; // NaN
            mul_result[i].mantissa = 1; // NaN的尾数不为0
            mul_result[i].exception.is_nan = 1; // 标记为NaN
        } else if(res_is_zero) {
            mul_result[i].exponent = -127;
            mul_result[i].mantissa = 0;
            if(mul_result[i].sign == 1) {
                mul_result[i].exception.is_nzero = 1; // 负零
            } else {
                mul_result[i].exception.is_pzero = 1; // 正零
            }
        } else if(res_is_inf) {
            mul_result[i].exponent = 128; // 正无穷
            mul_result[i].mantissa = 0;
            if(mul_result[i].sign == 1) {
                mul_result[i].exception.is_ninf = 1; // 负无穷
            } else {
                mul_result[i].exception.is_pinf = 1; // 正无穷
            }
        }
    }

    return 0;
}

int32_t fp16mac(int16_t* a, int16_t* b, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_fp16(a[i]);
        raw_b[i] = decode_fp16(b[i]);
        #ifdef FP16_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

    FloatDecode *decode_mul_result = (FloatDecode*)malloc((size + 1) * sizeof(FloatDecode));
    for(int i = 0; i < size; i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(raw_a, raw_b, size, decode_mul_result);
    decode_mul_result[size] = decode_fp32(c[0]);
    
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 3; // 小数点位置和c对齐
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t bf16mac(int16_t* a, int16_t* b, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_bf16(a[i]);
        raw_b[i] = decode_bf16(b[i]);
        #ifdef FP16_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

    FloatDecode *decode_mul_result = (FloatDecode*)malloc((size + 1) * sizeof(FloatDecode));
    for(int i = 0; i < size; i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(raw_a, raw_b, size, decode_mul_result);
    decode_mul_result[size] = decode_fp32(c[0]); // 添加一个零元素用于累加
    
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 9; // 小数点位置和c对齐
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t mxfp8e4m3mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int j = 0; j < size / 32; j++) {
        int32_t a_scale_int = (int32_t)(*(a_scale + j));
        int32_t b_scale_int = (int32_t)(*(b_scale + j));
        for (int i = 0; i < 32; i++) {
            raw_a[j * 32 + i] = decode_e4m3(a[j * 32 + i]);
            raw_a[j * 32 + i].exponent += ((a_scale_int & 0b11111111) - 127); // 加上scale
            raw_b[j * 32 + i] = decode_e4m3(b[j * 32 + i]);
            raw_b[j * 32 + i].exponent += ((b_scale_int & 0b11111111) - 127); // 加上scale
            #ifdef FP8_DECODE_DEBUG
                printf("a_scale_int & 0x11111111:%x\n", (a_scale_int & 0x11111111));
                printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", j * 32 + i, raw_a[j * 32 + i].sign, raw_a[j * 32 + i].exponent, raw_a[j * 32 + i].mantissa);
                printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", j * 32 + i, raw_b[j * 32 + i].sign, raw_b[j * 32 + i].exponent, raw_b[j * 32 + i].mantissa);
            #endif
        }
    }

    FloatDecode *decode_mul_result = (FloatDecode*)malloc((size + 1) * sizeof(FloatDecode));
    for(int i = 0; i < size; i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(raw_a, raw_b, size, decode_mul_result);
    decode_mul_result[size] = decode_fp32(c[0]); // 添加一个零元素用于累加
    
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 17; // 小数点位置和c对齐
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t fp8e4m3mac(int8_t* a, int8_t* b, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_e4m3(a[i]);
        raw_b[i] = decode_e4m3(b[i]);
        #ifdef FP8_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

    FloatDecode *decode_mul_result = (FloatDecode*)malloc((size + 1) * sizeof(FloatDecode));
    for(int i = 0; i < size; i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(raw_a, raw_b, size, decode_mul_result);
    decode_mul_result[size] = decode_fp32(c[0]); // 添加一个零元素用于累加
    
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 17; // 小数点位置和c对齐
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t mxfp8e5m2mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int j = 0; j < size / 32; j++) {
        int32_t a_scale_int = (int32_t)(*(a_scale + j));
        int32_t b_scale_int = (int32_t)(*(b_scale + j));
        for (int i = 0; i < 32; i++) {
            raw_a[j * 32 + i] = decode_e5m2(a[j * 32 + i]);
            raw_a[j * 32 + i].exponent += ((a_scale_int & 0b11111111) - 127); // 加上scale
            raw_b[j * 32 + i] = decode_e5m2(b[j * 32 + i]);
            raw_b[j * 32 + i].exponent += ((b_scale_int & 0b11111111) - 127); // 加上scale
            #ifdef FP8_DECODE_DEBUG
                printf("a_scale_int & 0x11111111:%x\n", (a_scale_int & 0x11111111));
                printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", j * 32 + i, raw_a[j * 32 + i].sign, raw_a[j * 32 + i].exponent, raw_a[j * 32 + i].mantissa);
                printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", j * 32 + i, raw_b[j * 32 + i].sign, raw_b[j * 32 + i].exponent, raw_b[j * 32 + i].mantissa);
            #endif
        }
    }

    FloatDecode *decode_mul_result = (FloatDecode*)malloc((size + 1) * sizeof(FloatDecode));
    for(int i = 0; i < size; i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(raw_a, raw_b, size, decode_mul_result);
    decode_mul_result[size] = decode_fp32(c[0]); // 添加一个零元素用于累加
    
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 19; // 小数点位置和c对齐
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t fp8e5m2mac(int8_t* a, int8_t* b, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_e5m2(a[i]);
        raw_b[i] = decode_e5m2(b[i]);
        #ifdef FP8_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

    FloatDecode *decode_mul_result = (FloatDecode*)malloc((size + 1) * sizeof(FloatDecode));
    for(int i = 0; i < size; i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(raw_a, raw_b, size, decode_mul_result);
    decode_mul_result[size] = decode_fp32(c[0]); // 添加一个零元素用于累加
    
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 19; // 小数点位置和c对齐
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t nvfp4mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {

    int block_num = size / 8;
    int32_t *block_dot = (int32_t*)malloc((block_num) * sizeof(int32_t));
    FloatDecode *scale_a = (FloatDecode*)malloc(block_num * sizeof(FloatDecode));
    FloatDecode *scale_b = (FloatDecode*)malloc(block_num * sizeof(FloatDecode));

    for (int j = 0; j < block_num; j++) {
        block_dot[j] = 0;
        scale_a[j] = decode_e4m3(a_scale[j]);
        scale_b[j] = decode_e4m3(b_scale[j]);
        if (scale_a[j].sign == 1) {
            scale_a[j].mantissa = -scale_a[j].mantissa;
            scale_a[j].sign = 0;
        }
        if (scale_b[j].sign == 1) {
            scale_b[j].mantissa = -scale_b[j].mantissa;
            scale_b[j].sign = 0;
        }
        for (int i = 0; i < 8; i++) {
            block_dot[j] += e2m1tofixed(a[j * 8 + i]) * e2m1tofixed(b[j * 8 + i]);
            block_dot[j] += e2m1tofixed(a[j * 8 + i] >> 4) * e2m1tofixed(b[j * 8 + i] >> 4);
        }
        #ifdef FP4_DECODE_DEBUG
        printf("a_scale:%x, b_scale:%x\n", a_scale[j], b_scale[j]);
        printf("block_dot[%d]:%x\n", j, block_dot[j]);
        #endif
    }

    FloatDecode *decode_mul_result = (FloatDecode*)malloc((block_num + 1) * sizeof(FloatDecode));
    for(int i = 0; i < (block_num + 1); i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(scale_a, scale_b, block_num, decode_mul_result);
    decode_mul_result[block_num] = decode_fp32(c[0]); // 添加一个零元素用于累加
    #ifdef FP4_DECODE_DEBUG
    for (int i = 0; i < (block_num + 1); i++) {
        
            printf("decode_mul_result[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, decode_mul_result[i].sign, decode_mul_result[i].exponent, decode_mul_result[i].mantissa);
        
    }
    #endif

    for (int i = 0; i < block_num; i++) {
        decode_mul_result[i].mantissa *= block_dot[i];
        decode_mul_result[i].exponent += 9;
        // if (decode_mul_result[i].mantissa < 0) {
        //     decode_mul_result[i].mantissa = -decode_mul_result[i].mantissa; 
        //     decode_mul_result[i].sign = 1 - decode_mul_result[i].sign;
        // }
        decode_mul_result[i].mantissa <<= 6; 
    }
    #ifdef FP4_DECODE_DEBUG
    for (int i = 0; i < block_num + 1; i++) {
        
            printf("final decode_mul_result[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, decode_mul_result[i].sign, decode_mul_result[i].exponent, decode_mul_result[i].mantissa);
        
    }
    #endif

    // 将结果转换为float
    int32_t bits = decodeadd_with3bit_reserve(decode_mul_result, block_num + 1); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t mxfp4mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {
    int block_num = size / 8;     // 32个fp4一个block,硬件实现按16个为一组

    int32_t *block_dot = (int32_t*)malloc((block_num) * sizeof(int32_t));
    FloatDecode *scale_a = (FloatDecode*)malloc(block_num * sizeof(FloatDecode));
    FloatDecode *scale_b = (FloatDecode*)malloc(block_num * sizeof(FloatDecode));

    for (int j = 0; j < block_num; j++) {
        block_dot[j] = 0;
        DecodeExceptionInit(&scale_a[j].exception);
        DecodeExceptionInit(&scale_b[j].exception);
        scale_a[j].sign = 0;
        scale_b[j].sign = 0;
        scale_a[j].mantissa = 1;
        scale_b[j].mantissa = 1;
        scale_a[j].exception.is_nan = a_scale[j/2] == (int8_t)0xFF ? 1 : 0;
        scale_b[j].exception.is_nan = b_scale[j/2] == (int8_t)0xFF ? 1 : 0;
        scale_a[j].exponent = (((int32_t)a_scale[j/2]) & 0x000000ff) - 127;
        scale_b[j].exponent = (((int32_t)b_scale[j/2]) & 0x000000ff) - 127;
        for (int i = 0; i < 8; i++) {
            block_dot[j] += e2m1tofixed(a[j * 8 + i]) * e2m1tofixed(b[j * 8 + i]);
            block_dot[j] += e2m1tofixed(a[j * 8 + i] >> 4) * e2m1tofixed(b[j * 8 + i] >> 4);
        }
        #ifdef FP4_DECODE_DEBUG
        printf("scale_a[%d]: exponent=%d, scale_b[%d]: exponent=%d\n", j, scale_a[j].exponent, j, scale_b[j].exponent);
        printf("a_scale:%x, b_scale:%x\n", a_scale[j/2], b_scale[j/2]);
        printf("block_dot[%d]:%x\n", j, block_dot[j]);
        #endif
    }

    FloatDecode *decode_mul_result = (FloatDecode*)malloc((block_num + 1) * sizeof(FloatDecode));
    for(int i = 0; i < (block_num + 1); i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(scale_a, scale_b, block_num, decode_mul_result);
    #ifdef FP4_DECODE_DEBUG
    for (int i = 0; i < block_num; i++) {
        
            printf("decode_mul_result[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, decode_mul_result[i].sign, decode_mul_result[i].exponent, decode_mul_result[i].mantissa);
        
    }
    #endif
    decode_mul_result[block_num] = decode_fp32(c[0]); // 添加一个零元素用于累加

    for (int i = 0; i < block_num; i++) {
        decode_mul_result[i].mantissa *= block_dot[i];
        decode_mul_result[i].exponent += 6;
        // if (decode_mul_result[i].mantissa < 0) {
        //     decode_mul_result[i].mantissa = -decode_mul_result[i].mantissa; 
        //     decode_mul_result[i].sign = 1 - decode_mul_result[i].sign;
        // }
        decode_mul_result[i].mantissa <<= 15; 
    }
    #ifdef FP4_DECODE_DEBUG
    for (int i = 0; i < block_num + 1; i++) {
        
            printf("final decode_mul_result[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, decode_mul_result[i].sign, decode_mul_result[i].exponent, decode_mul_result[i].mantissa);
        
    }
    #endif

    // 将结果转换为float
    int32_t bits = decodeadd_with3bit_reserve(decode_mul_result, block_num + 1); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t tf32mac(int32_t* a, int32_t* b, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_tf32(a[i]);
        raw_b[i] = decode_tf32(b[i]);
        #ifdef FP16_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

    FloatDecode *decode_mul_result = (FloatDecode*)malloc((size + 1) * sizeof(FloatDecode));
    for(int i = 0; i < size; i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(raw_a, raw_b, size, decode_mul_result);
    decode_mul_result[size] = decode_fp32(c[0]); // 添加一个零元素用于累加
    
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 3; // 小数点位置和c对齐
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t mymac(void *a, void *b, void *c, int size, int dtype) {
    if (dtype == 0) { // int8_t
        return i8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 1) { // int16_t
        return fp16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size);
    } else if (dtype == 2) { // int32_t
        return bf16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size);
    } else if (dtype == 3) { // bf16
        return tf32mac((int32_t*)a, (int32_t*)b, (int32_t*)c, size);
    } else if (dtype == 4) { // tf32
        return i8ui8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 5) { // tf32
        return ui8i8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 6) { // tf32
        return ui8ui8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 11) { // fp8e4m3
        return fp8e4m3mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 12) { // fp8e5m2
        return fp8e5m2mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    }
    return 0; // 错误处理
}

int32_t mymac_scale(void *a, void *a_scale, void *b, void *b_scale, void *c, int size, int dtype) {
    if (dtype == 7) { // mxfp8e4m3
        return mxfp8e4m3mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size);
    } else if (dtype == 8) { // mxfp8e5m2
        return mxfp8e5m2mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size);
    } else if (dtype == 9) { // nvfp4
        return nvfp4mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size);
    } else if (dtype == 10) { // mxfp4
        return mxfp4mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size);
    }
    return 0;
}

float my_float_abs(float x) {
    return (x < 0) ? -x : x;
}

float my_double_abs(double x) {
    return (x < 0) ? -x : x;
}