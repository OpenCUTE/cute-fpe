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

// #define DECODEADD_FP16_DEBUG


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


//做一个不带scale(4X)的通用累加处理，要求保证输入尾数的数据格式为x.y，其中y为25个小数位。
//需要注意的是，我们不会输出负0
//float_type == 0的时候目标格式为fp32；float_type == 1的时候目标格式为fp16
int32_t decodeadd(FloatDecode *x, int size, int decimal_num, bool float_type = 0){
    FloatDecode result = {0, -126, 0};
    // 初始化result的异常信息
    DecodeExceptionInit(&result.exception);
    // printf("HELLO__________________\n");

    int32_t has_nan = 0;
    int32_t has_pinf = 0;
    int32_t has_ninf = 0;
    int32_t only_nzero = 1; // 用于检查是否只有负零
    
    //找出其中的最大指数，并判断其中是否存在特殊值
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
    int32_t exception_res_bits ; 
    if(!float_type){
        exception_res_bits = has_nan || has_pinf && has_ninf ? 0x7FFFFFFF : // NaN
                             (has_pinf ? 0x7F800000 : // 正无穷
                             (has_ninf ? 0xFF800000 : // 负无穷
                              0x00000000)); // 负零或正常值
    }
    else{
        exception_res_bits = has_nan || has_pinf && has_ninf ? 0x00007FFF : // NaN
                             (has_pinf ? 0x00007C00 : // 正无穷
                             (has_ninf ? 0x0000FC00 : // 负无穷
                             0x00000000)); // 负零或正常值
    }
                            
    if(has_nan || has_pinf || has_ninf || only_nzero)
        return exception_res_bits;

    #ifdef DECODEADD_DEBUG
        printf("Max exponent: %d\n", result.exponent);
    #endif

    // int64_t mantissa_sum = 0;

    #ifdef DECODEADD_DEBUG
        printf("Before adding: mantissa=%x\n", result.mantissa);
    #endif

    //进行尾数对阶处理
    for (int i = 0; i < size; i++) {
        int right_shift = (result.exponent - x[i].exponent > 63) ? 63 : (result.exponent - x[i].exponent);
        int64_t mantissa_shifted = x[i].mantissa >> right_shift;
        #ifdef DECODEADD_DEBUG
            printf("x[%d]: before shift: mantissa=%x\t after shift: mantissa_shifted=%x, right_shift=%d\n", i, x[i].mantissa, mantissa_shifted, right_shift);
        #endif
        if(x[i].sign == 0){
            result.mantissa += mantissa_shifted;
        }
        else{
            result.mantissa -= mantissa_shifted;
        }
        // #ifdef DECODEADD_DEBUG
        // printf("After adding: mantissa=%x\n", result.mantissa);
        // #endif
    }

    #ifdef DECODEADD_DEBUG
        printf("After adding: mantissa=%x\n", result.mantissa);
    #endif


    if(result.mantissa < 0) {
        result.sign = 1;
        result.mantissa = -result.mantissa; // 取绝对值
    } else {
        result.sign = 0;
    }

    int32_t lz = count_leading_zeros(result.mantissa, 64);

    if (lz == 64)
        return 0; // 如果结果是零，直接返回0

    //存储被右移走的数据，从而在后面进行做舍入判断
    uint64_t lost_bits = 0;
    //因此此时我们内部的精度是25，因此我们查看先导0的个数来判断
    int shift_num = (63 - decimal_num);
    if (lz > shift_num) {
        int32_t left_shift = lz - shift_num; // 计算左移的位数
        result.exponent -= left_shift; // 减去左移的位数
        result.mantissa <<= left_shift; // 左移以规格化
    } else {
        int shift = shift_num - lz;

        lost_bits = result.mantissa & ((1ULL << shift) - 1); 
        result.mantissa >>= (shift_num - lz); // 右移以规格化
        result.exponent += (shift_num - lz); // 加上右移的位数
    }

    #ifdef DECODEADD_DEBUG
        printf("After shift: mantissa=%x\n", result.mantissa);
        printf("After shift : exponent=%d\n", result.exponent);
    #endif

    int32_t bits;

    //此时根据float_type进行目标个数的输出
    if(!float_type){   //目标浮点数：fp32，其舍入方式是RZ
        //还原成fp32的23位小数
        result.mantissa >>= (decimal_num - 23);

        // 尝试规格化后，现在的result.mantissa只要不是全0，就都可以移位得到一个规格化的格式 1.xxx
        // result.exponent不限范围
        // 处理阶码过小导致结果为0/subnormal/阶码过大导致无穷的情况
        if(result.exponent < -149){
            result.exponent = -127;
            result.mantissa = 0; // 设置为零
            result.sign = 0; //只有正0
        } else if(result.exponent < -126) {//非规格数
            result.mantissa >>= (-result.exponent - 126); // 将尾数左
            result.exponent = -127; // 设置为非规格化数的阶码
            result.exception.is_subnormal = 1; // 标记为非规格化数
        } else if(result.exponent > 127) {
            result.exponent = 128; // 设置为无穷
            result.mantissa = 0; // 尾数为0
        }

        bits = (result.sign << 31) | ((result.exponent + 127) << 23) | (result.mantissa & 0x7FFFFF);
    }
    else{   //fp16，其舍入方式是RNE

        //先判断是否为非规格化数，然后再进行舍入，再进行是否无穷的判断
        uint32_t mant = uint32_t(result.mantissa);

        #ifdef DECODEADD_FP16_DEBUG
            printf("before mant   = 0x%08x\n", mant);
        #endif

        //保留如果是非规格化数据的话，我们保留这个移出的数据
        uint32_t lost_bits_2 = 0;
        //判断是否为一个极小数据，或者非规格化数
        if(result.exponent < -26){
            result.exponent = -15;
            mant = 0; // 设置为零
            result.sign = 0; //只有正0
        } else if(result.exponent < -14) {//非规格数
            lost_bits_2 = mant & ((1u << (-result.exponent - 14)) - 1);
            mant >>= (-result.exponent - 14); // 将尾数左
            result.exponent = -15; // 设置为非规格化数的阶码
            result.exception.is_subnormal = 1; // 标记为非规格化数
        }

        // printf("mant=%x\n", mant);
        #ifdef DECODEADD_FP16_DEBUG
            printf("after mant   = 0x%08x\n", mant);
        #endif
        
        //保留有效尾数
        uint32_t mant_fp16 = (mant >> 15) & 0x7FF;

        // printf("mant_fp16=%x\n", mant_fp16);
        //保留被移走的数据
        uint32_t remainder = mant & 0x7FFF;

        //观察有效位数的下一位是不是1
        uint32_t guard = (remainder >> 14) & 1;
        uint32_t rest  = remainder & 0x3FFF;

        #ifdef DECODEADD_FP16_DEBUG
            printf("guard      = 0x%08x\n", guard);
            printf("rest       = 0x%08x\n", rest);
            printf("remainder  = 0x%08x\n", remainder);
            printf("lost_bits  = 0x%08x\n", lost_bits);
            printf("lost_bits2 = 0x%08x\n", lost_bits_2);
            printf("mant_fp16  = 0x%08x\n", mant_fp16);
        #endif
        // 打印所有关键变量

        if (guard && (rest || lost_bits || lost_bits_2 || (mant_fp16 & 1))) {
            mant_fp16++;
        }
        if (mant_fp16 == 2048 && !result.exception.is_subnormal) {
            mant_fp16 = 1024;
            result.exponent++;
        } else if(mant_fp16 == 1024 && result.exception.is_subnormal){
            mant_fp16 = 0;
            result.exponent++;
        }

        #ifdef DECODEADD_FP16_DEBUG
            printf("mant_fp16        = 0x%08x\n", mant_fp16);
            printf("result.exponent  = 0d%d\n", result.exponent);
        #endif

        //处理异常值
        if(result.exponent < -25){
            result.exponent = -15;
            mant_fp16 = 0; // 设置为零
            result.sign = 0; //只有正0
        } 
        // else if(result.exponent < -14) {//非规格数
        //     mant_fp16 >>= (-result.exponent - 14); // 将尾数左
        //     result.exponent = -15; // 设置为非规格化数的阶码
        //     result.exception.is_subnormal = 1; // 标记为非规格化数
        // } 
        else if(result.exponent > 15) {
            result.exponent = 16; // 设置为无穷
            mant_fp16 = 0; // 尾数为0
        } else {
            mant_fp16 = mant_fp16 & 0x3FF;
        }
        
        bits = (result.sign << 15) | ((result.exponent + 15) << 10) | mant_fp16 ;
    }

    return bits; // 返回结果的bits表示
}

//定点数累加
FloatDecode fixed_add(FloatDecode *x, int size){
    FloatDecode result = {0, -126, 0};
    //因为这一步仅仅是e2m1中存在，所以不存在特殊值，不需要进行判断
    //直接累加
    for (int i = 0; i < size; i++) {
        if(x[i].sign == 0)
            result.mantissa += x[i].mantissa;
        else
            result.mantissa -= x[i].mantissa;
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

    return result;
}

//我们是定点数乘积，即指数直接相加，尾数直接相乘，不进行归一化
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
            // printf("it is not number\n");
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

//float_type == 0的时候目标格式为fp32；float_type == 1的时候目标格式为fp16
int32_t fp16mac(int16_t* a, int16_t* b, int32_t *c, int size, bool float_type = 0) {
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


    if(!float_type){
        decode_mul_result[size] = decode_fp32(c[0]);
    }
    else {
        decode_mul_result[size] = decode_fp16((int16_t)c[0]);
    }

    // 小数点位置和25位小数位对齐
    //根据C的输入格式来进行小数位的对齐
    if(!float_type)
        decode_mul_result[size].mantissa <<= 2;
    else
        decode_mul_result[size].mantissa <<= 15;

    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 5;
    }

    // printf("decode_mul_result[0]: sign=%d, exponent=%d, mantissa=%x\n", decode_mul_result[0].sign, decode_mul_result[0].exponent, decode_mul_result[0].mantissa);


    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25, float_type); // 将初始值c和计算结果相加
    
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
    
    // 小数点位置和25位小数位对齐
    decode_mul_result[size].mantissa <<= 2;

    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 11;
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t mxfp8e4m3mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int j = 0; j < size / 32; j++) {
        uint8_t a_scale_u8 = (uint8_t)a_scale[j];
        uint8_t b_scale_u8 = (uint8_t)b_scale[j];

        bool a_scale_is_nan = (a_scale_u8 == 0xFF);
        bool b_scale_is_nan = (b_scale_u8 == 0xFF);

        int32_t a_scale_int = (int32_t)a_scale_u8;
        int32_t b_scale_int = (int32_t)b_scale_u8;

        for (int i = 0; i < 32; i++) {
            int idx = j * 32 + i;
            // decode
            raw_a[idx] = decode_e4m3(a[idx]);
            raw_b[idx] = decode_e4m3(b[idx]);

            //如果 scale 是 NaN，直接标记
            if (a_scale_is_nan) {
                raw_a[idx].exception.is_nan = 1;
            } else {
                raw_a[idx].exponent += (a_scale_int - 127);
            }
            if (b_scale_is_nan) {
                raw_b[idx].exception.is_nan = 1;
            } else {
                raw_b[idx].exponent += (b_scale_int - 127);
            }
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
    
    // 小数点位置和25位小数位对齐
    decode_mul_result[size].mantissa <<= 2;
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 19;
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t fp8e4m3mac(int8_t* a, int8_t* b, int32_t *c, int size, bool float_type = 0) {
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

    if(!float_type){
        decode_mul_result[size] = decode_fp32(c[0]);
    }
    else {
        decode_mul_result[size] = decode_fp16((int16_t)c[0]);
    }
    
    // 小数点位置和25位小数位对齐
    //根据C的输入格式来进行小数位的对齐
    if(!float_type)
        decode_mul_result[size].mantissa <<= 2;
    else
        decode_mul_result[size].mantissa <<= 15;

    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 19;
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25, float_type); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t mxfp8e5m2mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int j = 0; j < size / 32; j++) {
        uint8_t a_scale_u8 = (uint8_t)a_scale[j];
        uint8_t b_scale_u8 = (uint8_t)b_scale[j];

        bool a_scale_is_nan = (a_scale_u8 == 0xFF);
        bool b_scale_is_nan = (b_scale_u8 == 0xFF);

        int32_t a_scale_int = (int32_t)a_scale_u8;
        int32_t b_scale_int = (int32_t)b_scale_u8;

        for (int i = 0; i < 32; i++) {
            int idx = j * 32 + i;
            // decode
            raw_a[idx] = decode_e5m2(a[idx]);
            raw_b[idx] = decode_e5m2(b[idx]);

            //如果 scale 是 NaN，直接标记
            if (a_scale_is_nan) {
                raw_a[idx].exception.is_nan = 1;
            } else {
                raw_a[idx].exponent += (a_scale_int - 127);
            }
            if (b_scale_is_nan) {
                raw_b[idx].exception.is_nan = 1;
            } else {
                raw_b[idx].exponent += (b_scale_int - 127);
            }
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
    

    // 小数点位置和25位小数位对齐
    decode_mul_result[size].mantissa <<= 2;
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 21;
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t fp8e5m2mac(int8_t* a, int8_t* b, int32_t *c, int size, bool float_type = 0) {
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

    if(!float_type){
        decode_mul_result[size] = decode_fp32(c[0]);
    }
    else {
        decode_mul_result[size] = decode_fp16((int16_t)c[0]);
    }
    
    // 小数点位置和25位小数位对齐
    //根据C的输入格式来进行小数位的对齐
    if(!float_type)
        decode_mul_result[size].mantissa <<= 2;
    else
        decode_mul_result[size].mantissa <<= 15;

    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 21;
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25, float_type); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t fp6e3m2mac(int8_t* a, int8_t* b, int32_t *c, int size, bool float_type = 0) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_e3m2(a[i]);
        raw_b[i] = decode_e3m2(b[i]);
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

    if(!float_type){
        decode_mul_result[size] = decode_fp32(c[0]);
    }
    else {
        decode_mul_result[size] = decode_fp16((int16_t)c[0]);
    }
    
    // 小数点位置和25位小数位对齐
    //根据C的输入格式来进行小数位的对齐
    if(!float_type)
        decode_mul_result[size].mantissa <<= 2;
    else
        decode_mul_result[size].mantissa <<= 15;

    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 21;
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25, float_type); // 将初始值c和计算结果相加
    
    return bits;
}


int32_t fp6e2m3mac(int8_t* a, int8_t* b, int32_t *c, int size, bool float_type = 0) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_e2m3(a[i]);
        raw_b[i] = decode_e2m3(b[i]);
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

    if(!float_type){
        decode_mul_result[size] = decode_fp32(c[0]);
    }
    else {
        decode_mul_result[size] = decode_fp16((int16_t)c[0]);
    }
    
    // 小数点位置和25位小数位对齐
    //根据C的输入格式来进行小数位的对齐
    if(!float_type)
        decode_mul_result[size].mantissa <<= 2;
    else
        decode_mul_result[size].mantissa <<= 15;

    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 19;
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25, float_type); // 将初始值c和计算结果相加
    
    return bits;
}


int32_t mxfp6e3m2mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int j = 0; j < size / 32; j++) {
        uint8_t a_scale_u8 = (uint8_t)a_scale[j];
        uint8_t b_scale_u8 = (uint8_t)b_scale[j];

        bool a_scale_is_nan = (a_scale_u8 == 0xFF);
        bool b_scale_is_nan = (b_scale_u8 == 0xFF);

        int32_t a_scale_int = (int32_t)a_scale_u8;
        int32_t b_scale_int = (int32_t)b_scale_u8;

        for (int i = 0; i < 32; i++) {
            int idx = j * 32 + i;
            // decode
            raw_a[idx] = decode_e3m2(a[idx]);
            raw_b[idx] = decode_e3m2(b[idx]);

             //如果 scale 是 NaN，直接标记
            if (a_scale_is_nan) {
                raw_a[idx].exception.is_nan = 1;
            } else {
                raw_a[idx].exponent += (a_scale_int - 127);
            }
            if (b_scale_is_nan) {
                raw_b[idx].exception.is_nan = 1;
            } else {
                raw_b[idx].exponent += (b_scale_int - 127);
            }
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
    
    // 小数点位置和25位小数位对齐
    decode_mul_result[size].mantissa <<= 2;
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 21;
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25); // 将初始值c和计算结果相加
    
    return bits;
}

//这里即可以作为e2m3的矩阵运算，也可以当成e2m1的矩阵运算
int32_t mxfp6e2m3mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {
    FloatDecode *raw_a = (FloatDecode*)malloc(size * sizeof(FloatDecode));
    FloatDecode *raw_b = (FloatDecode*)malloc(size * sizeof(FloatDecode));

    for (int j = 0; j < size / 32; j++) {
        uint8_t a_scale_u8 = (uint8_t)a_scale[j];
        uint8_t b_scale_u8 = (uint8_t)b_scale[j];

        bool a_scale_is_nan = (a_scale_u8 == 0xFF);
        bool b_scale_is_nan = (b_scale_u8 == 0xFF);

        int32_t a_scale_int = (int32_t)a_scale_u8;
        int32_t b_scale_int = (int32_t)b_scale_u8;

        for (int i = 0; i < 32; i++) {
            int idx = j * 32 + i;
            // decode
            raw_a[idx] = decode_e2m3(a[idx]);
            raw_b[idx] = decode_e2m3(b[idx]);

            //如果 scale 是 NaN，直接标记
            if (a_scale_is_nan) {
                raw_a[idx].exception.is_nan = 1;
            } else {
                raw_a[idx].exponent += (a_scale_int - 127);
            }
            if (b_scale_is_nan) {
                raw_b[idx].exception.is_nan = 1;
            } else {
                raw_b[idx].exponent += (b_scale_int - 127);
            }

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
    
    // 小数点位置和25位小数位对齐
    decode_mul_result[size].mantissa <<= 2;
    for(int i = 0; i < size; i ++){
        decode_mul_result[i].mantissa <<= 19;
    }

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25); // 将初始值c和计算结果相加
    
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
            scale_a[j].sign = 0;
        }
        if (scale_b[j].sign == 1) {
            scale_b[j].sign = 0;
        }
        for (int i = 0; i < 8; i++) {
            block_dot[j] += e2m1tofixed(a[j * 8 + i]) * e2m1tofixed(b[j * 8 + i]);
            block_dot[j] += e2m1tofixed(a[j * 8 + i] >> 4) * e2m1tofixed(b[j * 8 + i] >> 4);
        }
        #ifdef FP4_DECODE_DEBUG
        printf("a_scale:%x, b_scale:%x\n", a_scale[j], b_scale[j]);
        printf("block_dot[%d]:%x\n", j, block_dot[j]);
        printf("scale_a[%d]: sign=%d, exponent=%d, mantissa=%x\n", j, scale_a[j].sign, scale_a[j].exponent, scale_a[j].mantissa);
        printf("scale_b[%d]: sign=%d, exponent=%d, mantissa=%x\n", j, scale_b[j].sign, scale_b[j].exponent, scale_b[j].mantissa);
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

    //因为我们知道前面的e2m1作为一个定点数之后，其相当于存在一位小数，然后两个e2m1相乘则相当于有两位小数
    //然后我们存在两个e4m3的scale，则这两个相乘又产生了6个小数位，则一共存在8个小数位，此时我们需要把它对齐到fp32中
    //所以我们一共需要将其左移23-8 == 15位，但是此时全部移位就会导致最上面的数据可能被移出（？）所以我们采用移位+指数增加的方法进行计算
    //指数上加9，尾数上左移6位，则这相当于一共左移了15位
    
    //此时我们需要对齐35位小数位
    for (int i = 0; i < block_num; i++) {
        if(block_dot[i] < 0){
            block_dot[i] = - block_dot[i];
            decode_mul_result[i].sign = 1;
        }
        decode_mul_result[i].mantissa *= block_dot[i];
        // decode_mul_result[i].exponent += 9;
        // // if (decode_mul_result[i].mantissa < 0) {
        // //     decode_mul_result[i].mantissa = -decode_mul_result[i].mantissa; 
        // //     decode_mul_result[i].sign = 1 - decode_mul_result[i].sign;
        // // }
        decode_mul_result[i].mantissa <<= 27; 
    }
    decode_mul_result[block_num].mantissa <<= 12;

    #ifdef FP4_DECODE_DEBUG
    for (int i = 0; i < block_num + 1; i++) {
        
            printf("final decode_mul_result[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, decode_mul_result[i].sign, decode_mul_result[i].exponent, decode_mul_result[i].mantissa);
        
    }
    #endif

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, block_num + 1, 35); // 将初始值c和计算结果相加
    
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
        //一共分为四组，每两组共享一个scale
        scale_a[j].exception.is_nan = a_scale[j/2] == (int8_t)0xFF ? 1 : 0;
        scale_b[j].exception.is_nan = b_scale[j/2] == (int8_t)0xFF ? 1 : 0;
        scale_a[j].exponent = (((int32_t)a_scale[j/2]) & 0x000000ff) - 127;
        scale_b[j].exponent = (((int32_t)b_scale[j/2]) & 0x000000ff) - 127;
        for (int i = 0; i < 8; i++) {
            int idx = j * 8 + i;

            // 低4bit
            int a_low = e2m1tofixed(a[idx]);
            int b_low = e2m1tofixed(b[idx]);
            int prod_low = a_low * b_low;

            block_dot[j] += prod_low;

            // printf("[j=%d i=%d LOW ] a=%d b=%d prod=%d sum=%d\n",
            //     j, i, a_low, b_low, prod_low, block_dot[j]);

            // 高4bit
            int a_high = e2m1tofixed(a[idx] >> 4);
            int b_high = e2m1tofixed(b[idx] >> 4);
            int prod_high = a_high * b_high;

            block_dot[j] += prod_high;

            // printf("[j=%d i=%d HIGH] a=%d b=%d prod=%d sum=%d\n",
            //     j, i, a_high, b_high, prod_high, block_dot[j]);
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
        if(block_dot[i] < 0){
            block_dot[i] = - block_dot[i];
            decode_mul_result[i].sign = 1;
        }
        decode_mul_result[i].mantissa *= block_dot[i];
        // decode_mul_result[i].exponent += 6;
        // if (decode_mul_result[i].mantissa < 0) {
        //     decode_mul_result[i].mantissa = -decode_mul_result[i].mantissa; 
        //     decode_mul_result[i].sign = 1 - decode_mul_result[i].sign;
        // }
        // decode_mul_result[i].mantissa <<= 15; 
        decode_mul_result[i].mantissa <<= 33; 
    }
    decode_mul_result[block_num].mantissa <<= 12;
    #ifdef FP4_DECODE_DEBUG
    for (int i = 0; i < block_num + 1; i++) {
        
            printf("final decode_mul_result[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, decode_mul_result[i].sign, decode_mul_result[i].exponent, decode_mul_result[i].mantissa);
        
    }
    #endif

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, block_num + 1, 35); // 将初始值c和计算结果相加
    
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
        decode_mul_result[i].mantissa <<= 5; // 小数点位置和c对齐
    }
    decode_mul_result[size].mantissa <<= 2;

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, size + 1, 25); // 将初始值c和计算结果相加
    
    return bits;
}

int32_t mymac(void *a, void *b, void *c, int size, int dtype, bool float_type = 0) {
    if (dtype == 0) { // int8_t
        return i8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 1) { // int16_t
        return fp16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size, float_type);
    } else if (dtype == 2) { // int32_t
        return bf16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size);
    } else if (dtype == 3) { // bf16
        return tf32mac((int32_t*)a, (int32_t*)b, (int32_t*)c, size);
    } else if (dtype == 4) { // int8*uint8
        return i8ui8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 5) { // uint8*int8
        return ui8i8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 6) { // uint8*uint8
        return ui8ui8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 13) { // fp8e4m3
        return fp8e4m3mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, float_type);
    } else if (dtype == 14) { // fp8e5m2
        return fp8e5m2mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, float_type);
    } else if (dtype == 15 || (dtype == 18)) { // fp6e2m3 or fp4e2m1
        return fp6e2m3mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, float_type);
    } else if (dtype == 16) { // fp6e3m2
        return fp6e3m2mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size, float_type);
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
    } else if (dtype == 11 || (dtype == 17)) { // mxfp6e2m3 or fp4e2m1
        return mxfp6e2m3mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size);
    } else if (dtype == 12) { // mxfp6e3m2
        return mxfp6e3m2mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size);
    }
    return 0;
}

float my_float_abs(float x) {
    return (x < 0) ? -x : x;
}

float my_double_abs(double x) {
    return (x < 0) ? -x : x;
}