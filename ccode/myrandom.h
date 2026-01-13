#include <random>
#include <cstdint>

// 生成随机规格化float（32位）
int32_t generate_normalized_fp32() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // IEEE 754 float格式：1位符号 + 8位指数 + 23位尾数
    // 规格化数的指数范围：1-254（避免0和255）
    std::uniform_int_distribution<uint32_t> sign_dist(0, 1);
    std::uniform_int_distribution<uint32_t> exp_dist(1, 254);  // 避免0（非规格化）和255（无穷/NaN）
    std::uniform_int_distribution<uint32_t> mantissa_dist(0, 0x7FFFFF);
    
    uint32_t sign = sign_dist(gen);
    uint32_t exp = exp_dist(gen);
    uint32_t mantissa = mantissa_dist(gen);
    
    uint32_t float_bits = (sign << 31) | (exp << 23) | mantissa;
    return *(int32_t *)(&float_bits);
}

// 生成随机规格化__fp16（16位）
int16_t generate_normalized_fp16() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // IEEE 754 __fp16格式：1位符号 + 5位指数 + 10位尾数
    // 规格化数的指数范围：1-30（避免0和31）
    std::uniform_int_distribution<uint16_t> sign_dist(0, 1);
    std::uniform_int_distribution<uint16_t> exp_dist(0, 31);  // 避免0（非规格化）和31（无穷/NaN）
    std::uniform_int_distribution<uint16_t> mantissa_dist(0, 0x3FF);
    
    uint16_t sign = sign_dist(gen);
    uint16_t exp = exp_dist(gen);
    uint16_t mantissa = mantissa_dist(gen);
    
    uint16_t fp16_bits = (sign << 15) | (exp << 10) | mantissa;
    return *(int16_t *)(&fp16_bits);
}
// 生成随机规格化__bf16（16位）
int16_t generate_normalized_bf16() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // IEEE 754 __bf16格式：1位符号 + 8位指数 + 7位尾数
    // 规格化数的指数范围：1-254（避免0和255）
    std::uniform_int_distribution<uint16_t> sign_dist(0, 1);
    std::uniform_int_distribution<uint16_t> exp_dist(0, 255);  // 避免0（非规格化）和255（无穷/NaN）
    std::uniform_int_distribution<uint16_t> mantissa_dist(0, 0x7F);
    
    uint16_t sign = sign_dist(gen);
    uint16_t exp = exp_dist(gen);
    uint16_t mantissa = mantissa_dist(gen);
    
    uint16_t bf16_bits = (sign << 15) | (exp << 7) | mantissa;
    return *(int16_t *)(&bf16_bits);
}

// 生成随机规格化__MXFP8 e4m3（8位）
int8_t generate_normalized_e4m3() {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_int_distribution<uint16_t> sign_dist(0, 1);
    std::uniform_int_distribution<uint16_t> bit7(0, 126); // e4m3只有低7位全1是NAN

    uint16_t e4m3_bits = (sign_dist(gen) << 7) | bit7(gen);
    return *(int8_t*)&e4m3_bits;
}

// 生成随机规格化__MXFP8 e5m2（8位）
int8_t generate_normalized_e5m2() {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_int_distribution<uint16_t> sign_dist(0, 1);
    std::uniform_int_distribution<uint16_t> exp_dist(0, 30);  // 避免31（无穷/NaN）
    std::uniform_int_distribution<uint16_t> mantissa_dist(0, 3);

    uint16_t sign = sign_dist(gen);
    uint16_t exp = exp_dist(gen);
    uint16_t mantissa = mantissa_dist(gen);
    uint16_t e5m2_bits = (sign << 7) | (exp << 2) | mantissa;
    return *(int8_t*)&e5m2_bits;
}

int8_t generate_normalized_e8m0() {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_int_distribution<uint16_t> bit8(0, 254);
    return bit8(gen);
}

int8_t generate_normalized_2e2m1() {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_int_distribution<uint16_t> bit8(0, 255);
    return bit8(gen);
}

int gen_exception_fp16(int16_t* a, int16_t* b, int32_t* c, int size, int is_rand){
    return 0;
};
int gen_exception_bf16(int16_t* a, int16_t* b, int32_t* c, int size, int is_rand){
    return 0;
};
int gen_exception_tf32(int32_t* a, int32_t* b, int32_t* c, int size, int is_rand){
    return 0;
};
int gen_exception_fp8 (int8_t* a, int8_t* b, int32_t* c, int size, int is_rand){
    return 0;
};