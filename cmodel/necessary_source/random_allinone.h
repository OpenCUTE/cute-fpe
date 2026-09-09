#include <random>
#include <cstdint>

#define RANDOM 0
#define FIXED_SEED 113
#define EXP_BIAS(exp_w) ((1 << (exp_w - 1)) - 1)
#define EXP_MASK(exp_w) ((1 << exp_w) - 1)
#define MANT_MASK(mant_w) ((1 << mant_w) - 1)


typedef struct{
    int32_t exp_width;
    int32_t mant_width;     // without hidden bit
    int32_t exp_bias;       // exp_bias = 1 << (exp_width - 1) - 1
    int32_t exp_max;        // non-biased max exp (inf/nan not include)
    const char* dtype;      // fp type (for debug)
}fp_descriptor_t;

typedef enum{       //enum类似localparam
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    E4M3 = 3,
    E2M1 = 4,
    E5M2 = 5,
    E3M2 = 6,       
    E2M3 = 7,
    INT  = 8,
    UE8M0= 9,
}fp_dtype_t;

const fp_descriptor_t FP_CONFIGS[] = {   
    {8,   23,   EXP_BIAS(8),   EXP_BIAS(8),    " FP32"    },
    {5,   10,   EXP_BIAS(5),   EXP_BIAS(5),    " FP16"    },
    {8,   7,    EXP_BIAS(8),   EXP_BIAS(8),    " BF16"    },
    {4,   3,    EXP_BIAS(4),   8,              " E4M3"    },
    {2,   1,    EXP_BIAS(2),   2,              " E2M1"    },
    {5,   2,    EXP_BIAS(5),   EXP_BIAS(5),    " E5M2"    },
    {3,   2,    EXP_BIAS(3),   4,              " E3M2"    },
    {2,   3,    EXP_BIAS(2),   2,              " E2M3"    },
    {0,   8,    0          ,   8          ,    "  INT"    }, // 整数有待考虑
    {8,   0,    EXP_BIAS(8),   EXP_BIAS(8),    "UE8M0"    },
};

// 随机生成浮点数的思路是，先按bit随机生成，再使用pack_fp打包为浮点数
// 这种随机方案对于表示范围较广（如fp32/tf32/bf16）的浮点数生成更有随机性
static random_device rd;       // random seed, static保证只会在第一次调用时初始化一次
static uint32_t seed = RANDOM ? rd() : FIXED_SEED;
static mt19937 gen(seed);      // 指定随机数生成算法为mersenne twister
static uniform_int_distribution<uint32_t> bit01(0, 1);  // 0/1随机分布生成器

inline void reset_random_generator(uint32_t new_seed) {
    seed = new_seed;
    gen.seed(seed);
}

inline uint32_t current_random_seed() {
    return seed;
}

// 根据符号位/指数位/尾数位把浮点数重新打包成 raw bit。
inline uint32_t pack_fp (uint32_t sign, uint32_t exp, uint32_t mant, uint32_t exp_width, uint32_t mant_width) {
    return (sign << (exp_width + mant_width)) | (exp << mant_width) | mant;
}

// 根据dtype生成该类型的随机浮点数（包括normaldenormal\nan\inf\zero）
// 生成满足给定 data_type 取值分布的随机浮点数。
inline  uint32_t gen_random_fp(int data_type ) {

    const fp_descriptor_t* fp_desc = &FP_CONFIGS[data_type];
    int32_t exp_width = fp_desc->exp_width;
    int32_t mant_width = fp_desc->mant_width;

    uint32_t exp_allones = EXP_MASK(exp_width);
    uint32_t mant_allones = MANT_MASK(mant_width);
    uint32_t sign = bit01(gen);

    uniform_int_distribution<uint32_t> gen_range(0, 999);   //根据range生成不同区间的浮点数
    uint32_t range = gen_range(gen);

    uint32_t rawhex;

    if (data_type == UE8M0) {
        uniform_int_distribution<uint8_t> gen_ue8m0(0, 255);
        rawhex = gen_ue8m0(gen);
    } else {
        if (range < 3) {       // 0.3% zero
            rawhex = pack_fp(sign, 0, 0, exp_width, mant_width);
        }
        else if (range < 6) {  // 0.3% inf
            if (data_type != E4M3 && data_type != E2M3 && data_type != E3M2 && data_type != E2M1 && data_type != UE8M0)
                rawhex = pack_fp(sign, exp_allones, 0, exp_width, mant_width);
            else
                rawhex = pack_fp(sign, 0, 0, exp_width, mant_width);
        }
        else if (range < 8) {  // 0.2% qnan
            if (data_type != E2M3 && data_type != E3M2 && data_type != E2M1)
                rawhex = pack_fp(sign, exp_allones, 1<<(mant_width-1), exp_width, mant_width);
            else if (data_type == E4M3)
                rawhex = pack_fp(sign, exp_allones, mant_allones, exp_width, mant_width);
            else
                rawhex = pack_fp(sign, 0, 0, exp_width, mant_width);
        }
        else if (range < 10) {  // 0.2% snan
            if (data_type != E2M3 && data_type != E3M2 && data_type != E2M1)
                rawhex = pack_fp(sign, exp_allones, 1u, exp_width, mant_width);
            else if (data_type == E4M3)
                rawhex = pack_fp(sign, exp_allones, mant_allones, exp_width, mant_width);
            else
                rawhex = pack_fp(sign, 0, 0, exp_width, mant_width);
        }
        else if (range < 900) { // 89% denormal
            uniform_int_distribution<uint32_t> gen_demant(1u, mant_allones);
            uint32_t demant = gen_demant(gen); 
            rawhex = pack_fp(sign, 0u, demant, exp_width, mant_width);
        }
        else {                  // 10%  normal
            uint32_t normant;
            uint32_t norexp;
            uniform_int_distribution<uint32_t> gen_normant(0u, mant_allones);
            normant = gen_normant(gen);
            if (data_type == E4M3 || data_type == E2M3 || data_type == E3M2 || data_type == E2M1) {
                uniform_int_distribution<uint32_t> gen_norexp(0u, exp_allones);
                norexp = gen_norexp(gen);
            } else {
                uniform_int_distribution<uint32_t> gen_norexp(0u, exp_allones-1u);
                norexp = gen_norexp(gen);
            }
            rawhex = pack_fp(sign, norexp, normant, exp_width, mant_width);
        }
    }

    return rawhex;
}
