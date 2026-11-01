#include <stdio.h>
#include "FloatDecode.h"
#include "mma_descriptor.h"

#include <cstdint>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <sys/stat.h>

#ifndef TRACE_FILE
#define TRACE_FILE "debug_data"
#endif

#ifndef FMAC_DUMP_OUTPUT
#define FMAC_DUMP_OUTPUT 1
#endif

#ifndef FMAC_DUMP_ALL
#define FMAC_DUMP_ALL 0
#endif
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

typedef struct {
    int enable;
    int vec_idx;
    int dtype;
    const char* src_dtype;
    const char* dst_dtype;
    FILE* fp_align;
    FILE* fp_sum;
} FmacTraceContext;

static inline FmacTraceContext& fmac_trace_context() {
    static FmacTraceContext ctx = {0, 0, -1, nullptr, nullptr, nullptr, nullptr};
    return ctx;
}

static inline void fmac_trace_close_all_files() {
    FmacTraceContext& ctx = fmac_trace_context();
    if (ctx.fp_align) {
        fclose(ctx.fp_align);
        ctx.fp_align = nullptr;
    }
    if (ctx.fp_sum) {
        fclose(ctx.fp_sum);
        ctx.fp_sum = nullptr;
    }
}

static inline int (*fmac_trace_counters())[2] {
    static int counters[256][2] = {{0}};
    return counters;
}

static inline int& fmac_trace_counter(int dtype, int dst_is_fp16) {
    int safe_dtype = (dtype >= 0 && dtype < 256) ? dtype : 255;
    return fmac_trace_counters()[safe_dtype][dst_is_fp16 ? 1 : 0];
}

static inline void fmac_trace_reset_counters() {
    memset(fmac_trace_counters(), 0, sizeof(int) * 256 * 2);
}

static inline const char* fmac_trace_dir() {
    const char* dir = getenv("FMAC_TRACE_DIR");
    return (dir && dir[0]) ? dir : TRACE_FILE;
}

static inline bool fmac_trace_enabled() {
    return FMAC_DUMP_OUTPUT != 0 && FMAC_DUMP_ALL != 0 && getenv("FMAC_TRACE_INTERMEDIATE") != NULL;
}

static inline bool fmac_trace_use_logical_payload() {
    return mma_topology_128_input_enabled();
}

// mxf4nvf4指令下group=4，其他情况group=2
static inline int fmac_trace_group_count(int dtype) {
    return mma_dtype_uses_mxf4_topology(dtype) ? 4 : 2;
}

static inline int fmac_trace_logical_payload_count(int payload_physical_count, int dtype);
static inline int fmac_trace_payload_index_logical_to_physical(int logical_idx, int payload_physical_count, int dtype);

static inline int32_t mxf4_block_dot_sum(const int8_t* a, const int8_t* b, int block_idx, bool* all_products_zero = NULL) {
    int32_t block_dot = 0;
    bool all_zero = true;
    int base = block_idx * 8;
    for (int i = 0; i < 8; ++i) {
        int idx = base + i;
        int32_t low_product = e2m1tofixed(a[idx]) * e2m1tofixed(b[idx]);
        int32_t high_product = e2m1tofixed(a[idx] >> 4) * e2m1tofixed(b[idx] >> 4);
        block_dot += low_product;
        block_dot += high_product;
        all_zero = all_zero && (low_product == 0) && (high_product == 0);
    }
    if (all_products_zero) {
        *all_products_zero = all_zero;
    }
    return block_dot;
}

// One external scale covers two 16-element product blocks, or 32 FP4/INT4 values.
static inline void expand_mx_scale_per_32_elements(
    const int8_t* external_scale, int block_num, int8_t* block_scale)
{
    for (int block = 0; block < block_num; ++block) {
        block_scale[block] = external_scale[block / 2];
    }
}

static inline int32_t mxint4_block_dot_sum(const int8_t* a, const int8_t* b, int block_idx, bool* all_products_zero = NULL) {
    int32_t block_dot = 0;
    bool all_zero = true;
    int base = block_idx * 8;
    for (int i = 0; i < 8; ++i) {
        int idx = base + i;
        uint8_t a_packed = static_cast<uint8_t>(a[idx]);
        uint8_t b_packed = static_cast<uint8_t>(b[idx]);
        int32_t a_low = a_packed & 0x0f;
        int32_t b_low = b_packed & 0x0f;
        int32_t a_high = (a_packed >> 4) & 0x0f;
        int32_t b_high = (b_packed >> 4) & 0x0f;

        if (a_low & 0x8) a_low -= 16;
        if (b_low & 0x8) b_low -= 16;
        if (a_high & 0x8) a_high -= 16;
        if (b_high & 0x8) b_high -= 16;

        int32_t low_product = a_low * b_low;
        int32_t high_product = a_high * b_high;
        block_dot += low_product + high_product;
        all_zero = all_zero && (low_product == 0) && (high_product == 0);
    }
    if (all_products_zero) {
        *all_products_zero = all_zero;
    }
    return block_dot;
}

// logical为128bit输入而physical为256bit输入
static inline int fmac_trace_logical_payload_count(int payload_physical_count, int dtype) {
    if (!fmac_trace_use_logical_payload() || payload_physical_count <= 1) { //不会有这种情况
        return payload_physical_count;
    }
    if (payload_physical_count <= fmac_trace_group_count(dtype)) { //也不会有这种情况
        return payload_physical_count;
    }
    return payload_physical_count / 2; // 默认logical_payload_count数量是physical_payload_count数量的一半
}

// 统一计算 payload 重排所需参数。
// 返回 true 代表需要启用“logical(128b) -> physical(256b)”分组重排；false 代表索引保持不变。
static inline bool fmac_trace_payload_remap_params(
    int payload_physical_count, int dtype,
    int* logical_count,
    int* logical_group_size,
    int* physical_group_size)
{
    int lc = fmac_trace_logical_payload_count(payload_physical_count, dtype);
    int group_count = fmac_trace_group_count(dtype);
    if (!fmac_trace_use_logical_payload() ||
        payload_physical_count <= 1 ||
        payload_physical_count <= group_count ||
        lc < group_count) {
        if (logical_count) *logical_count = lc;
        if (logical_group_size) *logical_group_size = 0;
        if (physical_group_size) *physical_group_size = 0;
        return false;
    }

    int lgs = lc / group_count;
    int pgs = payload_physical_count / group_count;
    if (lgs == 0 || pgs == 0) {
        if (logical_count) *logical_count = lc;
        if (logical_group_size) *logical_group_size = 0;
        if (physical_group_size) *physical_group_size = 0;
        return false;
    }

    if (logical_count) *logical_count = lc;
    if (logical_group_size) *logical_group_size = lgs;
    if (physical_group_size) *physical_group_size = pgs;
    return true;
}

// 将128bit的logical_payload输入映射到256bit的physical_payload输入
// 计算当前写入的地址索引值：每个 physical group 里只取“前半段”作为 logical
static inline int fmac_trace_payload_index_logical_to_physical(int logical_idx, int payload_physical_count, int dtype) {
    int logical_count = 0;
    int logical_group_size = 0;
    int physical_group_size = 0;
    if (!fmac_trace_payload_remap_params(payload_physical_count, dtype,
                                         &logical_count, &logical_group_size, &physical_group_size)) {
        return logical_idx;
    }
    int group_idx = logical_idx / logical_group_size;
    int offset_in_group = logical_idx % logical_group_size;
    return group_idx * physical_group_size + offset_in_group;
}

// 将256bit的physical_payload输入映射回128bit的logical_payload输入
static inline int fmac_trace_payload_index_physical_to_logical(int physical_idx, int payload_physical_count, int dtype) {
    int logical_count = 0;
    int logical_group_size = 0;
    int physical_group_size = 0;
    if (!fmac_trace_payload_remap_params(payload_physical_count, dtype,
                                         &logical_count, &logical_group_size, &physical_group_size)) {
        return physical_idx;
    }

    // O(1) 逆映射：physical group 的后半段没有对应的 logical 槽位。
    int group_idx = physical_idx / physical_group_size;
    int offset_in_group = physical_idx % physical_group_size;
    if (offset_in_group >= logical_group_size) {
        return -1;
    }

    int logical_idx = group_idx * logical_group_size + offset_in_group;
    if (logical_idx >= logical_count) {
        return -1;
    }
    return logical_idx;
}

static inline const char* fmac_trace_src_name(int dtype) {
    switch (dtype) {
        case 1: return "fp16";
        case 2: return "bf16";
        case 3: return "tf32";
        case 9: return "mxf4nvf4_nvfp4";
        case 10: return "mxf4nvf4_mxfp4";
        default: {
            return mma_dtype_file_stem(dtype);
        }
    }
}

static inline void fmac_trace_begin(int dtype, bool dst_is_fp16) {
    FmacTraceContext& ctx = fmac_trace_context();
    if (!fmac_trace_enabled()) {
        ctx.enable = 0;
        return;
    }
    const char* new_src_dtype = fmac_trace_src_name(dtype);
    const char* new_dst_dtype = dst_is_fp16 ? "fp16" : "fp32";
    if (ctx.fp_align || ctx.fp_sum) {
        bool same_dtype = (ctx.dtype == dtype);
        bool same_src = ctx.src_dtype && strcmp(ctx.src_dtype, new_src_dtype) == 0;
        bool same_dst = ctx.dst_dtype && strcmp(ctx.dst_dtype, new_dst_dtype) == 0;
        if (!same_dtype || !same_src || !same_dst) {
            fmac_trace_close_all_files();
        }
    }
    int& counter = fmac_trace_counter(dtype, dst_is_fp16 ? 1 : 0);
    counter++;
    ctx.enable = 1;
    ctx.vec_idx = counter;
    ctx.dtype = dtype;
    ctx.src_dtype = new_src_dtype;
    ctx.dst_dtype = new_dst_dtype;
}

static inline void fmac_trace_end() {
    fmac_trace_context().enable = 0;
}

static inline FILE* fmac_trace_open(const char* operation) {
    FmacTraceContext& ctx = fmac_trace_context();
    if (!ctx.enable || !ctx.src_dtype || !ctx.dst_dtype) {
        return NULL;
    }
    // 确保调试输出目录存在
    const char* trace_dir = fmac_trace_dir();
    if (mkdir(trace_dir, 0775) != 0 && errno != EEXIST) {
        perror(trace_dir);
        return NULL;
    }
    FILE** slot = NULL;
    if (strcmp(operation, "mant_align") == 0) {
        slot = &ctx.fp_align;
    } else if (strcmp(operation, "mant_sum") == 0) {
        slot = &ctx.fp_sum;
    } else {
        return NULL;
    }

    if (!*slot) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/%s_%s_to_%s.txt.log",
                 trace_dir, operation, ctx.src_dtype, ctx.dst_dtype);
        const char* mode = (ctx.vec_idx == 1) ? "w" : "a";
        *slot = fopen(filename, mode);
        if (!*slot) {
            perror(filename);
            return NULL;
        }
    }

    fprintf(*slot, "-----------%d-----------\n", ctx.vec_idx);
    return *slot;
}

typedef struct {
    FloatDecode* raw_a;
    size_t raw_a_capacity;
    FloatDecode* raw_b;
    size_t raw_b_capacity;
    FloatDecode* decode_mul_result;
    size_t decode_mul_capacity;
    int32_t* block_dot;
    size_t block_dot_capacity;
    FloatDecode* scale_a;
    size_t scale_a_capacity;
    FloatDecode* scale_b;
    size_t scale_b_capacity;
} FmacNumericWorkspace;

static inline FmacNumericWorkspace& fmac_numeric_workspace() {
    static FmacNumericWorkspace ws = {NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0};
    return ws;
}

template <typename T>
static inline T* fmac_workspace_reserve(T** buffer, size_t* capacity, size_t count) {
    if (*capacity < count) {
        T* new_buffer = (T*)realloc(*buffer, count * sizeof(T));
        if (!new_buffer) {
            return NULL;
        }
        *buffer = new_buffer;
        *capacity = count;
    }
    return *buffer;
}

static inline bool fmac_prepare_decode_workspace(
    int size,
    FloatDecode** raw_a,
    FloatDecode** raw_b,
    FloatDecode** decode_mul_result)
{
    FmacNumericWorkspace& ws = fmac_numeric_workspace();
    *raw_a = fmac_workspace_reserve(&ws.raw_a, &ws.raw_a_capacity, (size_t)size);
    *raw_b = fmac_workspace_reserve(&ws.raw_b, &ws.raw_b_capacity, (size_t)size);
    *decode_mul_result = fmac_workspace_reserve(&ws.decode_mul_result, &ws.decode_mul_capacity, (size_t)size + 1);
    return *raw_a && *raw_b && *decode_mul_result;
}

static inline bool fmac_prepare_scale_workspace(
    int block_num,
    int32_t** block_dot,
    FloatDecode** scale_a,
    FloatDecode** scale_b,
    FloatDecode** decode_mul_result)
{
    FmacNumericWorkspace& ws = fmac_numeric_workspace();
    *block_dot = fmac_workspace_reserve(&ws.block_dot, &ws.block_dot_capacity, (size_t)block_num);
    *scale_a = fmac_workspace_reserve(&ws.scale_a, &ws.scale_a_capacity, (size_t)block_num);
    *scale_b = fmac_workspace_reserve(&ws.scale_b, &ws.scale_b_capacity, (size_t)block_num);
    *decode_mul_result = fmac_workspace_reserve(&ws.decode_mul_result, &ws.decode_mul_capacity, (size_t)block_num + 1);
    return *block_dot && *scale_a && *scale_b && *decode_mul_result;
}


//size是数组元素数，mode是数据类型
#define DEFINE_INT_MAC_WITH_TRACE(fn_name, prod_expr) \
int32_t fn_name(int8_t* a, int8_t* b, int32_t *c, int size){ \
    int32_t result = 0; \
    FILE* fp_sum = fmac_trace_open("mant_sum"); \
    int trace_count = size; \
    int64_t* trace_products = fp_sum ? (int64_t*)calloc(trace_count, sizeof(int64_t)) : NULL; \
    for (int i = 0; i < size; i++) { \
        int64_t prod = (prod_expr); \
        result += (int32_t)prod; \
        if (trace_products) { \
            trace_products[i] = prod; \
        } \
    } \
    result += *c; \
    if (fp_sum) { \
        int64_t partial_sum[4] = {0, 0, 0, 0}; \
        int group_base = trace_count / 4; \
        int group_rem = trace_count % 4; \
        int cursor = 0; \
        for (int g = 0; g < 4; ++g) { \
            int group_len = group_base + (g < group_rem ? 1 : 0); \
            for (int i = 0; i < group_len; ++i) { \
                partial_sum[g] += trace_products[cursor++]; \
            } \
            fprintf(fp_sum, "partial  sum:%016llx\n", \
                    (unsigned long long)(uint64_t)partial_sum[g]); \
        } \
        fprintf(fp_sum, "complete sum:%016llx\n", \
                (unsigned long long)(uint64_t)(int64_t)result); \
        fflush(fp_sum); \
    } \
    free(trace_products); \
    return result; \
}

DEFINE_INT_MAC_WITH_TRACE(i8mac, (int32_t)a[i] * (int32_t)b[i])
DEFINE_INT_MAC_WITH_TRACE(i8ui8mac, (int32_t)a[i] * (int32_t)*(uint8_t *)&b[i])
DEFINE_INT_MAC_WITH_TRACE(ui8i8mac, (int32_t)*(uint8_t *)&a[i] * (int32_t)b[i])
DEFINE_INT_MAC_WITH_TRACE(ui8ui8mac, (int32_t)*(uint8_t *)&a[i] * (int32_t)*(uint8_t *)&b[i])

#undef DEFINE_INT_MAC_WITH_TRACE


//做一个不带scale(4X)的通用累加处理，要求保证输入尾数的数据格式为x.y，其中y为25个小数位。
//需要注意的是，我们不会输出负0
//float_type == 0的时候目标格式为fp32；float_type == 1的时候目标格式为fp16
// decodeadd累加的项数为mantA x mantB加上mantC，故size的大小包含了所有参与累加的项数（即乘积项数+1），其中最后一项为C的输入
int32_t decodeadd(FloatDecode *x,
                  int size,
                  int decimal_num,
                  bool float_type = 0,
                  bool mxnvfp4_ea = 0,
                  const bool* block_all_products_zero = NULL){
    bool fda_exp_max = (decimal_num == 25);
    FloatDecode result = {0, fda_exp_max ? 0 : -126, 0};
    if (size <= 0) {
        return 0;
    }
    // 初始化result的异常信息
    DecodeExceptionInit(&result.exception);
    // printf("HELLO__________________\n");

    int32_t has_nan = 0;
    int32_t has_pinf = 0;
    int32_t has_ninf = 0;
    int32_t only_nzero = 1; // 用于检查是否只有负零
    bool has_exp_candidate = false;
    
    //找出其中的最大指数，并判断其中是否存在特殊值
    for (int i = 0; i < size; i++) {
        bool use_exp_candidate = !fda_exp_max || x[i].mantissa != 0;
        //如果不是空指针，那么就是mxnvfp4
        if(block_all_products_zero){
            use_exp_candidate = !block_all_products_zero[i];
        }
        int32_t candidate_exponent = x[i].exponent;
        if (use_exp_candidate && (!has_exp_candidate || candidate_exponent > result.exponent)) {
            result.exponent = candidate_exponent;
            has_exp_candidate = true;
        }
        has_nan |= x[i].exception.is_nan; // 检查是否有NaN
        has_pinf |= x[i].exception.is_pinf; // 检查是否有正无穷
        has_ninf |= x[i].exception.is_ninf; // 检查是否有负无穷
        only_nzero &= x[i].exception.is_nzero; // 检查是否只有负零
    }
    if (!has_exp_candidate) {
        result.exponent = -127;
    }
    else if(!mxnvfp4_ea && result.exponent < -133){
        result.exponent = -133;
    }
    else if(result.exponent < -139){
        result.exponent = -139;
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
                            
    // 提前计算trace槽位信息，正常路径和异常返回路径共用。
    FmacTraceContext& trace_ctx = fmac_trace_context();
    int payload_physical_count = size - 1; // 不包含c的输入
    int payload_trace_count = fmac_trace_logical_payload_count(payload_physical_count, trace_ctx.dtype); // 转换为logical payload数量
    int trace_total_count = payload_trace_count + 1; // 恢复c的输入数量
    bool has_exception = has_nan || has_pinf || has_ninf || only_nzero;

    // 提前打开日志文件，保证编号连续；即使异常返回也先按同一套对阶逻辑生成trace。
    FILE* fp_align = fmac_trace_open("mant_align");
    FILE* fp_sum = fmac_trace_open("mant_sum");

    #ifdef DECODEADD_DEBUG
        printf("Max exponent: %d\n", result.exponent);
    #endif

    // int64_t mantissa_sum = 0;

    int32_t* trace_rsft = NULL;
    int32_t* trace_exp = NULL;
    uint64_t* trace_mul = NULL;
    uint64_t* trace_shifted = NULL;
    uint64_t* trace_discarded = NULL;
    int64_t* trace_signed = NULL;
    uint32_t* trace_guard = NULL;
    uint32_t* trace_sticky = NULL;
    uint32_t* trace_lsb = NULL;
    uint32_t* trace_rne_inc = NULL;
    if (fp_align || fp_sum) {
        trace_rsft = (int32_t*)calloc(trace_total_count, sizeof(int32_t));
        trace_exp = (int32_t*)calloc(trace_total_count, sizeof(int32_t));
        trace_mul = (uint64_t*)calloc(trace_total_count, sizeof(uint64_t));
        trace_shifted = (uint64_t*)calloc(trace_total_count, sizeof(uint64_t));
        trace_discarded = (uint64_t*)calloc(trace_total_count, sizeof(uint64_t));
        trace_signed = (int64_t*)calloc(trace_total_count, sizeof(int64_t));
        trace_guard = (uint32_t*)calloc(trace_total_count, sizeof(uint32_t));
        trace_sticky = (uint32_t*)calloc(trace_total_count, sizeof(uint32_t));
        trace_lsb = (uint32_t*)calloc(trace_total_count, sizeof(uint32_t));
        trace_rne_inc = (uint32_t*)calloc(trace_total_count, sizeof(uint32_t));
    }

    #ifdef DECODEADD_DEBUG
        printf("Before adding: mantissa=%x\n", result.mantissa);
    #endif

    //进行尾数对阶处理
    for (int i = 0; i < size; i++) {
        int right_shift = (result.exponent - x[i].exponent > 63) ? 63 : (result.exponent - x[i].exponent);
        uint64_t mantissa_raw = (uint64_t)x[i].mantissa;
        int align_shift = right_shift;
        int64_t mantissa_shifted = (align_shift >= 0)
            ? (x[i].mantissa >> align_shift)
            : (x[i].mantissa << (-align_shift));
        uint64_t discarded_bits = 0;
        uint32_t align_guard = 0;
        uint32_t align_sticky = 0;
        if (align_shift > 0) {
            uint64_t discarded_mask = (align_shift >= 63) ? UINT64_MAX : ((1ULL << align_shift) - 1ULL);
            discarded_bits = mantissa_raw & discarded_mask;
            align_guard = (uint32_t)((discarded_bits >> (align_shift - 1)) & 0x1ULL);
            if (align_shift > 1) {
                uint64_t sticky_mask = (1ULL << (align_shift - 1)) - 1ULL;
                align_sticky = (discarded_bits & sticky_mask) != 0;
            }
        }
        uint32_t align_lsb = (uint32_t)(mantissa_shifted & 0x1LL);
        uint32_t align_rne_inc = align_guard && (align_sticky || align_lsb);
        #ifdef DECODEADD_DEBUG
            printf("x[%d]: before shift: mantissa=%x\t after shift: mantissa_shifted=%x, right_shift=%d\n", i, x[i].mantissa, mantissa_shifted, right_shift);
        #endif
        if(x[i].sign == 0){
            result.mantissa += mantissa_shifted;
        }
        else{
            result.mantissa -= mantissa_shifted;
        }
        if (fp_align || fp_sum) {
            int trace_slot = -1;
            if (i == payload_physical_count) {
                trace_slot = payload_trace_count; 
            } else {
                trace_slot = fmac_trace_payload_index_physical_to_logical(i, payload_physical_count, trace_ctx.dtype);
            }
            if (trace_slot >= 0) {
                trace_rsft[trace_slot] = right_shift;
                trace_exp[trace_slot] = x[i].exponent;
                trace_mul[trace_slot] = mantissa_raw;
                trace_shifted[trace_slot] = (uint64_t)mantissa_shifted;
                trace_discarded[trace_slot] = discarded_bits;
                trace_signed[trace_slot] = (x[i].sign == 0) ? mantissa_shifted : -mantissa_shifted;
                trace_guard[trace_slot] = align_guard;
                trace_sticky[trace_slot] = align_sticky;
                trace_lsb[trace_slot] = align_lsb;
                trace_rne_inc[trace_slot] = align_rne_inc;
            }
        }
        // #ifdef DECODEADD_DEBUG
        // printf("After adding: mantissa=%x\n", result.mantissa);
        // #endif
    }

    if (fp_align) {
        // 常规分支也保持同样约定：最后一行为加数c，使用mant_c字段名。
        for (int i = 0; i < trace_total_count; ++i) {
            if (i == trace_total_count - 1) { // 最后一项是c的指数
                fprintf(fp_align, "exp_max:%04d exp_c: %04d rsft_amt:%04d mant_c  (point alignment):%016llx mant_shifted:%016llx discarded:%016llx guard:%u sticky:%u lsb:%u rne_inc:%u signed_mant:%016llx\n",
                        result.exponent,
                        trace_exp[i],
                        trace_rsft[i],
                        (unsigned long long)trace_mul[i],
                        (unsigned long long)trace_shifted[i],
                        (unsigned long long)trace_discarded[i],
                        trace_guard[i],
                        trace_sticky[i],
                        trace_lsb[i],
                        trace_rne_inc[i],
                        (unsigned long long)trace_signed[i]);
            } else { // 前面的项是axb的指数
                fprintf(fp_align, "exp_max:%04d exp_ab:%04d rsft_amt:%04d mant_mul(point alignment):%016llx mant_shifted:%016llx discarded:%016llx guard:%u sticky:%u lsb:%u rne_inc:%u signed_mant:%016llx\n",
                        result.exponent,
                        trace_exp[i],
                        trace_rsft[i],
                        (unsigned long long)trace_mul[i],
                        (unsigned long long)trace_shifted[i],
                        (unsigned long long)trace_discarded[i],
                        trace_guard[i],
                        trace_sticky[i],
                        trace_lsb[i],
                        trace_rne_inc[i],
                        (unsigned long long)trace_signed[i]);
            }
        }
        fflush(fp_align);
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

    if (fp_sum) {
        int64_t partial_sum[4] = {0, 0, 0, 0};
        int group_base = payload_trace_count / 4;
        int group_rem = payload_trace_count % 4;
        int cursor = 0;
        for (int g = 0; g < 4; ++g) {
            int group_len = group_base + (g < group_rem ? 1 : 0);
            for (int i = 0; i < group_len; ++i) {
                partial_sum[g] += trace_signed[cursor++];
            }
        }
        for (int i = 0; i < 4; ++i) {
            fprintf(fp_sum, "partial  sum:%016llx\n",
                    (unsigned long long)(uint64_t)partial_sum[i]);
        }
        fprintf(fp_sum, "complete sum:%016llx\n",
                (unsigned long long)(uint64_t)(result.sign ? -result.mantissa : result.mantissa)); //打印的时候仍然打印带符号的结果，方便对比
        fflush(fp_sum);
    }
    free(trace_rsft);
    free(trace_exp);
    free(trace_mul);
    free(trace_shifted);
    free(trace_discarded);
    free(trace_signed);
    free(trace_guard);
    free(trace_sticky);
    free(trace_lsb);
    free(trace_rne_inc);

    if (has_exception)
        return exception_res_bits;

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
        int output_drop_bits = decimal_num - 23;
        if (output_drop_bits > 0) {
            result.mantissa >>= output_drop_bits;
        } else if (output_drop_bits < 0) {
            result.mantissa <<= (-output_drop_bits);
        }

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
        if(result.exponent < -25){
            result.exponent = -15;
            mant = 0; // 设置为零
            result.sign = 0; //只有正0
        } else if(result.exponent < -14) {//非规格数
            lost_bits_2 = mant & ((1u << (-result.exponent - 14)) - 1);
            mant >>= (-result.exponent - 14); // 将尾数右移
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
        if (mant_fp16 == 2048 && !result.exception.is_subnormal) { //is_subnormal与mant_fp16 == 2048其实并不会同时成立，在is_subnormal时mant_16一定会右移，高位为0，累加不会产生进位
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

        if (result.exponent == -15 && mant_fp16 == 0) {
            result.sign = 0; // FP16 zero canonicalizes to +0 on NVIDIA MMA
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
    FmacTraceContext& trace_ctx = fmac_trace_context();
    // FILE* fp_mul = (trace_ctx.dtype == 9 || trace_ctx.dtype == 10) ? NULL : fmac_trace_open("mant_mul");
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

    // mxf4nvf4下的e2m1会先转换为定点数再进行尾数相乘
    // if (fp_mul) {
    //     int trace_count = fmac_trace_logical_payload_count(size, trace_ctx.dtype);
    //     for (int i = 0; i < trace_count; ++i) {
    //         int physical_idx = fmac_trace_payload_index_logical_to_physical(i, size, trace_ctx.dtype);
    //         fprintf(fp_mul, "%016llx\n",
    //                 (unsigned long long)(uint64_t)mul_result[physical_idx].mantissa);
    //     }
    //     fclose(fp_mul);
    // }

    return 0;
}

//float_type == 0的时候目标格式为fp32；float_type == 1的时候目标格式为fp16
int32_t fp16mac(int16_t* a, int16_t* b, int32_t *c, int size, bool float_type = 0) {
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_fp16(a[i]);
        raw_b[i] = decode_fp16(b[i]);
        #ifdef FP16_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

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
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_bf16(a[i]);
        raw_b[i] = decode_bf16(b[i]);
        #ifdef FP16_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

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
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

    for (int j = 0; j < size / 32; j++) { // j<2
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
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_e4m3(a[i]);
        raw_b[i] = decode_e4m3(b[i]);
        #ifdef FP8_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

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
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

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
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_e5m2(a[i]);
        raw_b[i] = decode_e5m2(b[i]);
        #ifdef FP8_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

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
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_e3m2(a[i]);
        raw_b[i] = decode_e3m2(b[i]);
        #ifdef FP8_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

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
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_e2m3(a[i]);
        raw_b[i] = decode_e2m3(b[i]);
        #ifdef FP8_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

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
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

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
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

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
    int32_t *block_dot = NULL;
    bool* block_all_products_zero = NULL;
    FloatDecode *scale_a = NULL;
    FloatDecode *scale_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_scale_workspace(block_num, &block_dot, &scale_a, &scale_b, &decode_mul_result)) {
        return 0;
    }
    block_all_products_zero = (bool*)calloc((size_t)block_num + 1, sizeof(bool));
    if (!block_all_products_zero) {
        return 0;
    }
    for (int j = 0; j < block_num; j++) {
        block_dot[j] = mxf4_block_dot_sum(a, b, j, &block_all_products_zero[j]);
        scale_a[j] = decode_e4m3(a_scale[j]);
        scale_b[j] = decode_e4m3(b_scale[j]);
        if (scale_a[j].sign == 1) {
            scale_a[j].sign = 0;
        }
        if (scale_b[j].sign == 1) {
            scale_b[j].sign = 0;
        }
        #ifdef FP4_DECODE_DEBUG
        printf("a_scale:%x, b_scale:%x\n", a_scale[j], b_scale[j]);
        printf("block_dot[%d]:%x\n", j, block_dot[j]);
        printf("block_all_products_zero[%d]:%d\n", j, block_all_products_zero[j] ? 1 : 0);
        printf("scale_a[%d]: sign=%d, exponent=%d, mantissa=%x\n", j, scale_a[j].sign, scale_a[j].exponent, scale_a[j].mantissa);
        printf("scale_b[%d]: sign=%d, exponent=%d, mantissa=%x\n", j, scale_b[j].sign, scale_b[j].exponent, scale_b[j].mantissa);
        #endif
    }
    for(int i = 0; i < (block_num + 1); i++) {
        DecodeExceptionInit(&decode_mul_result[i].exception); // 初始化异常信息
    }
    decodemul(scale_a, scale_b, block_num, decode_mul_result);
    decode_mul_result[block_num] = decode_fp32(c[0]); // 添加一个零元素用于累加

    //检测c是否为0
    if(decode_mul_result[block_num].mantissa == 0)
        block_all_products_zero[block_num] = true;
    else
        block_all_products_zero[block_num] = false;

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
            printf("final block_all_products_zero[%d]:%d\n", i, block_all_products_zero[i] ? 1 : 0);
        
    }
    #endif

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, block_num + 1, 35, false, true, block_all_products_zero); // 将初始值c和计算结果相加
    free(block_all_products_zero);
    
    return bits;
}

int32_t mxfp4mac(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {
    int block_num = size / 8;     // 32个fp4一个block,硬件实现按16个为一组, block_num=4

    int32_t *block_dot = NULL;
    bool* block_all_products_zero = NULL;
    FloatDecode *scale_a = NULL;
    FloatDecode *scale_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_scale_workspace(block_num, &block_dot, &scale_a, &scale_b, &decode_mul_result)) {
        return 0;
    }
    int8_t* a_block_scale = (int8_t*)calloc((size_t)block_num, sizeof(int8_t));
    int8_t* b_block_scale = (int8_t*)calloc((size_t)block_num, sizeof(int8_t));
    if (!a_block_scale || !b_block_scale) {
        free(a_block_scale);
        free(b_block_scale);
        return 0;
    }
    expand_mx_scale_per_32_elements(a_scale, block_num, a_block_scale);
    expand_mx_scale_per_32_elements(b_scale, block_num, b_block_scale);
    block_all_products_zero = (bool*)calloc((size_t)block_num + 1, sizeof(bool));
    if (!block_all_products_zero) {
        free(a_block_scale);
        free(b_block_scale);
        return 0;
    }
    for (int j = 0; j < block_num; j++) {
        block_dot[j] = mxf4_block_dot_sum(a, b, j, &block_all_products_zero[j]);
        DecodeExceptionInit(&scale_a[j].exception);
        DecodeExceptionInit(&scale_b[j].exception);
        scale_a[j].sign = 0;
        scale_b[j].sign = 0;
        scale_a[j].mantissa = 1;
        scale_b[j].mantissa = 1;
        scale_a[j].exception.is_nan = a_block_scale[j] == (int8_t)0xFF ? 1 : 0;
        scale_b[j].exception.is_nan = b_block_scale[j] == (int8_t)0xFF ? 1 : 0;
        scale_a[j].exponent = (((int32_t)a_block_scale[j]) & 0x000000ff) - 127;
        scale_b[j].exponent = (((int32_t)b_block_scale[j]) & 0x000000ff) - 127;
        #ifdef FP4_DECODE_DEBUG
        printf("scale_a[%d]: exponent=%d, scale_b[%d]: exponent=%d\n", j, scale_a[j].exponent, j, scale_b[j].exponent);
        printf("a_scale:%x, b_scale:%x\n", a_block_scale[j], b_block_scale[j]);
        printf("block_dot[%d]:%x\n", j, block_dot[j]);
        printf("block_all_products_zero[%d]:%d\n", j, block_all_products_zero[j] ? 1 : 0);
        #endif
    }
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

    //检测c是否为0
    if(decode_mul_result[block_num].mantissa == 0)
        block_all_products_zero[block_num] = true;
    else
        block_all_products_zero[block_num] = false;

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
            printf("final block_all_products_zero[%d]:%d\n", i, block_all_products_zero[i] ? 1 : 0);
        
    }
    #endif

    // 将结果转换为float
    int32_t bits = decodeadd(decode_mul_result, block_num + 1, 35, false, true, block_all_products_zero); // 将初始值c和计算结果相加
    free(block_all_products_zero);
    free(a_block_scale);
    free(b_block_scale);
    
    return bits;
}

int32_t mxint4(int8_t* a, int8_t* a_scale, int8_t* b, int8_t* b_scale, int32_t *c, int size) {
    int block_num = size / 8;
    int32_t *block_dot = NULL;
    bool* block_all_products_zero = NULL;
    FloatDecode *scale_a = NULL;
    FloatDecode *scale_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_scale_workspace(block_num, &block_dot, &scale_a, &scale_b, &decode_mul_result)) {
        return 0;
    }
    int8_t* a_block_scale = (int8_t*)calloc((size_t)block_num, sizeof(int8_t));
    int8_t* b_block_scale = (int8_t*)calloc((size_t)block_num, sizeof(int8_t));
    block_all_products_zero = (bool*)calloc((size_t)block_num + 1, sizeof(bool));
    if (!a_block_scale || !b_block_scale || !block_all_products_zero) {
        free(a_block_scale);
        free(b_block_scale);
        free(block_all_products_zero);
        return 0;
    }
    expand_mx_scale_per_32_elements(a_scale, block_num, a_block_scale);
    expand_mx_scale_per_32_elements(b_scale, block_num, b_block_scale);

    for (int j = 0; j < block_num; ++j) {
        block_dot[j] = mxint4_block_dot_sum(a, b, j, &block_all_products_zero[j]);
        DecodeExceptionInit(&scale_a[j].exception);
        DecodeExceptionInit(&scale_b[j].exception);
        scale_a[j].sign = 0;
        scale_b[j].sign = 0;
        scale_a[j].mantissa = 1;
        scale_b[j].mantissa = 1;
        scale_a[j].exception.is_nan = a_block_scale[j] == (int8_t)0xFF ? 1 : 0;
        scale_b[j].exception.is_nan = b_block_scale[j] == (int8_t)0xFF ? 1 : 0;
        scale_a[j].exponent = (((int32_t)a_block_scale[j]) & 0xff) - 127;
        scale_b[j].exponent = (((int32_t)b_block_scale[j]) & 0xff) - 127;
    }
    for (int i = 0; i < block_num + 1; ++i) {
        DecodeExceptionInit(&decode_mul_result[i].exception);
    }
    decodemul(scale_a, scale_b, block_num, decode_mul_result);
    decode_mul_result[block_num] = decode_fp32(c[0]);
    block_all_products_zero[block_num] = decode_mul_result[block_num].mantissa == 0;

    for (int i = 0; i < block_num; ++i) {
        if (block_dot[i] < 0) {
            block_dot[i] = -block_dot[i];
            decode_mul_result[i].sign = 1;
        }
        decode_mul_result[i].mantissa *= block_dot[i];
        decode_mul_result[i].mantissa <<= 31;
    }
    decode_mul_result[block_num].mantissa <<= 12;
    int32_t bits = decodeadd(decode_mul_result, block_num + 1, 35, false, true, block_all_products_zero);
    free(block_all_products_zero);
    free(a_block_scale);
    free(b_block_scale);
    return bits;
}

int32_t tf32mac(int32_t* a, int32_t* b, int32_t *c, int size) {
    FloatDecode *raw_a = NULL;
    FloatDecode *raw_b = NULL;
    FloatDecode *decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }
    for (int i = 0; i < size; i++) {
        raw_a[i] = decode_tf32(a[i]);
        raw_b[i] = decode_tf32(b[i]);
        #ifdef FP16_DECODE_DEBUG
            printf("a[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_a[i].sign, raw_a[i].exponent, raw_a[i].mantissa);
            printf("b[%d]: sign=%d, exponent=%d, mantissa=%x\n", i, raw_b[i].sign, raw_b[i].exponent, raw_b[i].mantissa);
        #endif
    }

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

static inline int fmac_effective_mantissa_bits(MmaOperandKind kind) {
    switch (kind) {
        case MmaOperandKind::FP16: return 11;
        case MmaOperandKind::BF16: return 8;
        case MmaOperandKind::TF32: return 11;
        case MmaOperandKind::E4M3: return 4;
        case MmaOperandKind::E5M2: return 3;
        case MmaOperandKind::E2M3: return 4;
        case MmaOperandKind::E3M2: return 3;
        case MmaOperandKind::E2M1: return 2;
        default: return 0;
    }
}

static inline FloatDecode fmac_decode_operand_by_kind(const void* src, int idx, MmaOperandKind kind) {
    switch (kind) {
        case MmaOperandKind::FP16:
            return decode_fp16(((const int16_t*)src)[idx]);
        case MmaOperandKind::BF16:
            return decode_bf16(((const int16_t*)src)[idx]);
        case MmaOperandKind::TF32:
            return decode_tf32(((const int32_t*)src)[idx]);
        case MmaOperandKind::E4M3:
            return decode_e4m3(((const int8_t*)src)[idx]);
        case MmaOperandKind::E5M2:
            return decode_e5m2(((const int8_t*)src)[idx]);
        case MmaOperandKind::E2M3:
            return decode_e2m3(((const int8_t*)src)[idx]);
        case MmaOperandKind::E3M2:
            return decode_e3m2(((const int8_t*)src)[idx]);
        case MmaOperandKind::E2M1: {
            uint8_t raw = ((const uint8_t*)src)[idx] >> 2;
            return decode_nvfp4((int8_t)raw);
        }
        default: {
            FloatDecode decoded = {};
            DecodeExceptionInit(&decoded.exception);
            return decoded;
        }
    }
}

static inline int32_t fmac_generic_mixed_float_mac(
    const void* a,
    const void* b,
    int32_t* c,
    int size,
    const MmaDescriptor& desc,
    bool float_type)
{
    FloatDecode* raw_a = NULL;
    FloatDecode* raw_b = NULL;
    FloatDecode* decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

    for (int i = 0; i < size; ++i) {
        raw_a[i] = fmac_decode_operand_by_kind(a, i, desc.a_kind);
        raw_b[i] = fmac_decode_operand_by_kind(b, i, desc.b_kind);
    }

    for (int i = 0; i < size; ++i) {
        DecodeExceptionInit(&decode_mul_result[i].exception);
    }

    decodemul(raw_a, raw_b, size, decode_mul_result);
    if (!float_type) {
        decode_mul_result[size] = decode_fp32(c[0]);
        decode_mul_result[size].mantissa <<= 2;
    } else {
        decode_mul_result[size] = decode_fp16((int16_t)c[0]);
        decode_mul_result[size].mantissa <<= 15;
    }

    int product_shift = 27 - (fmac_effective_mantissa_bits(desc.a_kind) +
                              fmac_effective_mantissa_bits(desc.b_kind));
    if (product_shift < 0) {
        product_shift = 0;
    }
    for (int i = 0; i < size; ++i) {
        decode_mul_result[i].mantissa <<= product_shift;
    }

    int32_t bits = decodeadd(decode_mul_result, size + 1, 25, float_type);
    return bits;
}

static inline int32_t fmac_decode_integer_operand(const int8_t* src, int idx, MmaOperandKind kind) {
    switch (kind) {
        case MmaOperandKind::S8:
            return (int8_t)src[idx];
        case MmaOperandKind::U8:
            return (uint8_t)src[idx];
        case MmaOperandKind::S4: {
            uint8_t raw = (uint8_t)src[idx] & 0x0f;
            return (raw & 0x8) ? ((int32_t)raw - 16) : (int32_t)raw;
        }
        case MmaOperandKind::U4:
            return (uint8_t)src[idx] & 0x0f;
        default:
            return 0;
    }
}

static inline int32_t fmac_generic_integer_mac(
    const int8_t* a,
    const int8_t* b,
    int32_t* c,
    int size,
    const MmaDescriptor& desc)
{
    int64_t acc = (int64_t)c[0];
    FILE* fp_sum = fmac_trace_open("mant_sum");
    int trace_count = size;
    int64_t* trace_products = fp_sum ? (int64_t*)calloc(trace_count, sizeof(int64_t)) : NULL;
    for (int i = 0; i < size; ++i) {
        int64_t aval = fmac_decode_integer_operand(a, i, desc.a_kind);
        int64_t bval = fmac_decode_integer_operand(b, i, desc.b_kind);
        int64_t prod = aval * bval;
        acc += prod;
        if (trace_products) {
            trace_products[i] = prod;
        }
    }
    if (fp_sum) {
        int64_t partial_sum[4] = {0, 0, 0, 0};
        int group_base = trace_count / 4;
        int group_rem = trace_count % 4;
        int cursor = 0;
        for (int g = 0; g < 4; ++g) {
            int group_len = group_base + (g < group_rem ? 1 : 0);
            for (int i = 0; i < group_len; ++i) {
                partial_sum[g] += trace_products[cursor++];
            }
            fprintf(fp_sum, "partial  sum:%016llx\n",
                    (unsigned long long)(uint64_t)partial_sum[g]);
        }
        fprintf(fp_sum, "complete sum:%016llx\n",
                (unsigned long long)(uint64_t)acc);
        fflush(fp_sum);
    }
    free(trace_products);
    if (desc.satfinite) {
        if (acc > INT32_MAX) acc = INT32_MAX;
        if (acc < INT32_MIN) acc = INT32_MIN;
    }
    return (int32_t)acc;
}

static inline int32_t fmac_generic_mixed_scale_mac(
    const int8_t* a,
    const int8_t* a_scale,
    const int8_t* b,
    const int8_t* b_scale,
    int32_t* c,
    int size,
    const MmaDescriptor& desc)
{
    FloatDecode* raw_a = NULL;
    FloatDecode* raw_b = NULL;
    FloatDecode* decode_mul_result = NULL;
    if (!fmac_prepare_decode_workspace(size, &raw_a, &raw_b, &decode_mul_result)) {
        return 0;
    }

    int block_elems = 32;
    int block_count = (size + block_elems - 1) / block_elems;
    for (int j = 0; j < block_count; ++j) {
        uint8_t a_scale_u8 = (uint8_t)a_scale[j];
        uint8_t b_scale_u8 = (uint8_t)b_scale[j];
        bool a_scale_is_nan = (a_scale_u8 == 0xFF);
        bool b_scale_is_nan = (b_scale_u8 == 0xFF);
        int32_t a_scale_int = (int32_t)a_scale_u8;
        int32_t b_scale_int = (int32_t)b_scale_u8;

        int block_begin = j * block_elems;
        int block_end = block_begin + block_elems;
        if (block_end > size) {
            block_end = size;
        }
        for (int idx = block_begin; idx < block_end; ++idx) {
            raw_a[idx] = fmac_decode_operand_by_kind(a, idx, desc.a_kind);
            raw_b[idx] = fmac_decode_operand_by_kind(b, idx, desc.b_kind);

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
        }
    }

    for (int i = 0; i < size; ++i) {
        DecodeExceptionInit(&decode_mul_result[i].exception);
    }

    decodemul(raw_a, raw_b, size, decode_mul_result);
    decode_mul_result[size] = decode_fp32(c[0]);
    decode_mul_result[size].mantissa <<= 2;

    int product_shift = 27 - (fmac_effective_mantissa_bits(desc.a_kind) +
                              fmac_effective_mantissa_bits(desc.b_kind));
    if (product_shift < 0) {
        product_shift = 0;
    }
    for (int i = 0; i < size; ++i) {
        decode_mul_result[i].mantissa <<= product_shift;
    }

    int32_t bits = decodeadd(decode_mul_result, size + 1, 25);
    return bits;
}

int32_t mymac(void *a, void *b, void *c, int size, int dtype, bool float_type = 0) {
    fmac_trace_begin(dtype, float_type);
    int32_t bits = 0;
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (desc && desc->category == MmaCategory::MixedFloat) {
        bits = fmac_generic_mixed_float_mac(a, b, (int32_t*)c, size, *desc, float_type && desc->allow_fp16_output);
    } else if (desc && desc->category == MmaCategory::Integer) {
        bits = fmac_generic_integer_mac((const int8_t*)a, (const int8_t*)b, (int32_t*)c, size, *desc);
    } else if (dtype == 0) { // int8_t
        bits = i8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 1) { // int16_t
        bits = fp16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size, float_type);
    } else if (dtype == 2) { // int32_t
        bits = bf16mac((int16_t*)a, (int16_t*)b, (int32_t*)c, size);
    } else if (dtype == 3) { // bf16
        bits = tf32mac((int32_t*)a, (int32_t*)b, (int32_t*)c, size);
    } else if (dtype == 4) { // int8*uint8
        bits = i8ui8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 5) { // uint8*int8
        bits = ui8i8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    } else if (dtype == 6) { // uint8*uint8
        bits = ui8ui8mac((int8_t*)a, (int8_t*)b, (int32_t*)c, size);
    }
    fmac_trace_end();
    return bits; // 错误处理
}

// size一般为32: m16n8k64，k=64, bitsize=256bit, size=bitsize/8bits per byte=256/8=32
// 计算时每8个bytes构成一个block，也就是block_num=32/8=4
int32_t mymac_scale(void *a, void *a_scale, void *b, void *b_scale, void *c, int size, int dtype) {
    fmac_trace_begin(dtype, 0);
    int32_t bits = 0;
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (desc && desc->category == MmaCategory::MixedFloat && desc->uses_scale) {
        bits = fmac_generic_mixed_scale_mac((const int8_t*)a, (const int8_t*)a_scale,
                                            (const int8_t*)b, (const int8_t*)b_scale,
                                            (int32_t*)c, size, *desc);
    } else if (dtype == 9) { // nvfp4
        bits = nvfp4mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size);
    } else if (dtype == 10) { // mxfp4
        bits = mxfp4mac((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size);
    } else if (dtype == 19) { // mxint4
        bits = mxint4((int8_t*)a, (int8_t*)a_scale, (int8_t*)b, (int8_t*)b_scale, (int32_t*)c, size);
    }
    fmac_trace_end();
    return bits;
}

float my_float_abs(float x) {
    return (x < 0) ? -x : x;
}

float my_double_abs(double x) {
    return (x < 0) ? -x : x;
}
