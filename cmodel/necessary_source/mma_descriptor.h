#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Active legacy dtype ids.
constexpr int MMA_DTYPE_INT8_LEGACY   = 0;
constexpr int MMA_DTYPE_FP16          = 1;
constexpr int MMA_DTYPE_BF16          = 2;
constexpr int MMA_DTYPE_TF32          = 3;
constexpr int MMA_DTYPE_INT8_UINT8_LEGACY = 4;
constexpr int MMA_DTYPE_UINT8_INT8_LEGACY = 5;
constexpr int MMA_DTYPE_UINT8_UINT8_LEGACY = 6;
constexpr int MMA_DTYPE_NVFP4         = 9;
constexpr int MMA_DTYPE_MXFP4         = 10;
constexpr int MMA_DTYPE_MXINT4        = 19;

// RTL-compatible integer int4 opcodes. Each operand is represented by the
// low nibble of an int8_t in the C-model data path.
constexpr int MMA_DTYPE_INT4_INT4_RTL  = 15;
constexpr int MMA_DTYPE_INT4_UINT4_RTL = 16;
constexpr int MMA_DTYPE_UINT4_INT4_RTL = 17;
constexpr int MMA_DTYPE_UINT4_UINT4_RTL = 18;

static inline bool mma_is_rtl_int4_dtype(int dtype) {
    return dtype >= MMA_DTYPE_INT4_INT4_RTL &&
           dtype <= MMA_DTYPE_UINT4_UINT4_RTL;
}

// A value of zero keeps the normal --bitsize-derived vector length.  The
// all_fmac_test --int4-k option sets a positive value for one RTL int4 run.
static inline int& mma_rtl_int4_k_runtime_value() {
    static int value = 0;
    return value;
}

static inline void mma_set_rtl_int4_k(int k) {
    mma_rtl_int4_k_runtime_value() = k;
}

static inline int mma_rtl_int4_k() {
    return mma_rtl_int4_k_runtime_value();
}

// Removed CModel-only dtype aliases are not RTL opcode mappings. The RTL
// adapter selects the active mixed-float ids explicitly.
// 20 reserved: deprecated mixed bf16/tf32 entry

// Runtime-expanded heterogeneous f8/f6/f4 instructions.
constexpr int MMA_DTYPE_F8F6F4_MIXED_BASE = 32;   // 25 contiguous ids
constexpr int MMA_DTYPE_F8F6F4_MIXED_COUNT = 25;
constexpr int MMA_DTYPE_MXF8F6F4_MIXED_BASE = 57; // 25 contiguous ids
constexpr int MMA_DTYPE_MXF8F6F4_MIXED_COUNT = 25;

// Integer instructions.
constexpr int MMA_DTYPE_IMMA_U8S8_BASE = 96;      // 24 contiguous ids
constexpr int MMA_DTYPE_IMMA_U8S8_COUNT = 24;
constexpr int MMA_DTYPE_IMMA_U4S4_BASE = 128;     // 24 contiguous ids
constexpr int MMA_DTYPE_IMMA_U4S4_COUNT = 24;

enum class MmaCategory {
    Unknown,
    Legacy,
    MixedFloat,
    Integer,
};

enum class MmaShape {
    Invalid,
    M8N8K16,
    M8N8K32,
    M16N8K8,
    M16N8K16,
    M16N8K32,
    M16N8K64,
};

enum class MmaOperandKind {
    Invalid,
    FP16,
    BF16,
    TF32,
    E4M3,
    E5M2,
    E2M3,
    E3M2,
    E2M1,
    S8,
    U8,
    S4,
    U4,
};

struct MmaDescriptor {
    int id;
    MmaCategory category;
    MmaShape shape;
    MmaOperandKind a_kind;
    MmaOperandKind b_kind;
    bool satfinite;
    bool allow_fp16_output;
    bool uses_scale;
    bool is_legacy;
    char name[128];
    char file_stem[128];
};

static inline int& mma_mxf4_blocksize16_runtime_value() {
    static int value = 1;
    return value;
}

static inline void mma_set_mxf4_blocksize16(int blocksize16) {
    mma_mxf4_blocksize16_runtime_value() = blocksize16 ? 1 : 0;
}

static inline bool mma_mxf4_blocksize16_enabled() {
    return mma_mxf4_blocksize16_runtime_value() != 0;
}

static inline bool mma_dtype_is_mxf4(int dtype) {
    return dtype == MMA_DTYPE_NVFP4 || dtype == MMA_DTYPE_MXFP4;
}

static inline bool mma_dtype_uses_mxf4_topology(int dtype) {
    return mma_dtype_is_mxf4(dtype) || dtype == MMA_DTYPE_MXINT4;
}

// MXFP4 and MXINT4 use one external scale per 32 four-bit elements.
static inline bool mma_dtype_uses_32element_scale(int dtype) {
    return dtype == MMA_DTYPE_MXFP4 || dtype == MMA_DTYPE_MXINT4;
}

static inline bool mma_topology_128_input_enabled() {
    return getenv("TOPOLOGY_128_INPUT") != NULL || getenv("FP4_TOPOLOGY_128") != NULL;
}

static inline const char* mma_operand_kind_token(MmaOperandKind kind) {
    switch (kind) {
        case MmaOperandKind::FP16: return "fp16";
        case MmaOperandKind::BF16: return "bf16";
        case MmaOperandKind::TF32: return "tf32";
        case MmaOperandKind::E4M3: return "e4m3";
        case MmaOperandKind::E5M2: return "e5m2";
        case MmaOperandKind::E2M3: return "e2m3";
        case MmaOperandKind::E3M2: return "e3m2";
        case MmaOperandKind::E2M1: return "e2m1";
        case MmaOperandKind::S8: return "s8";
        case MmaOperandKind::U8: return "u8";
        case MmaOperandKind::S4: return "s4";
        case MmaOperandKind::U4: return "u4";
        default: return "unknown";
    }
}

static inline const char* mma_scaled_operand_token(MmaOperandKind kind) {
    switch (kind) {
        case MmaOperandKind::E4M3: return "mxe4m3";
        case MmaOperandKind::E5M2: return "mxe5m2";
        case MmaOperandKind::E2M3: return "mxe2m3";
        case MmaOperandKind::E3M2: return "mxe3m2";
        case MmaOperandKind::E2M1: return "mxe2m1";
        default: return "mxunknown";
    }
}

static inline const char* mma_operand_file_token(const MmaDescriptor& desc, bool operand_a) {
    MmaOperandKind kind = operand_a ? desc.a_kind : desc.b_kind;
    if (!desc.uses_scale) {
        return mma_operand_kind_token(kind);
    }
    if (desc.id == MMA_DTYPE_NVFP4) {
        return "nvfp4";
    }
    if (desc.id == MMA_DTYPE_MXFP4) {
        return "mxfp4";
    }
    if (desc.id == MMA_DTYPE_MXINT4) {
        return "mxint4";
    }
    return mma_scaled_operand_token(kind);
}

static inline const char* mma_shape_token(MmaShape shape) {
    switch (shape) {
        case MmaShape::M8N8K16: return "m8n8k16";
        case MmaShape::M8N8K32: return "m8n8k32";
        case MmaShape::M16N8K8: return "m16n8k8";
        case MmaShape::M16N8K16: return "m16n8k16";
        case MmaShape::M16N8K32: return "m16n8k32";
        case MmaShape::M16N8K64: return "m16n8k64";
        default: return "invalid";
    }
}

static inline int mma_shape_k(MmaShape shape) {
    switch (shape) {
        case MmaShape::M8N8K16: return 16;
        case MmaShape::M8N8K32: return 32;
        case MmaShape::M16N8K8: return 8;
        case MmaShape::M16N8K16: return 16;
        case MmaShape::M16N8K32: return 32;
        case MmaShape::M16N8K64: return 64;
        default: return 0;
    }
}

static inline const char* mma_accumulator_kind_token(const MmaDescriptor& desc, bool float_type) {
    if (desc.category == MmaCategory::Integer) {
        return "s32";
    }
    return (float_type && desc.allow_fp16_output) ? "fp16" : "fp32";
}

static inline void mma_output_file_prefix(
    const MmaDescriptor& desc,
    bool float_type,
    char* out,
    size_t out_size)
{
    if (!out || out_size == 0) {
        return;
    }
    snprintf(out, out_size, "%s_%s_%s_to_%s",
             mma_shape_token(desc.shape),
             mma_operand_file_token(desc, true),
             mma_operand_file_token(desc, false),
             mma_accumulator_kind_token(desc, float_type));
}

static inline int mma_operand_storage_bits(MmaOperandKind kind) {
    switch (kind) {
        case MmaOperandKind::FP16:
        case MmaOperandKind::BF16: return 16;
        case MmaOperandKind::TF32: return 32;
        case MmaOperandKind::E4M3:
        case MmaOperandKind::E5M2:
        case MmaOperandKind::E2M3:
        case MmaOperandKind::E3M2:
        case MmaOperandKind::E2M1:
        case MmaOperandKind::S8:
        case MmaOperandKind::U8:
        case MmaOperandKind::S4:
        case MmaOperandKind::U4: return 8;
        default: return 0;
    }
}

static inline bool mma_operand_is_integer(MmaOperandKind kind) {
    return kind == MmaOperandKind::S8 || kind == MmaOperandKind::U8 ||
           kind == MmaOperandKind::S4 || kind == MmaOperandKind::U4;
}

static inline bool mma_operand_is_float(MmaOperandKind kind) {
    return !mma_operand_is_integer(kind) && kind != MmaOperandKind::Invalid;
}

static inline int mma_operand_file_debug_type(MmaOperandKind kind) {
    switch (kind) {
        case MmaOperandKind::FP16: return 1;
        case MmaOperandKind::BF16: return 2;
        case MmaOperandKind::TF32: return 0;
        case MmaOperandKind::E4M3: return 3;
        case MmaOperandKind::E2M1: return 4;
        case MmaOperandKind::E5M2: return 5;
        case MmaOperandKind::E3M2: return 6;
        case MmaOperandKind::E2M3: return 7;
        case MmaOperandKind::S8: return 10;
        case MmaOperandKind::U8: return 11;
        case MmaOperandKind::S4: return 12;
        case MmaOperandKind::U4: return 13;
        default: return 8;
    }
}

static inline bool mma_operand_needs_e2m1_centered_storage(MmaOperandKind kind) {
    return kind == MmaOperandKind::E2M1;
}

static inline void mma_descriptor_clear(MmaDescriptor* desc) {
    if (!desc) return;
    memset(desc, 0, sizeof(*desc));
    desc->id = -1;
    desc->category = MmaCategory::Unknown;
    desc->shape = MmaShape::Invalid;
    desc->a_kind = MmaOperandKind::Invalid;
    desc->b_kind = MmaOperandKind::Invalid;
}

static inline bool mma_fill_legacy_descriptor(int dtype, MmaDescriptor* desc) {
    if (!desc) return false;
    mma_descriptor_clear(desc);
    desc->id = dtype;
    desc->category = MmaCategory::Legacy;
    desc->is_legacy = true;

    switch (dtype) {
        case MMA_DTYPE_INT8_LEGACY:
            desc->shape = MmaShape::M16N8K32;
            desc->a_kind = MmaOperandKind::S8;
            desc->b_kind = MmaOperandKind::S8;
            snprintf(desc->name, sizeof(desc->name), "INT8_LEGACY");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "int8");
            return true;
        case MMA_DTYPE_FP16:
            desc->shape = MmaShape::M16N8K16;
            desc->a_kind = MmaOperandKind::FP16;
            desc->b_kind = MmaOperandKind::FP16;
            desc->allow_fp16_output = true;
            snprintf(desc->name, sizeof(desc->name), "FP16");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "fp16");
            return true;
        case MMA_DTYPE_BF16:
            desc->shape = MmaShape::M16N8K16;
            desc->a_kind = MmaOperandKind::BF16;
            desc->b_kind = MmaOperandKind::BF16;
            snprintf(desc->name, sizeof(desc->name), "BF16");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "bf16");
            return true;
        case MMA_DTYPE_TF32:
            desc->shape = MmaShape::M16N8K8;
            desc->a_kind = MmaOperandKind::TF32;
            desc->b_kind = MmaOperandKind::TF32;
            snprintf(desc->name, sizeof(desc->name), "TF32");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "tf32");
            return true;
        case MMA_DTYPE_INT8_UINT8_LEGACY:
            desc->shape = MmaShape::M16N8K32;
            desc->a_kind = MmaOperandKind::S8;
            desc->b_kind = MmaOperandKind::U8;
            snprintf(desc->name, sizeof(desc->name), "I8U8_LEGACY");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "i8u8");
            return true;
        case MMA_DTYPE_UINT8_INT8_LEGACY:
            desc->shape = MmaShape::M16N8K32;
            desc->a_kind = MmaOperandKind::U8;
            desc->b_kind = MmaOperandKind::S8;
            snprintf(desc->name, sizeof(desc->name), "U8I8_LEGACY");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "u8i8");
            return true;
        case MMA_DTYPE_UINT8_UINT8_LEGACY:
            desc->shape = MmaShape::M16N8K32;
            desc->a_kind = MmaOperandKind::U8;
            desc->b_kind = MmaOperandKind::U8;
            snprintf(desc->name, sizeof(desc->name), "U8U8_LEGACY");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "u8u8");
            return true;
        case MMA_DTYPE_NVFP4:
            desc->shape = MmaShape::M16N8K64;
            desc->a_kind = MmaOperandKind::E2M1;
            desc->b_kind = MmaOperandKind::E2M1;
            desc->uses_scale = true;
            snprintf(desc->name, sizeof(desc->name), "NVFP4");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "mxf4nvf4_nvfp4");
            return true;
        case MMA_DTYPE_MXFP4:
            desc->shape = MmaShape::M16N8K64;
            desc->a_kind = MmaOperandKind::E2M1;
            desc->b_kind = MmaOperandKind::E2M1;
            desc->uses_scale = true;
            snprintf(desc->name, sizeof(desc->name), "MXFP4");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "mxf4nvf4_mxfp4");
            return true;
        case MMA_DTYPE_MXINT4:
            desc->shape = MmaShape::M16N8K64;
            desc->a_kind = MmaOperandKind::S4;
            desc->b_kind = MmaOperandKind::S4;
            desc->uses_scale = true;
            snprintf(desc->name, sizeof(desc->name), "MXINT4");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "mxint4");
            return true;
        default:
            mma_descriptor_clear(desc);
            return false;
    }
}

static inline bool mma_fill_rtl_int4_descriptor(int dtype, MmaDescriptor* desc) {
    if (!desc) return false;
    mma_descriptor_clear(desc);

    desc->id = dtype;
    desc->category = MmaCategory::Integer;
    desc->shape = MmaShape::M16N8K64;
    desc->allow_fp16_output = false;

    switch (dtype) {
        case MMA_DTYPE_INT4_INT4_RTL:
            desc->a_kind = MmaOperandKind::S4;
            desc->b_kind = MmaOperandKind::S4;
            snprintf(desc->name, sizeof(desc->name), "I4I4I32");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "m16n8k64_s4_s4");
            return true;
        case MMA_DTYPE_INT4_UINT4_RTL:
            desc->a_kind = MmaOperandKind::S4;
            desc->b_kind = MmaOperandKind::U4;
            snprintf(desc->name, sizeof(desc->name), "I4U4I32");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "m16n8k64_s4_u4");
            return true;
        case MMA_DTYPE_UINT4_INT4_RTL:
            desc->a_kind = MmaOperandKind::U4;
            desc->b_kind = MmaOperandKind::S4;
            snprintf(desc->name, sizeof(desc->name), "U4I4I32");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "m16n8k64_u4_s4");
            return true;
        case MMA_DTYPE_UINT4_UINT4_RTL:
            desc->a_kind = MmaOperandKind::U4;
            desc->b_kind = MmaOperandKind::U4;
            snprintf(desc->name, sizeof(desc->name), "U4U4I32");
            snprintf(desc->file_stem, sizeof(desc->file_stem), "m16n8k64_u4_u4");
            return true;
        default:
            return false;
    }
}

static inline bool mma_fill_mixed_descriptor(int dtype, MmaDescriptor* desc) {
    if (!desc) return false;
    mma_descriptor_clear(desc);

    if (dtype >= MMA_DTYPE_F8F6F4_MIXED_BASE &&
        dtype < MMA_DTYPE_F8F6F4_MIXED_BASE + MMA_DTYPE_F8F6F4_MIXED_COUNT) {
        static constexpr MmaOperandKind kinds[5] = {
            MmaOperandKind::E4M3,
            MmaOperandKind::E5M2,
            MmaOperandKind::E2M3,
            MmaOperandKind::E3M2,
            MmaOperandKind::E2M1,
        };
        int offset = dtype - MMA_DTYPE_F8F6F4_MIXED_BASE;
        int a_idx = offset / 5;
        int b_idx = offset % 5;
        desc->id = dtype;
        desc->category = MmaCategory::MixedFloat;
        desc->shape = MmaShape::M16N8K32;
        desc->a_kind = kinds[a_idx];
        desc->b_kind = kinds[b_idx];
        desc->allow_fp16_output = true;
        snprintf(desc->name, sizeof(desc->name), "F8F6F4_%s_X_%s",
                 mma_operand_kind_token(desc->a_kind),
                 mma_operand_kind_token(desc->b_kind));
        snprintf(desc->file_stem, sizeof(desc->file_stem), "%s_%s_%s",
                 mma_shape_token(desc->shape),
                 mma_operand_kind_token(desc->a_kind),
                 mma_operand_kind_token(desc->b_kind));
        return true;
    }

    if (dtype >= MMA_DTYPE_MXF8F6F4_MIXED_BASE &&
        dtype < MMA_DTYPE_MXF8F6F4_MIXED_BASE + MMA_DTYPE_MXF8F6F4_MIXED_COUNT) {
        static constexpr MmaOperandKind kinds[5] = {
            MmaOperandKind::E4M3,
            MmaOperandKind::E5M2,
            MmaOperandKind::E2M3,
            MmaOperandKind::E3M2,
            MmaOperandKind::E2M1,
        };
        int offset = dtype - MMA_DTYPE_MXF8F6F4_MIXED_BASE;
        int a_idx = offset / 5;
        int b_idx = offset % 5;
        desc->id = dtype;
        desc->category = MmaCategory::MixedFloat;
        desc->shape = MmaShape::M16N8K32;
        desc->a_kind = kinds[a_idx];
        desc->b_kind = kinds[b_idx];
        desc->uses_scale = true;
        desc->allow_fp16_output = false;
        snprintf(desc->name, sizeof(desc->name), "MXF8F6F4_%s_X_%s",
                 mma_operand_kind_token(desc->a_kind),
                 mma_operand_kind_token(desc->b_kind));
        snprintf(desc->file_stem, sizeof(desc->file_stem), "mxf8f6f4_%s_%s",
                 mma_operand_kind_token(desc->a_kind),
                 mma_operand_kind_token(desc->b_kind));
        return true;
    }

    return false;
}

static inline bool mma_fill_integer_descriptor(int dtype, MmaDescriptor* desc) {
    if (!desc) return false;
    mma_descriptor_clear(desc);

    int base = -1;
    bool nibble = false;
    if (dtype >= MMA_DTYPE_IMMA_U8S8_BASE && dtype < MMA_DTYPE_IMMA_U8S8_BASE + MMA_DTYPE_IMMA_U8S8_COUNT) {
        base = MMA_DTYPE_IMMA_U8S8_BASE;
        nibble = false;
    } else if (dtype >= MMA_DTYPE_IMMA_U4S4_BASE && dtype < MMA_DTYPE_IMMA_U4S4_BASE + MMA_DTYPE_IMMA_U4S4_COUNT) {
        base = MMA_DTYPE_IMMA_U4S4_BASE;
        nibble = true;
    } else {
        return false;
    }

    int offset = dtype - base;
    int shape_idx = offset / 8;
    int local = offset % 8;
    int combo = local / 2;
    bool sat = (local % 2) != 0;

    desc->id = dtype;
    desc->category = MmaCategory::Integer;
    desc->satfinite = sat;
    desc->allow_fp16_output = false;

    if (!nibble) {
        static constexpr MmaShape shapes[3] = {
            MmaShape::M8N8K16,
            MmaShape::M16N8K16,
            MmaShape::M16N8K32,
        };
        desc->shape = shapes[shape_idx];
        desc->a_kind = (combo & 0x2) ? MmaOperandKind::U8 : MmaOperandKind::S8;
        desc->b_kind = (combo & 0x1) ? MmaOperandKind::U8 : MmaOperandKind::S8;
    } else {
        static constexpr MmaShape shapes[3] = {
            MmaShape::M8N8K32,
            MmaShape::M16N8K32,
            MmaShape::M16N8K64,
        };
        desc->shape = shapes[shape_idx];
        desc->a_kind = (combo & 0x2) ? MmaOperandKind::U4 : MmaOperandKind::S4;
        desc->b_kind = (combo & 0x1) ? MmaOperandKind::U4 : MmaOperandKind::S4;
    }

    snprintf(desc->name, sizeof(desc->name), "%s_%s_X_%s%s",
             mma_shape_token(desc->shape),
             mma_operand_kind_token(desc->a_kind),
             mma_operand_kind_token(desc->b_kind),
             desc->satfinite ? "_satfinite" : "");
    snprintf(desc->file_stem, sizeof(desc->file_stem), "%s_%s_%s%s",
             mma_shape_token(desc->shape),
             mma_operand_kind_token(desc->a_kind),
             mma_operand_kind_token(desc->b_kind),
             desc->satfinite ? "_sat" : "");
    return true;
}

static inline bool mma_lookup_descriptor(int dtype, MmaDescriptor* desc) {
    return mma_fill_legacy_descriptor(dtype, desc) ||
           mma_fill_rtl_int4_descriptor(dtype, desc) ||
           mma_fill_mixed_descriptor(dtype, desc) ||
           mma_fill_integer_descriptor(dtype, desc);
}

static inline const MmaDescriptor* mma_get_descriptor(int dtype) {
    static thread_local MmaDescriptor desc;
    if (!mma_lookup_descriptor(dtype, &desc)) {
        return nullptr;
    }
    return &desc;
}

static inline bool mma_is_known_dtype(int dtype) {
    return mma_get_descriptor(dtype) != nullptr;
}

static inline bool mma_is_new_mixed_dtype(int dtype) {
    MmaDescriptor desc;
    return mma_fill_mixed_descriptor(dtype, &desc);
}

static inline bool mma_is_new_integer_dtype(int dtype) {
    MmaDescriptor desc;
    return mma_fill_integer_descriptor(dtype, &desc);
}

static inline int mma_dtype_deprecated_replacement(int dtype);

static inline bool mma_dtype_is_deprecated(int dtype) {
    return mma_dtype_deprecated_replacement(dtype) >= 0;
}

static inline int mma_dtype_deprecated_replacement(int dtype) {
    switch (dtype) {
        case 7: return 57;
        case 8: return 63;
        case 11: return 69;
        case 12: return 75;
        case 13: return 32;
        case 14: return 38;
        default: return -1;
    }
}

static inline const char* mma_dtype_name(int dtype) {
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    return desc ? desc->name : "UNKNOWN";
}

static inline const char* mma_dtype_file_stem(int dtype) {
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    return desc ? desc->file_stem : "unknown";
}
