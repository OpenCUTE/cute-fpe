#include "fmac.h"
#include "FileManager.h"
#include "mma_descriptor.h"
#include "myrandom.h"
#include "nv_mac.h"
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>  // ← 添加这一行
#include "random_allinone.h"

#ifndef SCALE_OUTPUT_DATA_DIR
#define SCALE_OUTPUT_DATA_DIR "scale_output_data"
#endif

// using namespace std;
// #define DEBUG
#define NEW_DIFF_PERCENT 0.0000045 // < 2^(-21) 用来测4个元素的向量
// #define NEW_DIFF_PERCENT 0.0000009 // < 2^(-20) 用来测8个元素的向量
// #define NEW_DIFF_PERCENT 0.0000018 // < 2^(-19) 用来测16个元素的向量
// #define NEW_DIFF_PERCENT 0.0000036 // < 2^(-18) 用来测32个元素的向量

// dtype:
// 0: INT8
// 1: FP16
// 2: BF16
// 3: TF32

typedef struct{
    struct DataFileWorkspace* workspace;
    FileIntArrayManager* a; // 文件管理器
    FileIntArrayManager* a_scale;
    FileIntArrayManager* b; // 文件管理器
    FileIntArrayManager* b_scale;
    FileIntArrayManager* c; // 文件管理器
    FileIntArrayManager* d; // 文件管理器
} DataFile;

enum DataFileWorkspaceSlot {
    DATA_WS_A = 0,
    DATA_WS_B,
    DATA_WS_C,
    DATA_WS_D,
    DATA_WS_D_NV,
    DATA_WS_A_PAYLOAD,
    DATA_WS_B_PAYLOAD,
    DATA_WS_A_SCALE,
    DATA_WS_B_SCALE,
    DATA_WS_A_SCALE_FILE,
    DATA_WS_B_SCALE_FILE,
    DATA_WS_A_FP4,
    DATA_WS_B_FP4,
    DATA_WS_SLOT_COUNT
};

typedef struct DataFileWorkspace {
    void* slots[DATA_WS_SLOT_COUNT];
    size_t capacities[DATA_WS_SLOT_COUNT];
} DataFileWorkspace;

static inline DataFileWorkspace* ensure_data_file_workspace(DataFile* df) {
    if (!df) return NULL;
    if (!df->workspace) {
        df->workspace = (DataFileWorkspace*)calloc(1, sizeof(DataFileWorkspace));
    }
    return df->workspace;
}

static inline void* data_file_workspace_reserve(
    DataFile* df,
    DataFileWorkspaceSlot slot,
    size_t bytes,
    bool zero_fill = false)
{
    if (bytes == 0) {
        return NULL;
    }
    DataFileWorkspace* ws = ensure_data_file_workspace(df);
    if (!ws) {
        return NULL;
    }
    if (ws->capacities[slot] < bytes) {
        void* new_ptr = realloc(ws->slots[slot], bytes);
        if (!new_ptr) {
            return NULL;
        }
        ws->slots[slot] = new_ptr;
        ws->capacities[slot] = bytes;
    }
    if (zero_fill) {
        memset(ws->slots[slot], 0, bytes);
    }
    return ws->slots[slot];
}

template <typename T>
static inline T* data_file_workspace_reserve_typed(
    DataFile* df,
    DataFileWorkspaceSlot slot,
    size_t count,
    bool zero_fill = false)
{
    return (T*)data_file_workspace_reserve(df, slot, count * sizeof(T), zero_fill);
}

static inline void destroy_data_file_workspace(DataFile* df) {
    if (!df || !df->workspace) {
        return;
    }
    for (int i = 0; i < DATA_WS_SLOT_COUNT; ++i) {
        free(df->workspace->slots[i]);
    }
    free(df->workspace);
    df->workspace = NULL;
}

typedef struct {
    int seen;
    int dtype;
    int vecnum;
    int32_t my_result;
    int32_t nv_result;
    const char* tag;
} MismatchRecord;

static inline MismatchRecord& last_mismatch_record() {
    static MismatchRecord record = {0, -1, -1, 0, 0, nullptr};
    return record;
}

static inline void clear_last_mismatch_record() {
    last_mismatch_record() = {0, -1, -1, 0, 0, nullptr};
}

static inline void record_mac_mismatch(
    const char* tag,
    int dtype,
    int vecnum,
    int32_t my_result,
    int32_t nv_result)
{
    last_mismatch_record() = {1, dtype, vecnum, my_result, nv_result, tag};
}

static inline bool dump_mismatch_input_enabled() {
    return getenv("FMAC_DUMP_MISMATCH_INPUT") != NULL;
}

static inline void print_u8_array_literal(const char* name, const void* data, int count) {
    const uint8_t* bytes = (const uint8_t*)data;
    printf("%s=[", name);
    for (int i = 0; i < count; ++i) {
        printf("%s0x%02x", i ? "," : "", (unsigned)bytes[i]);
    }
    printf("]\n");
}

static inline void dump_scale_mismatch_input(
    const MmaDescriptor& desc,
    int vecnum,
    const void* a_payload,
    int a_logical,
    const void* a_physical,
    int a_physical_count,
    const void* b_payload,
    int b_logical,
    const void* b_physical,
    int b_physical_count,
    const int8_t* a_scale,
    const int8_t* b_scale,
    int scale_valid_num,
    int32_t c_value,
    int32_t my_result,
    int32_t nv_result)
{
    if (!dump_mismatch_input_enabled()) return;
    printf("FMAC_MISMATCH_INPUT_BEGIN\n");
    printf("dtype=%d name=%s vec=%d a_kind=%s b_kind=%s shape=%s\n",
           desc.id,
           desc.name,
           vecnum,
           mma_operand_kind_token(desc.a_kind),
           mma_operand_kind_token(desc.b_kind),
           mma_shape_token(desc.shape));
    printf("c=0x%08x my=0x%08x nv=0x%08x\n",
           (uint32_t)c_value,
           (uint32_t)my_result,
           (uint32_t)nv_result);
    print_u8_array_literal("a", a_payload, a_logical);
    print_u8_array_literal("b", b_payload, b_logical);
    print_u8_array_literal("a_payload", a_payload, a_logical);
    print_u8_array_literal("b_payload", b_payload, b_logical);
    print_u8_array_literal("a_physical", a_physical, a_physical_count);
    print_u8_array_literal("b_physical", b_physical, b_physical_count);
    print_u8_array_literal("a_scale", a_scale, scale_valid_num);
    print_u8_array_literal("b_scale", b_scale, scale_valid_num);
    printf("FMAC_MISMATCH_INPUT_END\n");
}

static inline void dump_descriptor_mismatch_input(
    const MmaDescriptor& desc,
    int vecnum,
    const void* a_payload,
    int a_logical_elems,
    const void* a_physical,
    int a_physical_elems,
    const void* b_payload,
    int b_logical_elems,
    const void* b_physical,
    int b_physical_elems,
    const int8_t* a_scale,
    const int8_t* b_scale,
    int scale_valid_num,
    int32_t c_value,
    int32_t my_result,
    int32_t nv_result,
    bool dst_fp16)
{
    if (!dump_mismatch_input_enabled()) return;
    int a_elem_bytes = mma_operand_storage_bits(desc.a_kind) / 8;
    int b_elem_bytes = mma_operand_storage_bits(desc.b_kind) / 8;
    int a_logical_bytes = a_logical_elems * a_elem_bytes;
    int b_logical_bytes = b_logical_elems * b_elem_bytes;
    int a_physical_bytes = a_physical_elems * a_elem_bytes;
    int b_physical_bytes = b_physical_elems * b_elem_bytes;

    printf("FMAC_MISMATCH_INPUT_BEGIN\n");
    printf("dtype=%d name=%s vec=%d a_kind=%s b_kind=%s shape=%s dst=%s\n",
           desc.id,
           desc.name,
           vecnum,
           mma_operand_kind_token(desc.a_kind),
           mma_operand_kind_token(desc.b_kind),
           mma_shape_token(desc.shape),
           dst_fp16 ? "fp16" : "fp32");
    printf("c=0x%08x my=0x%08x nv=0x%08x\n",
           (uint32_t)c_value,
           (uint32_t)my_result,
           (uint32_t)nv_result);
    print_u8_array_literal("a", a_payload, a_logical_bytes);
    print_u8_array_literal("b", b_payload, b_logical_bytes);
    print_u8_array_literal("a_payload", a_payload, a_logical_bytes);
    print_u8_array_literal("b_payload", b_payload, b_logical_bytes);
    print_u8_array_literal("a_physical", a_physical, a_physical_bytes);
    print_u8_array_literal("b_physical", b_physical, b_physical_bytes);
    print_u8_array_literal("a_scale", a_scale, scale_valid_num);
    print_u8_array_literal("b_scale", b_scale, scale_valid_num);
    printf("FMAC_MISMATCH_INPUT_END\n");
}

static inline bool last_call_had_mismatch() {
    return last_mismatch_record().seen != 0;
}

static inline const MismatchRecord& get_last_mismatch_record() {
    return last_mismatch_record();
}

static inline bool use_topology_preserving_128b_adapter() {
    return mma_topology_128_input_enabled();
}

static inline bool& reference_validation_override() {
    static bool enabled = false;
    return enabled;
}

static inline void set_reference_validation_enabled(bool enabled) {
    reference_validation_override() = enabled;
}

static inline bool reference_validation_enabled() {
#if NVIDIA_ON
    return true;
#else
    return reference_validation_override() || getenv("FMAC_VALIDATE_REFERENCE") != NULL;
#endif
}

static inline int32_t maybe_reference_mac(
    void* a, void* b, void* c, int size, int dtype, bool float_type, int32_t my_result)
{
    return reference_validation_enabled() ? nvmac(a, b, c, size, dtype, float_type) : my_result;
}

static inline int32_t maybe_reference_mac_scale(
    void* a, void* a_scale, void* b, void* b_scale, void* c, int size, int dtype, int32_t my_result)
{
    // MXINT4 is a custom CModel/RTL format, not a native NVIDIA MMA format.
    if (dtype == MMA_DTYPE_MXINT4) {
        return my_result;
    }
    return reference_validation_enabled() ? nvmac_scale(a, a_scale, b, b_scale, c, size, dtype) : my_result;
}

// Expand a logical 128b payload into a physical 256b payload while keeping the
// original group topology alive. Each destination group receives the logical
// group's payload in its low half and zeros in its high half.
// When the adapter is enabled, simulation consumes the expanded 256b payload,
// while exported files should still record the logical 128b payload that RTL
// is expected to source from.
template <typename T>
static inline void expand_payload_preserve_group_topology(
    const T* src, int src_size,
    T* dst, int dst_size,
    int group_count)
{
    memset(dst, 0, dst_size * sizeof(T));
    int src_group_bytes = src_size / group_count;
    int dst_group_bytes = dst_size / group_count;

    for (int g = 0; g < group_count; ++g) {
        memcpy(dst + g * dst_group_bytes,
               src + g * src_group_bytes,
               src_group_bytes * sizeof(T));
    }
}

template <typename T>
static inline void expand_mxf4_payload_128b_to_256b_two_blocks(
    const T* src, int src_size,
    T* dst, int dst_size)
{
    memset(dst, 0, dst_size * sizeof(T));
    int copy_count = src_size;
    if (copy_count > dst_size / 2) {
        copy_count = dst_size / 2;
    }
    memcpy(dst, src, copy_count * sizeof(T));
}

static inline bool mxf4_use_two_full_blocks_128b_topology(int dtype) {
    return mma_dtype_uses_mxf4_topology(dtype) && use_topology_preserving_128b_adapter();
}

static inline int mxf4_topology_effective_scale_count() {
    return mma_mxf4_blocksize16_enabled() ? 2 : 1;
}

static inline int mxf4_scale_storage_size() {
    return 4;
}

static inline int mxf4_scale_storage_size_for_bitsize(int dtype, int bitsize) {
    int per_16element_block_count = bitsize > 0 ? (bitsize / 4 / 16) : 0;
    if (mma_dtype_uses_32element_scale(dtype)) {
        return (per_16element_block_count + 1) / 2;
    }
    return per_16element_block_count;
}

static inline int mxf4_valid_scale_count_for_bitsize(int dtype, int bitsize) {
    int scale_count = mxf4_scale_storage_size_for_bitsize(dtype, bitsize);
    if (mxf4_use_two_full_blocks_128b_topology(dtype)) {
        int effective_count = mxf4_topology_effective_scale_count();
        return effective_count < scale_count ? effective_count : scale_count;
    }
    return scale_count;
}

static inline void normalize_mxf4_scale_array_for_topology(int dtype, int8_t* scale, int scale_size) {
    if (!scale || scale_size <= 0 || !mxf4_use_two_full_blocks_128b_topology(dtype)) {
        return;
    }
    if (mma_mxf4_blocksize16_enabled()) {
        if (scale_size > 2) {
            for (int i = 2; i < scale_size; ++i) {
                scale[i] = 0;
            }
        }
    } else {
        if (scale_size > 1) {
            scale[1] = scale[0];
        }
        if (scale_size > 2) {
            for (int i = 2; i < scale_size; ++i) {
                scale[i] = 0;
            }
        }
    }
}

template <typename T>
static inline void expand_payload_for_dtype(
    int dtype,
    const T* src,
    int src_size,
    T* dst,
    int dst_size)
{
    if (mxf4_use_two_full_blocks_128b_topology(dtype)) {
        expand_mxf4_payload_128b_to_256b_two_blocks(src, src_size, dst, dst_size);
        return;
    }
    int group_count = mma_dtype_uses_mxf4_topology(dtype) ? 4 : 2;
    expand_payload_preserve_group_topology(src, src_size, dst, dst_size, group_count);
}

static inline int topology_group_count_for_dtype(int dtype) {
    return mma_dtype_uses_mxf4_topology(dtype) ? 4 : 2;
}

static inline string scale_output_path_for(const string& operand_fname) {
    size_t pos = operand_fname.find_last_of('/');
    const string basename = (pos == string::npos) ? operand_fname : operand_fname.substr(pos + 1);
    return string(SCALE_OUTPUT_DATA_DIR) + "/" + basename;
}

static inline FileIntArrayManager::DataType file_manager_dtype_for_operand_kind(MmaOperandKind kind) {
    switch (mma_operand_storage_bits(kind)) {
        case 16: return FileIntArrayManager::INT16;
        case 32: return FileIntArrayManager::INT32;
        default: return FileIntArrayManager::INT8;
    }
}

static inline DataFile* create_data_file_ex(
    const string& a_fname,
    const string& b_fname,
    const string& c_fname,
    const string& d_fname,
    FileIntArrayManager::DataType a_dtype,
    FileIntArrayManager::DataType b_dtype,
    FileIntArrayManager::DataType c_dtype,
    FileIntArrayManager::DataType d_dtype,
    int is_create,
    bool need_scale = false)
{
    DataFile* df = (DataFile*)calloc(1, sizeof(DataFile));
    if (!df) return NULL;

    df->a = new FileIntArrayManager(a_fname, a_dtype, is_create, FileIntArrayManager::DUMP_BIN);
    df->b = new FileIntArrayManager(b_fname, b_dtype, is_create, FileIntArrayManager::DUMP_BIN);
    df->c = new FileIntArrayManager(c_fname, c_dtype, is_create, FileIntArrayManager::DUMP_BIN);
    df->d = new FileIntArrayManager(
        d_fname, d_dtype, is_create,
        FileIntArrayManager::DUMP_TEXT | FileIntArrayManager::DUMP_BIN);
    if (need_scale) {
        df->a_scale = new FileIntArrayManager(scale_output_path_for(a_fname), FileIntArrayManager::INT8, is_create, FileIntArrayManager::DUMP_BIN);
        df->b_scale = new FileIntArrayManager(scale_output_path_for(b_fname), FileIntArrayManager::INT8, is_create, FileIntArrayManager::DUMP_BIN);
    }
    return df;
}

DataFile* create_data_file(string a_fname, string b_fname, string c_fname, string d_fname, int dtype, int is_create, bool float_type = 0) {
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (desc) {
        FileIntArrayManager::DataType cd_dtype = float_type ? FileIntArrayManager::INT16 : FileIntArrayManager::INT32;
        return create_data_file_ex(
            a_fname,
            b_fname,
            c_fname,
            d_fname,
            file_manager_dtype_for_operand_kind(desc->a_kind),
            file_manager_dtype_for_operand_kind(desc->b_kind),
            cd_dtype,
            cd_dtype,
            is_create,
            desc->uses_scale);
    }

    DataFile* df = (DataFile*)calloc(1, sizeof(DataFile));
    if (!df) return NULL;
    if (mma_dtype_uses_mxf4_topology(dtype)) {
        df->a_scale = new FileIntArrayManager(scale_output_path_for(a_fname), FileIntArrayManager::INT8, is_create, FileIntArrayManager::DUMP_BIN);
        df->b_scale = new FileIntArrayManager(scale_output_path_for(b_fname), FileIntArrayManager::INT8, is_create, FileIntArrayManager::DUMP_BIN);
    }
    if((dtype == 0) || (dtype == 4) || (dtype == 5) || (dtype == 6) || mma_dtype_uses_mxf4_topology(dtype)){
        df->a = new FileIntArrayManager(a_fname, FileIntArrayManager::INT8, is_create, FileIntArrayManager::DUMP_BIN);
        df->b = new FileIntArrayManager(b_fname, FileIntArrayManager::INT8, is_create, FileIntArrayManager::DUMP_BIN);
    }
    else if((dtype == 1) || (dtype == 2)){
        df->a = new FileIntArrayManager(a_fname, FileIntArrayManager::INT16, is_create, FileIntArrayManager::DUMP_BIN);
        df->b = new FileIntArrayManager(b_fname, FileIntArrayManager::INT16, is_create, FileIntArrayManager::DUMP_BIN);
    }
    else{
        df->a = new FileIntArrayManager(a_fname, FileIntArrayManager::INT32, is_create, FileIntArrayManager::DUMP_BIN);
        df->b = new FileIntArrayManager(b_fname, FileIntArrayManager::INT32, is_create, FileIntArrayManager::DUMP_BIN);
    }
    
    if(!float_type){
        df->c = new FileIntArrayManager(c_fname, FileIntArrayManager::INT32, is_create, FileIntArrayManager::DUMP_BIN);
        df->d = new FileIntArrayManager(
            d_fname, FileIntArrayManager::INT32, is_create,
            FileIntArrayManager::DUMP_TEXT | FileIntArrayManager::DUMP_BIN);
    } 
    else{
        df->c = new FileIntArrayManager(c_fname, FileIntArrayManager::INT16, is_create, FileIntArrayManager::DUMP_BIN);
        df->d = new FileIntArrayManager(
            d_fname, FileIntArrayManager::INT16, is_create,
            FileIntArrayManager::DUMP_TEXT | FileIntArrayManager::DUMP_BIN);
    }
    return df;
}

static inline bool read_binary_blob_at(const char* path, uint64_t offset, void* dst, size_t bytes) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return false;
    }
    if (fseek(fp, static_cast<long>(offset), SEEK_SET) != 0) {
        perror("fseek failed");
        fclose(fp);
        return false;
    }
    bool ok = fread(dst, 1, bytes, fp) == bytes;
    if (!ok) {
        if (ferror(fp)) {
            perror("fread failed");
        } else {
            fprintf(stderr,
                    "Short read from %s at offset %llu: expected %zu bytes\n",
                    path,
                    (unsigned long long)offset,
                    bytes);
        }
    }
    fclose(fp);
    return ok;
}

static inline bool read_int8_vector_from_bin(const char* path, int vecnum, int elem_count, int8_t* dst) {
    uint64_t offset = static_cast<uint64_t>(vecnum) * static_cast<uint64_t>(elem_count);
    return read_binary_blob_at(path, offset, dst, static_cast<size_t>(elem_count));
}

static inline bool read_packed_nibble_vector_from_bin(const char* path, int vecnum, int elem_count, int8_t* dst) {
    int packed_count = (elem_count + 1) / 2;
    uint64_t offset = static_cast<uint64_t>(vecnum) * static_cast<uint64_t>(packed_count);
    uint8_t* raw = (uint8_t*)malloc(static_cast<size_t>(packed_count) * sizeof(uint8_t));
    if (!raw) return false;
    bool ok = read_binary_blob_at(path, offset, raw, static_cast<size_t>(packed_count));
    if (ok) {
        for (int i = 0; i < packed_count; ++i) {
            dst[2 * i] = static_cast<int8_t>(raw[i] & 0x0f);
            if (2 * i + 1 < elem_count) {
                dst[2 * i + 1] = static_cast<int8_t>((raw[i] >> 4) & 0x0f);
            }
        }
    }
    free(raw);
    return ok;
}

static inline bool read_int16_vector_from_bin(const char* path, int vecnum, int elem_count, int16_t* dst) {
    uint64_t offset = static_cast<uint64_t>(vecnum) * static_cast<uint64_t>(elem_count) * sizeof(uint16_t);
    uint16_t* raw = (uint16_t*)malloc(static_cast<size_t>(elem_count) * sizeof(uint16_t));
    if (!raw) return false;
    bool ok = read_binary_blob_at(path, offset, raw, static_cast<size_t>(elem_count) * sizeof(uint16_t));
    if (ok) {
        for (int i = 0; i < elem_count; ++i) {
            dst[i] = static_cast<int16_t>(__builtin_bswap16(raw[i]));
        }
    }
    free(raw);
    return ok;
}

static inline bool read_int32_vector_from_bin(const char* path, int vecnum, int elem_count, int32_t* dst) {
    uint64_t offset = static_cast<uint64_t>(vecnum) * static_cast<uint64_t>(elem_count) * sizeof(uint32_t);
    uint32_t* raw = (uint32_t*)malloc(static_cast<size_t>(elem_count) * sizeof(uint32_t));
    if (!raw) return false;
    bool ok = read_binary_blob_at(path, offset, raw, static_cast<size_t>(elem_count) * sizeof(uint32_t));
    if (ok) {
        for (int i = 0; i < elem_count; ++i) {
            dst[i] = static_cast<int32_t>(__builtin_bswap32(raw[i]));
        }
    }
    free(raw);
    return ok;
}

static inline int logical_elem_count_for_desc(const MmaDescriptor& desc) {
    return mma_shape_k(desc.shape);
}

static inline int logical_128b_elem_count_for_kind(MmaOperandKind kind) {
    switch (kind) {
        case MmaOperandKind::FP16:
        case MmaOperandKind::BF16:
            return 8;
        case MmaOperandKind::TF32:
            return 4;
        case MmaOperandKind::E4M3:
        case MmaOperandKind::E5M2:
        case MmaOperandKind::E2M3:
        case MmaOperandKind::E3M2:
            return 16;
        case MmaOperandKind::E2M1:
            return 32;
        case MmaOperandKind::S8:
        case MmaOperandKind::U8:
            return 16;
        case MmaOperandKind::S4:
        case MmaOperandKind::U4:
            return 32;
        default:
            return 0;
    }
}

static inline int physical_elem_count_for_desc(const MmaDescriptor& desc) {
    return mma_shape_k(desc.shape);
}

static inline int elem_count_for_bitsize(MmaOperandKind kind, int bitsize) {
    int elem_bits = mma_operand_storage_bits(kind);
    if (elem_bits <= 0 || bitsize <= 0) {
        return 0;
    }
    return bitsize / elem_bits;
}

static inline int physical_elem_count_for_desc_bitsize(const MmaDescriptor& desc, int bitsize) {
    // RTL int4 vectors use one byte per logical nibble in host memory.  A
    // positive runtime override makes the requested dot-product length exact,
    // including lengths that cannot be represented by a whole 32-bit word.
    if (mma_is_rtl_int4_dtype(desc.id) && mma_rtl_int4_k() > 0) {
        return mma_rtl_int4_k();
    }
    int a_count = elem_count_for_bitsize(desc.a_kind, bitsize);
    int b_count = elem_count_for_bitsize(desc.b_kind, bitsize);
    if (a_count > 0 && a_count == b_count) {
        return a_count;
    }
    return physical_elem_count_for_desc(desc);
}

static inline int logical_elem_count_for_desc_operand(const MmaDescriptor& desc, MmaOperandKind kind) {
    int physical = physical_elem_count_for_desc(desc);
    if (mma_is_rtl_int4_dtype(desc.id)) {
        return physical;
    }
    if (!use_topology_preserving_128b_adapter()) {
        return physical;
    }
    int logical = physical;
    switch (desc.shape) {
        case MmaShape::M16N8K8:
            logical = (kind == MmaOperandKind::TF32) ? 4 : physical;
            break;
        case MmaShape::M16N8K16:
            if (kind == MmaOperandKind::FP16 || kind == MmaOperandKind::BF16) {
                logical = 8;
            } else {
                logical = physical;
            }
            break;
        case MmaShape::M8N8K16:
            logical = physical;
            break;
        case MmaShape::M8N8K32:
            logical = physical;
            break;
        case MmaShape::M16N8K32:
            if (kind == MmaOperandKind::S8 || kind == MmaOperandKind::U8) {
                logical = 16;
            } else if (kind == MmaOperandKind::S4 || kind == MmaOperandKind::U4) {
                logical = 32;
            } else {
                logical = 16;
            }
            break;
        case MmaShape::M16N8K64:
            logical = 32;
            break;
        default:
            logical = physical;
            break;
    }
    if (logical <= 0 || logical > physical) {
        return physical;
    }
    return logical;
}

static inline int logical_elem_count_for_desc_operand_bitsize(
    const MmaDescriptor& desc,
    MmaOperandKind kind,
    int bitsize)
{
    int physical = physical_elem_count_for_desc_bitsize(desc, bitsize);
    if (mma_is_rtl_int4_dtype(desc.id)) {
        return physical;
    }
    if (!use_topology_preserving_128b_adapter()) {
        return physical;
    }

    int desc_physical = physical_elem_count_for_desc(desc);
    int desc_logical = logical_elem_count_for_desc_operand(desc, kind);
    if (desc_physical <= 0 || desc_logical <= 0 || desc_logical >= desc_physical) {
        return physical;
    }

    int logical = (physical * desc_logical) / desc_physical;
    if (logical <= 0 || logical > physical) {
        return physical;
    }
    return logical;
}

static inline bool desc_operand_needs_128b_expansion(const MmaDescriptor& desc, MmaOperandKind kind) {
    return logical_elem_count_for_desc_operand(desc, kind) != physical_elem_count_for_desc(desc);
}

static inline bool desc_operand_needs_128b_expansion_bitsize(
    const MmaDescriptor& desc,
    MmaOperandKind kind,
    int bitsize)
{
    return logical_elem_count_for_desc_operand_bitsize(desc, kind, bitsize) !=
           physical_elem_count_for_desc_bitsize(desc, bitsize);
}

static inline int logical_elem_count_for_desc_128b_mode(const MmaDescriptor& desc) {
    int a_logical = logical_elem_count_for_desc_operand(desc, desc.a_kind);
    int b_logical = logical_elem_count_for_desc_operand(desc, desc.b_kind);
    return a_logical > b_logical ? a_logical : b_logical;
}

static inline bool desc_needs_128b_adapter(const MmaDescriptor& desc) {
    return desc_operand_needs_128b_expansion(desc, desc.a_kind) ||
           desc_operand_needs_128b_expansion(desc, desc.b_kind);
}

static inline int replay_physical_elem_count_for_desc(const MmaDescriptor& desc, int bitsize) {
    if (desc.is_legacy && mma_dtype_uses_mxf4_topology(desc.id)) {
        return bitsize / 8;
    }
    return physical_elem_count_for_desc_bitsize(desc, bitsize);
}

static inline int replay_logical_elem_count_for_desc_operand(
    const MmaDescriptor& desc,
    MmaOperandKind kind,
    int bitsize)
{
    if (desc.is_legacy && mma_dtype_uses_mxf4_topology(desc.id)) {
        int physical = replay_physical_elem_count_for_desc(desc, bitsize);
        return use_topology_preserving_128b_adapter() ? (physical / 2) : physical;
    }
    return logical_elem_count_for_desc_operand_bitsize(desc, kind, bitsize);
}

static inline bool use_generation_descriptor_path(const MmaDescriptor* desc) {
    return desc && !desc->is_legacy && !desc->uses_scale;
}

static inline bool use_generation_scale_descriptor_path(const MmaDescriptor* desc) {
    return desc && !desc->is_legacy && desc->uses_scale;
}

static inline bool use_replay_descriptor_path(const MmaDescriptor* desc) {
    return desc && !desc->uses_scale;
}

static inline bool use_replay_scale_descriptor_path(const MmaDescriptor* desc) {
    return desc && desc->uses_scale;
}

static inline bool descriptor_integer_result(const MmaDescriptor& desc) {
    return desc.category == MmaCategory::Integer ||
           (desc.category == MmaCategory::Legacy &&
            mma_operand_is_integer(desc.a_kind) &&
            mma_operand_is_integer(desc.b_kind));
}

static inline uint32_t generate_random_operand_raw(MmaOperandKind kind) {
    switch (kind) {
        case MmaOperandKind::FP16: return gen_random_fp(FP16);
        case MmaOperandKind::BF16: return gen_random_fp(BF16);
        case MmaOperandKind::TF32: return gen_random_fp(FP32) & 0xFFFFE000u;
        case MmaOperandKind::E4M3: return gen_random_fp(E4M3);
        case MmaOperandKind::E5M2: return gen_random_fp(E5M2);
        case MmaOperandKind::E2M3: return gen_random_fp(E2M3);
        case MmaOperandKind::E3M2: return gen_random_fp(E3M2);
        case MmaOperandKind::E2M1: return (gen_random_fp(E2M1) & 0x0fu) << 2;
        case MmaOperandKind::S8: return (uint8_t)(rand() % 256 - 128);
        case MmaOperandKind::U8: return (uint8_t)(rand() % 256);
        case MmaOperandKind::S4: return (uint8_t)(rand() % 16);
        case MmaOperandKind::U4: return (uint8_t)(rand() % 16);
        default: return 0;
    }
}

static inline int32_t generate_random_s32() {
    uint32_t upper = (uint32_t)(rand() & 0xffff);
    uint32_t lower = (uint32_t)(rand() & 0xffff);
    return (int32_t)((upper << 16) | lower);
}

static inline uint32_t generate_sequential_operand_raw(MmaOperandKind kind, int idx) {
    switch (kind) {
        case MmaOperandKind::FP16: return (uint16_t)gen_random_fp(FP16);
        case MmaOperandKind::BF16: return (uint16_t)gen_random_fp(BF16);
        case MmaOperandKind::TF32: {
            float v = (float)idx;
            uint32_t raw = 0;
            memcpy(&raw, &v, sizeof(raw));
            return raw & 0xFFFFE000u;
        }
        case MmaOperandKind::E2M1: return (uint8_t)((idx & 0x0f) << 2);
        case MmaOperandKind::S8:
        case MmaOperandKind::U8:
        case MmaOperandKind::E4M3:
        case MmaOperandKind::E5M2:
        case MmaOperandKind::E2M3:
        case MmaOperandKind::E3M2:
        case MmaOperandKind::S4:
        case MmaOperandKind::U4:
            return (uint8_t)(idx & 0xff);
        default: return 0;
    }
}

static inline void write_operand_file(FileIntArrayManager* fm, MmaOperandKind kind, void* data, int count, bool packed_fp4_text = false) {
    if (!fm || !data) return;
    int debug_type = mma_operand_file_debug_type(kind);
    switch (mma_operand_storage_bits(kind)) {
        case 16:
            fm->write_int16_array_to_file((int16_t*)data, count, 1, debug_type);
            break;
        case 32:
            fm->write_int32_array_to_file((int32_t*)data, count, 1, debug_type);
            break;
        default:
            fm->write_int8_array_to_file((int8_t*)data, count, 1, debug_type, packed_fp4_text ? 1 : 0);
            break;
    }
}

static inline bool read_operand_vector_from_bin(const char* path, int vecnum, int elem_count, MmaOperandKind kind, void* dst) {
    switch (mma_operand_storage_bits(kind)) {
        case 16:
            return read_int16_vector_from_bin(path, vecnum, elem_count, (int16_t*)dst);
        case 32:
            return read_int32_vector_from_bin(path, vecnum, elem_count, (int32_t*)dst);
        default:
            if (kind == MmaOperandKind::S4 || kind == MmaOperandKind::U4) {
                return read_packed_nibble_vector_from_bin(path, vecnum, elem_count, (int8_t*)dst);
            }
            return read_int8_vector_from_bin(path, vecnum, elem_count, (int8_t*)dst);
    }
}

static inline void write_accumulator_file(FileIntArrayManager* fm, int32_t value, bool float_type, bool integer_result) {
    if (!fm) return;
    if (integer_result) {
        fm->write_int32_array_to_file(&value, 1, 1, 8);
    } else if (!float_type) {
        fm->write_int32_array_to_file(&value, 1, 1, 0);
    } else {
        int32_t value32 = (int32_t)(uint16_t)value;
        fm->write_int32_array_to_file(&value32, 1, 1, 1);
    }
}

static inline int scale_storage_size_for_desc(const MmaDescriptor& desc) {
    if (!desc.uses_scale) {
        return 0;
    }
    if (mma_dtype_uses_mxf4_topology(desc.id)) {
        return mma_dtype_uses_32element_scale(desc.id) ? 2 : mxf4_scale_storage_size();
    }
    if (desc.shape == MmaShape::M16N8K32) {
        return 4;
    }
    return 0;
}

static inline int valid_scale_count_for_desc(const MmaDescriptor& desc) {
    if (!desc.uses_scale) {
        return 0;
    }
    if (mma_dtype_uses_mxf4_topology(desc.id)) {
        if (mxf4_use_two_full_blocks_128b_topology(desc.id)) {
            int count = mma_dtype_uses_32element_scale(desc.id) ? 1 : mxf4_topology_effective_scale_count();
            return count;
        }
        return mma_dtype_uses_32element_scale(desc.id) ? 2 : mxf4_scale_storage_size();
    }
    if (desc.shape == MmaShape::M16N8K32) {
        return 1;
    }
    return 0;
}

static inline int valid_scale_count_for_desc_bitsize(const MmaDescriptor& desc, int bitsize) {
    if (!desc.uses_scale) {
        return 0;
    }
    if (mma_dtype_uses_mxf4_topology(desc.id)) {
        return mxf4_valid_scale_count_for_bitsize(desc.id, bitsize);
    }
    if (desc.category == MmaCategory::MixedFloat && desc.shape == MmaShape::M16N8K32) {
        int elems = physical_elem_count_for_desc_bitsize(desc, bitsize);
        return elems > 0 ? ((elems + 31) / 32) : 0;
    }
    return valid_scale_count_for_desc(desc);
}

static inline int scale_file_debug_type_for_desc(const MmaDescriptor& desc) {
    return desc.id == MMA_DTYPE_NVFP4 ? 3 : 9;
}

static inline string scale_raw_order_path(const char* path, const char* suffix) {
    string p = path ? string(path) : string();
    string s = suffix ? string(suffix) : string();
    if (p.size() >= s.size() && p.compare(p.size() - s.size(), s.size(), s) == 0) {
        return p.substr(0, p.size() - s.size()) + "_raw_order" + s;
    }
    return p + "_raw_order";
}

typedef struct {
    string path;
    FILE* fp;
} RawScaleOutputHandle;

static inline vector<RawScaleOutputHandle>& raw_scale_output_handles() {
    static vector<RawScaleOutputHandle> handles;
    return handles;
}

static inline void close_raw_scale_output_handles() {
    vector<RawScaleOutputHandle>& handles = raw_scale_output_handles();
    for (size_t i = 0; i < handles.size(); ++i) {
        if (handles[i].fp) {
            fclose(handles[i].fp);
            handles[i].fp = NULL;
        }
    }
}

static inline FILE* open_raw_scale_output_once(const string& path, bool binary) {
    vector<RawScaleOutputHandle>& handles = raw_scale_output_handles();
    for (size_t i = 0; i < handles.size(); ++i) {
        if (handles[i].path == path) {
            return handles[i].fp;
        }
    }

    static bool atexit_registered = false;
    if (!atexit_registered) {
        atexit(close_raw_scale_output_handles);
        atexit_registered = true;
    }

    FILE* fp = fopen(path.c_str(), binary ? "wb" : "w");
    if (fp) {
        setvbuf(fp, NULL, _IOFBF, 1 << 20);
        RawScaleOutputHandle handle;
        handle.path = path;
        handle.fp = fp;
        handles.push_back(handle);
    }
    return fp;
}

static inline void write_scale_raw_order_file(
    FileIntArrayManager* fm,
    const int8_t* scale,
    int file_scale_count)
{
    if (!fmac_dump_output_enabled() || !fmac_dump_all_enabled() || !fm || !scale || file_scale_count <= 0) return;

    string text_path = scale_raw_order_path(fm->filename, ".txt.log");
    string rawhex_path = scale_raw_order_path(fm->filename_rawhex, "_rawhex.txt.log");
    string bin_path = scale_raw_order_path(fm->filename_double_bin, "_double.bin.log");

    FILE* fp_text = open_raw_scale_output_once(text_path, false);
    FILE* fp_rawhex = open_raw_scale_output_once(rawhex_path, false);
    FILE* fp_bin = open_raw_scale_output_once(bin_path, true);
    if (!fp_text || !fp_rawhex || !fp_bin) {
        perror("open raw scale output failed");
        return;
    }

    for (int i = 0; i < file_scale_count; ++i) {
        uint8_t raw = (uint8_t)scale[i];
        string bits = bitset<8>(raw).to_string();
        fprintf(fp_text, "%s\n", bits.c_str());
        fprintf(fp_rawhex, "%02x %s\n", raw, bits.c_str());
        fwrite(&scale[i], sizeof(int8_t), 1, fp_bin);
    }

}

static inline void write_scale_file(
    FileIntArrayManager* fm,
    const int8_t* scale,
    int file_scale_count,
    int debug_type = 9)
{
    if (!fmac_dump_output_enabled() || !fm || !scale || file_scale_count <= 0) return;
    write_scale_raw_order_file(fm, scale, file_scale_count);
    int8_t* scale_to_file = (int8_t*)malloc(file_scale_count * sizeof(int8_t));
    if (!scale_to_file) return;
    for (int i = 0; i < file_scale_count; ++i) {
        scale_to_file[file_scale_count - 1 - i] = scale[i];
    }
    fm->write_int8_array_to_file(scale_to_file, file_scale_count, 1, debug_type, 0);
    free(scale_to_file);
}

static inline void unpack_fp4_stream_to_bytes(const int8_t* logical_fp4, int packed_count, int8_t* packed_bytes) {
    for (int i = 0; i < packed_count; ++i) {
        uint8_t low = static_cast<uint8_t>(logical_fp4[2 * i]) & 0x0f;
        uint8_t high = static_cast<uint8_t>(logical_fp4[2 * i + 1]) & 0x0f;
        packed_bytes[i] = static_cast<int8_t>((high << 4) | low);
    }
}

static inline void recenter_e2m1_payload_for_replay(int8_t* logical_payload, int elem_count) {
    for (int i = 0; i < elem_count; ++i) {
        logical_payload[i] = static_cast<int8_t>((static_cast<uint8_t>(logical_payload[i]) & 0x0f) << 2);
    }
}

static inline void rebuild_scale_array_for_replay(
    const int8_t* file_scale,
    int file_scale_count,
    int scale_size,
    int8_t* scale_out)
{
    for (int i = 0; i < scale_size; ++i) {
        scale_out[i] = 0;
    }
    for (int i = 0; i < file_scale_count; ++i) {
        scale_out[i] = file_scale[file_scale_count - 1 - i];
    }
}

static inline void broadcast_single_scale_for_mixed_desc(
    const MmaDescriptor& desc,
    int8_t* scale,
    int file_scale_count,
    int scale_size)
{
    if (desc.category != MmaCategory::MixedFloat || !desc.uses_scale ||
        !scale || file_scale_count != 1 || scale_size <= 1) {
        return;
    }
    int8_t value = scale[0];
    for (int i = 1; i < scale_size; ++i) {
        scale[i] = value;
    }
}

static inline void print_replay_summary(
    int dtype,
    int vecnum,
    const char* tag,
    int32_t stored_result,
    int32_t my_result,
    int32_t nv_result)
{
    printf("Replay summary: dtype=%d vec=%d tag=%s stored_d=0x%08x my=0x%08x nv=0x%08x\n",
           dtype,
           vecnum,
           tag ? tag : "unknown",
           static_cast<uint32_t>(stored_result),
           static_cast<uint32_t>(my_result),
           static_cast<uint32_t>(nv_result));
}

static inline bool is_legacy_mxf4_style_desc(const MmaDescriptor& desc) {
    return desc.is_legacy && mma_dtype_uses_mxf4_topology(desc.id);
}

static inline int legacy_mxf4_scale_format(const MmaDescriptor& desc) {
    return desc.id == MMA_DTYPE_NVFP4 ? E4M3 : UE8M0;
}

static inline void fill_random_e2m1_payload_bytes(int8_t* payload, int logical_size) {
    for (int i = 0; i < logical_size; ++i) {
        int8_t low = static_cast<int8_t>(static_cast<uint8_t>(gen_random_fp(E2M1)));
        int8_t high = static_cast<int8_t>(static_cast<uint8_t>(gen_random_fp(E2M1)));
        payload[i] = (high << 4) | (low & 0x0f);
    }
}

static inline void fill_random_s4_payload_bytes(int8_t* payload, int logical_size) {
    for (int i = 0; i < logical_size; ++i) {
        uint8_t low = (uint8_t)(rand() & 0x0f);
        uint8_t high = (uint8_t)(rand() & 0x0f);
        payload[i] = (int8_t)((high << 4) | low);
    }
}

static inline void fill_legacy_mxf4_random_scales(
    const MmaDescriptor& desc,
    int8_t* a_scale,
    int8_t* b_scale,
    int scale_valid_num)
{
    int scale_format = legacy_mxf4_scale_format(desc);
    for (int i = 0; i < scale_valid_num; ++i) {
        a_scale[i] = (int8_t)gen_random_fp(scale_format);
        b_scale[i] = (int8_t)gen_random_fp(scale_format);
    }
}

static inline void fill_legacy_mxf4_known_case(
    const MmaDescriptor& desc,
    int logical_size,
    int scale_size,
    int scale_valid_num,
    int8_t* a_payload,
    int8_t* b_payload,
    int32_t* c,
    int8_t* a_scale,
    int8_t* b_scale)
{
    memset(a_payload, 0, (size_t)logical_size * sizeof(int8_t));
    memset(b_payload, 0, (size_t)logical_size * sizeof(int8_t));
    memset(a_scale, 0, (size_t)scale_size * sizeof(int8_t));
    memset(b_scale, 0, (size_t)scale_size * sizeof(int8_t));

    for (int i = 0; i < scale_valid_num; ++i) {
        a_scale[i] = 0x00;
        b_scale[i] = 0x00;
    }

    if (desc.id == MMA_DTYPE_NVFP4) {
        a_payload[0] = (int8_t)0xda;
        b_payload[0] = (int8_t)0x19;
        c[0] = (int32_t)0x447ca75a;
        a_scale[0] = (int8_t)0xb0;
        b_scale[0] = (int8_t)0x1c;
    } else if (desc.id == MMA_DTYPE_MXFP4) {
        a_payload[0] = (int8_t)0x92;
        b_payload[0] = (int8_t)0x0f;
        c[0] = (int32_t)0x59297454;
    } else { // MXINT4
        a_payload[0] = (int8_t)0x98; // low=-8, high=-7
        b_payload[0] = (int8_t)0x17; // low=7, high=1
        c[0] = 0;
        for (int i = 0; i < scale_valid_num; ++i) {
            a_scale[i] = 127;
            b_scale[i] = 127;
        }
    }
}

static inline void print_legacy_mxf4_generation_result(
    const MmaDescriptor& desc,
    int vecnum,
    int32_t my_result,
    int32_t nv_result,
    bool print_en)
{
    const char* legacy_prefix = "mxf4nvf4_";
    const char* display_name = desc.file_stem;
    if (strncmp(display_name, legacy_prefix, strlen(legacy_prefix)) == 0) {
        display_name += strlen(legacy_prefix);
    }
    if (my_result != nv_result) {
        record_mac_mismatch(desc.name, desc.id, vecnum, my_result, nv_result);
        printf("%s MAC results mismatch!\n", display_name);
        printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", my_result, my_result);
        printf("NVIDIA nvmac: d_nv[0] = 0x%08x (decimal: %d)\n", nv_result, nv_result);
    } else if (print_en) {
        printf("%s MAC results match: d[0] = 0x%08x\n",
               display_name,
               (uint32_t)my_result);
    }
}

static inline int gen_legacy_mxf4_descriptor_file(
    DataFile* df,
    const MmaDescriptor& desc,
    int bitsize,
    int is_rand,
    bool print_en,
    int vecnum)
{
    int size = bitsize / 8;
    int logical_size = use_topology_preserving_128b_adapter() ? (size / 2) : size;
    int scale_size = mxf4_scale_storage_size_for_bitsize(desc.id, bitsize);
    int scale_valid_num = valid_scale_count_for_desc_bitsize(desc, bitsize);
    int8_t* a = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_A, size);
    int8_t* b = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_B, size);
    int8_t* a_scale = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_A_SCALE, scale_size, true);
    int8_t* b_scale = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_B_SCALE, scale_size, true);
    int32_t* c = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_C, 1);
    int32_t* d = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_D, 1);
    int32_t* d_nv = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_D_NV, 1);
    int8_t* a_payload = use_topology_preserving_128b_adapter()
        ? data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_A_PAYLOAD, logical_size)
        : a;
    int8_t* b_payload = use_topology_preserving_128b_adapter()
        ? data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_B_PAYLOAD, logical_size)
        : b;
    if (!a || !b || !a_scale || !b_scale || !c || !d || !d_nv || !a_payload || !b_payload) {
        return -1;
    }

    if (is_rand == 1) {
        if (desc.id == MMA_DTYPE_MXINT4) {
            fill_random_s4_payload_bytes(a_payload, logical_size);
            fill_random_s4_payload_bytes(b_payload, logical_size);
        } else {
            fill_random_e2m1_payload_bytes(a_payload, logical_size);
            fill_random_e2m1_payload_bytes(b_payload, logical_size);
        }
        c[0] = gen_random_fp(FP32);
        fill_legacy_mxf4_random_scales(desc, a_scale, b_scale, scale_valid_num);
        normalize_mxf4_scale_array_for_topology(desc.id, a_scale, scale_size);
        normalize_mxf4_scale_array_for_topology(desc.id, b_scale, scale_size);
    } else if (is_rand != 0 && desc.id != MMA_DTYPE_MXINT4) {
        gen_exception_fp8(a, b, c, size, is_rand);
    } else {
        fill_legacy_mxf4_known_case(
            desc,
            logical_size,
            scale_size,
            scale_valid_num,
            a_payload,
            b_payload,
            c,
            a_scale,
            b_scale);
        normalize_mxf4_scale_array_for_topology(desc.id, a_scale, scale_size);
        normalize_mxf4_scale_array_for_topology(desc.id, b_scale, scale_size);
    }

    if (use_topology_preserving_128b_adapter()) {
        expand_payload_for_dtype(desc.id, a_payload, logical_size, a, size);
        expand_payload_for_dtype(desc.id, b_payload, logical_size, b, size);
    }

    d[0] = mymac_scale(a, a_scale, b, b_scale, c, size, desc.id);
    d_nv[0] = maybe_reference_mac_scale(a, a_scale, b, b_scale, c, size, desc.id, d[0]);

    bool packed_fp4_text = is_legacy_mxf4_style_desc(desc);
    if (desc.id == MMA_DTYPE_MXINT4) {
        // The payload is already packed as [S4 high | S4 low]; do not pack it again.
        df->a->write_int8_array_to_file(a_payload, logical_size, 1, 8, 0);
        df->b->write_int8_array_to_file(b_payload, logical_size, 1, 8, 0);
    } else {
        write_operand_file(df->a, desc.a_kind, a_payload, logical_size, packed_fp4_text);
        write_operand_file(df->b, desc.b_kind, b_payload, logical_size, packed_fp4_text);
    }
    df->c->write_int32_array_to_file(c, 1, 1, 0);
    df->d->write_int32_array_to_file(d, 1, 1, 0);
    write_scale_file(df->a_scale, a_scale, scale_valid_num, scale_file_debug_type_for_desc(desc));
    write_scale_file(df->b_scale, b_scale, scale_valid_num, scale_file_debug_type_for_desc(desc));

    if (d[0] != d_nv[0]) {
        dump_scale_mismatch_input(desc,
                                  vecnum,
                                  a_payload,
                                  logical_size,
                                  a,
                                  size,
                                  b_payload,
                                  logical_size,
                                  b,
                                  size,
                                  a_scale,
                                  b_scale,
                                  scale_valid_num,
                                  c[0],
                                  d[0],
                                  d_nv[0]);
    }
    print_legacy_mxf4_generation_result(desc, vecnum, d[0], d_nv[0], print_en);
    return 0;
}

static inline int replay_generic_descriptor_file(
    DataFile* df,
    const MmaDescriptor& desc,
    int bitsize,
    int vecnum,
    bool float_type,
    bool print_en)
{
    int k = physical_elem_count_for_desc_bitsize(desc, bitsize);
    int a_logical = logical_elem_count_for_desc_operand_bitsize(desc, desc.a_kind, bitsize);
    int b_logical = logical_elem_count_for_desc_operand_bitsize(desc, desc.b_kind, bitsize);
    bool effective_fp16 = float_type && desc.allow_fp16_output;
    bool integer_result = descriptor_integer_result(desc);

    size_t a_elem_bytes = (size_t)(mma_operand_storage_bits(desc.a_kind) / 8);
    size_t b_elem_bytes = (size_t)(mma_operand_storage_bits(desc.b_kind) / 8);
    void* a = data_file_workspace_reserve(df, DATA_WS_A, (size_t)k * a_elem_bytes);
    void* b = data_file_workspace_reserve(df, DATA_WS_B, (size_t)k * b_elem_bytes);
    void* a_payload = desc_operand_needs_128b_expansion_bitsize(desc, desc.a_kind, bitsize)
        ? data_file_workspace_reserve(df, DATA_WS_A_PAYLOAD, (size_t)a_logical * a_elem_bytes)
        : a;
    void* b_payload = desc_operand_needs_128b_expansion_bitsize(desc, desc.b_kind, bitsize)
        ? data_file_workspace_reserve(df, DATA_WS_B_PAYLOAD, (size_t)b_logical * b_elem_bytes)
        : b;
    if (!a || !b || !a_payload || !b_payload) {
        return -1;
    }

    int32_t c_file = 0;
    int32_t d_file = 0;
    bool ok =
        read_operand_vector_from_bin(df->a->filename_double_bin, vecnum, a_logical, desc.a_kind, a_payload) &&
        read_operand_vector_from_bin(df->b->filename_double_bin, vecnum, b_logical, desc.b_kind, b_payload);

    if (ok) {
        ok = read_int32_vector_from_bin(df->c->filename_double_bin, vecnum, 1, &c_file) &&
             read_int32_vector_from_bin(df->d->filename_double_bin, vecnum, 1, &d_file);
        if (ok && effective_fp16 && !integer_result) {
            c_file = (int32_t)(uint16_t)c_file;
            d_file = (int32_t)(uint16_t)d_file;
        }
    }

    if (!ok) {
        return -1;
    }

    if (desc_operand_needs_128b_expansion_bitsize(desc, desc.a_kind, bitsize)) {
        expand_payload_for_dtype(desc.id, (uint8_t*)a_payload, a_logical, (uint8_t*)a, k);
    }
    if (desc_operand_needs_128b_expansion_bitsize(desc, desc.b_kind, bitsize)) {
        expand_payload_for_dtype(desc.id, (uint8_t*)b_payload, b_logical, (uint8_t*)b, k);
    }

    int32_t my_result = mymac(a, b, &c_file, k, desc.id, effective_fp16);
    int32_t nv_result = maybe_reference_mac(a, b, &c_file, k, desc.id, effective_fp16, my_result);
    if (my_result != nv_result) {
        record_mac_mismatch(desc.name, desc.id, vecnum, my_result, nv_result);
    }
    if (print_en) {
        print_replay_summary(desc.id, vecnum, desc.name, d_file, my_result, nv_result);
    }

    return (my_result == nv_result && d_file == my_result) ? 0 : 1;
}

static inline int gen_generic_descriptor_file(
    DataFile* df,
    const MmaDescriptor& desc,
    int bitsize,
    int is_rand,
    bool float_type,
    bool print_en,
    int vecnum)
{
    int k = physical_elem_count_for_desc_bitsize(desc, bitsize);
    int a_logical = logical_elem_count_for_desc_operand_bitsize(desc, desc.a_kind, bitsize);
    int b_logical = logical_elem_count_for_desc_operand_bitsize(desc, desc.b_kind, bitsize);
    bool effective_fp16 = float_type && desc.allow_fp16_output;
    bool integer_result = descriptor_integer_result(desc);

    size_t a_elem_bytes = (size_t)(mma_operand_storage_bits(desc.a_kind) / 8);
    size_t b_elem_bytes = (size_t)(mma_operand_storage_bits(desc.b_kind) / 8);
    void* a = data_file_workspace_reserve(df, DATA_WS_A, (size_t)k * a_elem_bytes);
    void* b = data_file_workspace_reserve(df, DATA_WS_B, (size_t)k * b_elem_bytes);
    void* a_payload = desc_operand_needs_128b_expansion_bitsize(desc, desc.a_kind, bitsize)
        ? data_file_workspace_reserve(df, DATA_WS_A_PAYLOAD, (size_t)a_logical * a_elem_bytes)
        : a;
    void* b_payload = desc_operand_needs_128b_expansion_bitsize(desc, desc.b_kind, bitsize)
        ? data_file_workspace_reserve(df, DATA_WS_B_PAYLOAD, (size_t)b_logical * b_elem_bytes)
        : b;
    if (!a || !b || !a_payload || !b_payload) {
        return -1;
    }

    for (int i = 0; i < a_logical; ++i) {
        uint32_t a_raw = is_rand ? generate_random_operand_raw(desc.a_kind)
                                 : generate_sequential_operand_raw(desc.a_kind, i);
        switch (mma_operand_storage_bits(desc.a_kind)) {
            case 16: ((uint16_t*)a_payload)[i] = (uint16_t)a_raw; break;
            case 32: ((uint32_t*)a_payload)[i] = a_raw; break;
            default: ((uint8_t*)a_payload)[i] = (uint8_t)a_raw; break;
        }
    }
    for (int i = 0; i < b_logical; ++i) {
        uint32_t b_raw = is_rand ? generate_random_operand_raw(desc.b_kind)
                                 : generate_sequential_operand_raw(desc.b_kind, i);
        switch (mma_operand_storage_bits(desc.b_kind)) {
            case 16: ((uint16_t*)b_payload)[i] = (uint16_t)b_raw; break;
            case 32: ((uint32_t*)b_payload)[i] = b_raw; break;
            default: ((uint8_t*)b_payload)[i] = (uint8_t)b_raw; break;
        }
    }

    if (desc_operand_needs_128b_expansion_bitsize(desc, desc.a_kind, bitsize)) {
        expand_payload_for_dtype(desc.id, (uint8_t*)a_payload, a_logical, (uint8_t*)a, k);
    }
    if (desc_operand_needs_128b_expansion_bitsize(desc, desc.b_kind, bitsize)) {
        expand_payload_for_dtype(desc.id, (uint8_t*)b_payload, b_logical, (uint8_t*)b, k);
    }

    int32_t c_value = 0;
    if (integer_result) {
        c_value = is_rand ? generate_random_s32() : 0;
    } else if (!effective_fp16) {
        c_value = is_rand ? (int32_t)gen_random_fp(FP32) : 0;
    } else {
        c_value = is_rand ? (uint16_t)gen_random_fp(FP16) : 0x00003C00;
    }

    int32_t my_result = mymac(a, b, &c_value, k, desc.id, effective_fp16);
    int32_t nv_result = maybe_reference_mac(a, b, &c_value, k, desc.id, effective_fp16, my_result);

    bool packed_fp4_text = is_legacy_mxf4_style_desc(desc);
    write_operand_file(df->a, desc.a_kind, a_payload, a_logical, packed_fp4_text);
    write_operand_file(df->b, desc.b_kind, b_payload, b_logical, packed_fp4_text);
    write_accumulator_file(df->c, c_value, effective_fp16, integer_result);
    write_accumulator_file(df->d, my_result, effective_fp16, integer_result);

    if (my_result != nv_result) {
        record_mac_mismatch(desc.name, desc.id, vecnum, my_result, nv_result);
        dump_descriptor_mismatch_input(desc,
                                       vecnum,
                                       a_payload,
                                       a_logical,
                                       a,
                                       k,
                                       b_payload,
                                       b_logical,
                                       b,
                                       k,
                                       nullptr,
                                       nullptr,
                                       0,
                                       c_value,
                                       my_result,
                                       nv_result,
                                       effective_fp16);
        if (print_en) {
            printf("%s mismatch: my=0x%08x nv=0x%08x\n",
                   desc.name,
                   (uint32_t)my_result,
                   (uint32_t)nv_result);
        }
    } else if (print_en) {
        printf("%s result match: 0x%08x\n", desc.name, (uint32_t)my_result);
    }

    return 0;
}

static inline int replay_generic_scale_descriptor_file(
    DataFile* df,
    const MmaDescriptor& desc,
    int bitsize,
    int vecnum,
    bool print_en)
{
    int size = replay_physical_elem_count_for_desc(desc, bitsize);
    int a_logical = replay_logical_elem_count_for_desc_operand(desc, desc.a_kind, bitsize);
    int b_logical = replay_logical_elem_count_for_desc_operand(desc, desc.b_kind, bitsize);
    int scale_size = is_legacy_mxf4_style_desc(desc)
        ? mxf4_scale_storage_size_for_bitsize(desc.id, bitsize)
        : scale_storage_size_for_desc(desc);
    int scale_file_count = valid_scale_count_for_desc_bitsize(desc, bitsize);

    size_t a_bytes = (size_t)a_logical * (size_t)(mma_operand_storage_bits(desc.a_kind) / 8);
    size_t b_bytes = (size_t)b_logical * (size_t)(mma_operand_storage_bits(desc.b_kind) / 8);
    int8_t* a_scale_file = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_A_SCALE_FILE, scale_file_count);
    int8_t* b_scale_file = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_B_SCALE_FILE, scale_file_count);
    int8_t* a_scale = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_A_SCALE, scale_size);
    int8_t* b_scale = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_B_SCALE, scale_size);
    void* a = data_file_workspace_reserve(df, DATA_WS_A, (size_t)size * (size_t)(mma_operand_storage_bits(desc.a_kind) / 8));
    void* b = data_file_workspace_reserve(df, DATA_WS_B, (size_t)size * (size_t)(mma_operand_storage_bits(desc.b_kind) / 8));
    void* a_payload = desc_operand_needs_128b_expansion(desc, desc.a_kind) ? data_file_workspace_reserve(df, DATA_WS_A_PAYLOAD, a_bytes) : a;
    void* b_payload = desc_operand_needs_128b_expansion(desc, desc.b_kind) ? data_file_workspace_reserve(df, DATA_WS_B_PAYLOAD, b_bytes) : b;
    int32_t c_file = 0;
    int32_t d_file = 0;

    bool ok = a && b && a_payload && b_payload && a_scale_file && b_scale_file && a_scale && b_scale &&
              (desc.id == MMA_DTYPE_MXINT4
                   ? read_int8_vector_from_bin(df->a->filename_double_bin, vecnum, a_logical, (int8_t*)a_payload)
                   : read_operand_vector_from_bin(df->a->filename_double_bin, vecnum, a_logical, desc.a_kind, a_payload)) &&
              (desc.id == MMA_DTYPE_MXINT4
                   ? read_int8_vector_from_bin(df->b->filename_double_bin, vecnum, b_logical, (int8_t*)b_payload)
                   : read_operand_vector_from_bin(df->b->filename_double_bin, vecnum, b_logical, desc.b_kind, b_payload)) &&
              read_int8_vector_from_bin(df->a_scale->filename_double_bin, vecnum, scale_file_count, a_scale_file) &&
              read_int8_vector_from_bin(df->b_scale->filename_double_bin, vecnum, scale_file_count, b_scale_file) &&
              read_int32_vector_from_bin(df->c->filename_double_bin, vecnum, 1, &c_file) &&
              read_int32_vector_from_bin(df->d->filename_double_bin, vecnum, 1, &d_file);

    if (!ok) {
        return -1;
    }

    rebuild_scale_array_for_replay(a_scale_file, scale_file_count, scale_size, a_scale);
    rebuild_scale_array_for_replay(b_scale_file, scale_file_count, scale_size, b_scale);
    broadcast_single_scale_for_mixed_desc(desc, a_scale, scale_file_count, scale_size);
    broadcast_single_scale_for_mixed_desc(desc, b_scale, scale_file_count, scale_size);
    normalize_mxf4_scale_array_for_topology(desc.id, a_scale, scale_size);
    normalize_mxf4_scale_array_for_topology(desc.id, b_scale, scale_size);

    if (desc_operand_needs_128b_expansion(desc, desc.a_kind)) {
        expand_payload_for_dtype(desc.id, (uint8_t*)a_payload, a_logical, (uint8_t*)a, size);
    }
    if (desc_operand_needs_128b_expansion(desc, desc.b_kind)) {
        expand_payload_for_dtype(desc.id, (uint8_t*)b_payload, b_logical, (uint8_t*)b, size);
    }

    int32_t my_result = mymac_scale(a, a_scale, b, b_scale, &c_file, size, desc.id);
    int32_t nv_result = maybe_reference_mac_scale(a, a_scale, b, b_scale, &c_file, size, desc.id, my_result);
    if (my_result != nv_result) {
        record_mac_mismatch(desc.name, desc.id, vecnum, my_result, nv_result);
    }
    if (print_en) {
        print_replay_summary(desc.id, vecnum, desc.name, d_file, my_result, nv_result);
    }

    return (my_result == nv_result && d_file == my_result) ? 0 : 1;
}

static inline int gen_generic_scale_descriptor_file(
    DataFile* df,
    const MmaDescriptor& desc,
    int bitsize,
    int is_rand,
    bool print_en,
    int vecnum)
{
    int size = physical_elem_count_for_desc_bitsize(desc, bitsize);
    int a_logical = logical_elem_count_for_desc_operand_bitsize(desc, desc.a_kind, bitsize);
    int b_logical = logical_elem_count_for_desc_operand_bitsize(desc, desc.b_kind, bitsize);
    int scale_size = scale_storage_size_for_desc(desc);
    int scale_valid_num = valid_scale_count_for_desc_bitsize(desc, bitsize);

    int8_t* a_scale = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_A_SCALE, scale_size, true);
    int8_t* b_scale = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_B_SCALE, scale_size, true);
    void* a = data_file_workspace_reserve(df, DATA_WS_A, (size_t)size * (size_t)(mma_operand_storage_bits(desc.a_kind) / 8));
    void* b = data_file_workspace_reserve(df, DATA_WS_B, (size_t)size * (size_t)(mma_operand_storage_bits(desc.b_kind) / 8));
    void* a_payload = desc_operand_needs_128b_expansion_bitsize(desc, desc.a_kind, bitsize)
        ? data_file_workspace_reserve(df, DATA_WS_A_PAYLOAD, (size_t)a_logical * (size_t)(mma_operand_storage_bits(desc.a_kind) / 8))
        : a;
    void* b_payload = desc_operand_needs_128b_expansion_bitsize(desc, desc.b_kind, bitsize)
        ? data_file_workspace_reserve(df, DATA_WS_B_PAYLOAD, (size_t)b_logical * (size_t)(mma_operand_storage_bits(desc.b_kind) / 8))
        : b;
    if (!a || !b || !a_payload || !b_payload || !a_scale || !b_scale) {
        return -1;
    }

    for (int i = 0; i < a_logical; ++i) {
        uint32_t a_raw = is_rand ? generate_random_operand_raw(desc.a_kind)
                                 : generate_sequential_operand_raw(desc.a_kind, i);
        ((uint8_t*)a_payload)[i] = (uint8_t)a_raw;
    }
    for (int i = 0; i < b_logical; ++i) {
        uint32_t b_raw = is_rand ? generate_random_operand_raw(desc.b_kind)
                                 : generate_sequential_operand_raw(desc.b_kind, i);
        ((uint8_t*)b_payload)[i] = (uint8_t)b_raw;
    }

    for (int i = 0; i < scale_valid_num; ++i) {
        if (is_rand) {
            a_scale[i] = (int8_t)gen_random_fp(UE8M0);
            b_scale[i] = (int8_t)gen_random_fp(UE8M0);
        } else {
            a_scale[i] = 127;
            b_scale[i] = 127;
        }
    }
    broadcast_single_scale_for_mixed_desc(desc, a_scale, scale_valid_num, scale_size);
    broadcast_single_scale_for_mixed_desc(desc, b_scale, scale_valid_num, scale_size);
    normalize_mxf4_scale_array_for_topology(desc.id, a_scale, scale_size);
    normalize_mxf4_scale_array_for_topology(desc.id, b_scale, scale_size);

    if (desc_operand_needs_128b_expansion_bitsize(desc, desc.a_kind, bitsize)) {
        expand_payload_for_dtype(desc.id, (uint8_t*)a_payload, a_logical, (uint8_t*)a, size);
    }
    if (desc_operand_needs_128b_expansion_bitsize(desc, desc.b_kind, bitsize)) {
        expand_payload_for_dtype(desc.id, (uint8_t*)b_payload, b_logical, (uint8_t*)b, size);
    }

    int32_t c_value = is_rand ? (int32_t)gen_random_fp(FP32) : 0;
    int32_t my_result = mymac_scale(a, a_scale, b, b_scale, &c_value, size, desc.id);
    int32_t nv_result = maybe_reference_mac_scale(a, a_scale, b, b_scale, &c_value, size, desc.id, my_result);

    bool packed_fp4_text = is_legacy_mxf4_style_desc(desc);
    write_operand_file(df->a, desc.a_kind, a_payload, a_logical, packed_fp4_text);
    write_operand_file(df->b, desc.b_kind, b_payload, b_logical, packed_fp4_text);
    write_accumulator_file(df->c, c_value, false, false);
    write_accumulator_file(df->d, my_result, false, false);
    write_scale_file(df->a_scale, a_scale, scale_valid_num, scale_file_debug_type_for_desc(desc));
    write_scale_file(df->b_scale, b_scale, scale_valid_num, scale_file_debug_type_for_desc(desc));

    if (my_result != nv_result) {
        record_mac_mismatch(desc.name, desc.id, vecnum, my_result, nv_result);
        dump_scale_mismatch_input(desc,
                                  vecnum,
                                  a_payload,
                                  a_logical,
                                  a,
                                  size,
                                  b_payload,
                                  b_logical,
                                  b,
                                  size,
                                  a_scale,
                                  b_scale,
                                  scale_valid_num,
                                  c_value,
                                  my_result,
                                  nv_result);
        if (print_en) {
            printf("%s mismatch: my=0x%08x nv=0x%08x\n",
                   desc.name,
                   (uint32_t)my_result,
                   (uint32_t)nv_result);
        }
    } else if (print_en) {
        printf("%s result match: 0x%08x\n", desc.name, (uint32_t)my_result);
    }

    return 0;
}

int replay_data_file(DataFile* df, int bitsize, int dtype, int vecnum, bool float_type = 0, bool print_en = 1) {
    if (!df || !df->a || !df->b || !df->c || !df->d) return -1;
    clear_last_mismatch_record();

    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (!desc) {
        fprintf(stderr, "Replay is not implemented for dtype=%d yet.\n", dtype);
        return -1;
    }
    if (use_replay_descriptor_path(desc)) {
        return replay_generic_descriptor_file(df, *desc, bitsize, vecnum, float_type, print_en);
    }
    if (use_replay_scale_descriptor_path(desc)) {
        return replay_generic_scale_descriptor_file(df, *desc, bitsize, vecnum, print_en);
    }
    return -1;
}

int gen_data_file(DataFile* df, int bitsize, int dtype, int is_rand, bool float_type = 0, bool print_en = 0, int vecnum = 0) {
    if (!df || !df->a || !df->b || !df->c || !df->d) return -1;
    clear_last_mismatch_record();

    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (desc && is_legacy_mxf4_style_desc(*desc)) {
        return gen_legacy_mxf4_descriptor_file(df, *desc, bitsize, is_rand, print_en, vecnum);
    }
    if (use_generation_descriptor_path(desc)) {
        return gen_generic_descriptor_file(df, *desc, bitsize, is_rand, float_type, print_en, vecnum);
    }
    if (use_generation_scale_descriptor_path(desc)) {
        return gen_generic_scale_descriptor_file(df, *desc, bitsize, is_rand, print_en, vecnum);
    }

    if ((dtype == 0) || (dtype == 4) || (dtype == 5) || (dtype == 6)) { // INT8
        int size = bitsize / 8; // INT8每个元素1字节  (256bit输入)
        bool use_128b_adapter = use_topology_preserving_128b_adapter();
        int logical_size = use_128b_adapter ? (size / 2) : size; //(可采用256或128bit输入，但在128bit输入模式下仍会使用零拓展到256bit)
        int8_t* a = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_A, size);
        int8_t* b = data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_B, size);
        int32_t* c = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_C, 1);
        int32_t* d = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_D, 1);
        int8_t* a_payload = use_128b_adapter ? data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_A_PAYLOAD, logical_size) : a;
        int8_t* b_payload = use_128b_adapter ? data_file_workspace_reserve_typed<int8_t>(df, DATA_WS_B_PAYLOAD, logical_size) : b;
        if (!a || !b || !c || !d || !a_payload || !b_payload) {
            return -1;
        }

        if(is_rand){
            for (int i = 0; i < logical_size; ++i) {
                // 随机生成-128到127的值
                a_payload[i] = rand() % 256 - 128; 
                b_payload[i] = rand() % 256 - 128;
            }
            c[0] = generate_random_s32();
        }
        else {
            for (int i = 0; i < logical_size; ++i){
                a_payload[i] = i;
                b_payload[i] = i;
            }
            c[0] = 0; // 初始值为0
        }
        if (use_128b_adapter) {
            expand_payload_preserve_group_topology(a_payload, logical_size, a, size, topology_group_count_for_dtype(dtype));
            expand_payload_preserve_group_topology(b_payload, logical_size, b, size, topology_group_count_for_dtype(dtype));
        }
        // int32_t temp_zero = 0;
        d[0] = mymac(a, b, c, size, dtype);

        df->a->write_int8_array_to_file(a_payload, logical_size, 1, 8, 0);
        df->b->write_int8_array_to_file(b_payload, logical_size, 1, 8, 0);
        df->c->write_int32_array_to_file(c, 1, 1, 8);
        df->d->write_int32_array_to_file(d, 1, 1, 8);

    }
    else if (dtype == 1) {
        int size = bitsize / 16; // INT16每个元素2字节，INT32每个元素4字节
        bool use_128b_adapter = use_topology_preserving_128b_adapter();
        int logical_size = use_128b_adapter ? (size / 2) : size;
        int16_t* a = data_file_workspace_reserve_typed<int16_t>(df, DATA_WS_A, size);
        int16_t* b = data_file_workspace_reserve_typed<int16_t>(df, DATA_WS_B, size);
        int32_t* c = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_C, 1);
        int32_t* d = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_D, 1);
        int32_t* d_nv = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_D_NV, 1);
        int16_t* a_payload = use_128b_adapter ? data_file_workspace_reserve_typed<int16_t>(df, DATA_WS_A_PAYLOAD, logical_size) : a;
        int16_t* b_payload = use_128b_adapter ? data_file_workspace_reserve_typed<int16_t>(df, DATA_WS_B_PAYLOAD, logical_size) : b;
        if (!a || !b || !c || !d || !d_nv || !a_payload || !b_payload) {
            return -1;
        }

        if(is_rand == 1){
            for (int i = 0; i < logical_size; ++i) {
                a_payload[i] = (int16_t)gen_random_fp(1); // 将__fp16转换为int16_t
                b_payload[i] = (int16_t)gen_random_fp(1); // 将__fp16转换为int16_t
            }
            if(!float_type){
                c[0] = gen_random_fp(0); // 将float转换为int32_t
            }
            else {
                c[0] = (uint16_t)gen_random_fp(1); // 先去掉符号扩展,只保留低16位
            }

        } else if(is_rand != 0){
            gen_exception_fp16(a, b, c, size, is_rand);
        } else{
            uint16_t val;

            // val = 0xc8a8;
            // memcpy(&a[0], &val, sizeof(uint16_t));
            // val = 0x1bec;
            // memcpy(&a[0], &val, sizeof(uint16_t));
            val = 0x02b1;
            memcpy(&a_payload[0], &val, sizeof(uint16_t));

            // val = 0x94d6;
            // memcpy(&b[0], &val, sizeof(uint16_t));
            // val = 0x4fdc;
            // memcpy(&b[0], &val, sizeof(uint16_t));
            val = 0x22ae;
            memcpy(&b_payload[0], &val, sizeof(uint16_t));
            if(!float_type){
                // c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            }
            else {
                uint16_t val16 = 0x8009;
                c[0] = (int32_t)val16;
            }
        }
        if (use_128b_adapter) {
            expand_payload_preserve_group_topology(a_payload, logical_size, a, size, topology_group_count_for_dtype(dtype));
            expand_payload_preserve_group_topology(b_payload, logical_size, b, size, topology_group_count_for_dtype(dtype));
        }
        d[0] = mymac(a, b, c, size, dtype, float_type);

        d_nv[0] = maybe_reference_mac(a, b, c, size, dtype, float_type, d[0]);

        if (d[0] != d_nv[0]) {
            record_mac_mismatch("FP16", dtype, vecnum, d[0], d_nv[0]);
            if (desc) {
                dump_descriptor_mismatch_input(*desc,
                                               vecnum,
                                               a_payload,
                                               logical_size,
                                               a,
                                               size,
                                               b_payload,
                                               logical_size,
                                               b,
                                               size,
                                               nullptr,
                                               nullptr,
                                               0,
                                               c[0],
                                               d[0],
                                               d_nv[0],
                                               float_type);
            }
            printf("FP16 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvmac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
        }
        df->a->write_int16_array_to_file(a_payload, logical_size, 1, 1);
        df->b->write_int16_array_to_file(b_payload, logical_size, 1, 1);
        if(!float_type){
            df->c->write_int32_array_to_file(c, 1, 1, 0);
            df->d->write_int32_array_to_file(d, 1, 1, 0);
        }
        else {
            int32_t c32 = (int32_t)(uint16_t)c[0];
            int32_t d32 = (int32_t)(uint16_t)d[0];

            df->c->write_int32_array_to_file(&c32, 1, 1, 1);
            df->d->write_int32_array_to_file(&d32, 1, 1, 1);
        }
        if (d[0] == d_nv[0] && print_en) {
            // Optional: log success for debugging
            printf("FP16 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

    }
    else if (dtype == 2) {
        int size = bitsize / 16; // INT16每个元素2字节，INT32每个元素4字节
        bool use_128b_adapter = use_topology_preserving_128b_adapter();
        int logical_size = use_128b_adapter ? (size / 2) : size;
        int16_t* a = data_file_workspace_reserve_typed<int16_t>(df, DATA_WS_A, size);
        int16_t* b = data_file_workspace_reserve_typed<int16_t>(df, DATA_WS_B, size);
        int32_t* c = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_C, 1);
        int32_t* d = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_D, 1);
        int32_t* d_nv = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_D_NV, 1);
        int16_t* a_payload = use_128b_adapter ? data_file_workspace_reserve_typed<int16_t>(df, DATA_WS_A_PAYLOAD, logical_size) : a;
        int16_t* b_payload = use_128b_adapter ? data_file_workspace_reserve_typed<int16_t>(df, DATA_WS_B_PAYLOAD, logical_size) : b;
        if (!a || !b || !c || !d || !d_nv || !a_payload || !b_payload) {
            return -1;
        }

        if(is_rand == 1){
            for (int i = 0; i < logical_size; ++i) {
                a_payload[i] = (uint16_t)gen_random_fp(2); // 将__fp16转换为int16_t
                b_payload[i] = (uint16_t)gen_random_fp(2); // 将__fp16转换为int16_t
            }
            c[0] = gen_random_fp(0); // 将float转换为int32_t
        } else if(is_rand != 0){
            gen_exception_bf16(a, b, c, size, is_rand);
        } else{
            for (int i = 0; i < logical_size; ++i){
                float temp_a_f = i; // 将i转换为__fp16
                float temp_b_f = i; // 将i转换为__fp16
                int32_t temp_a_32 = *((int32_t*)&temp_a_f) >> 16; // 将float转换为int32_t
                int32_t temp_b_32 = *((int32_t*)&temp_b_f) >> 16; // 将float转换为int32_t
                int16_t temp_a = (int16_t)(temp_a_32 & 0xFFFF); // 取低16位
                int16_t temp_b = (int16_t)(temp_b_32 & 0xFFFF); // 取低16位
                // 将int32_t转换为int16
                a_payload[i] = temp_a;
                b_payload[i] = temp_b;
            }
            c[0] = 0; // 初始值为0
        }
        if (use_128b_adapter) {
            expand_payload_preserve_group_topology(a_payload, logical_size, a, size, topology_group_count_for_dtype(dtype));
            expand_payload_preserve_group_topology(b_payload, logical_size, b, size, topology_group_count_for_dtype(dtype));
        }
        d[0] = mymac(a, b, c, size, dtype, float_type);

        d_nv[0] = maybe_reference_mac(a, b, c, size, dtype, float_type, d[0]);

        if (d[0] != d_nv[0]) {
            record_mac_mismatch("BF16", dtype, vecnum, d[0], d_nv[0]);
            if (desc) {
                dump_descriptor_mismatch_input(*desc,
                                               vecnum,
                                               a_payload,
                                               logical_size,
                                               a,
                                               size,
                                               b_payload,
                                               logical_size,
                                               b,
                                               size,
                                               nullptr,
                                               nullptr,
                                               0,
                                               c[0],
                                               d[0],
                                               d_nv[0],
                                               float_type);
            }
            printf("BF16 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvmac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
        }
        df->a->write_int16_array_to_file(a_payload, logical_size, 1, 2);
        df->b->write_int16_array_to_file(b_payload, logical_size, 1, 2);
        df->c->write_int32_array_to_file(c, 1, 1, 0);
        df->d->write_int32_array_to_file(d, 1, 1, 0);
        if (d[0] == d_nv[0] && print_en) {
            // Optional: log success for debugging
            printf("BF16 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

    }
    else if (dtype == 3) {
        int size = bitsize / 32; // INT16每个元素2字节，INT32每个元素4字节
        bool use_128b_adapter = use_topology_preserving_128b_adapter();
        int logical_size = use_128b_adapter ? (size / 2) : size;
        int32_t* a = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_A, size);
        int32_t* b = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_B, size);
        int32_t* c = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_C, 1);
        int32_t* d = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_D, 1);
        int32_t* d_nv = data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_D_NV, 1);
        int32_t* a_payload = use_128b_adapter ? data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_A_PAYLOAD, logical_size) : a;
        int32_t* b_payload = use_128b_adapter ? data_file_workspace_reserve_typed<int32_t>(df, DATA_WS_B_PAYLOAD, logical_size) : b;
        if (!a || !b || !c || !d || !d_nv || !a_payload || !b_payload) {
            return -1;
        }

        if(is_rand == 1){
            for (int i = 0; i < logical_size; ++i) {
                a_payload[i] = gen_random_fp(0) & 0xFFFFE000; // 将__fp16转换为int16_t
                b_payload[i] = gen_random_fp(0) & 0xFFFFE000; // 将__fp16转换为int16_t
                // a[i] = 0x7F802000;
                // b[i] = 0x00100000;
            }
            c[0] = gen_random_fp(0); // 将float转换为int32_t
        } else if(is_rand != 0){
            gen_exception_tf32(a, b, c, size, is_rand);
        } else{
            for (int i = 0; i < logical_size; ++i){
                float temp_a = (float)i; // 将i转换为__fp16
                float temp_b = (float)i; // 将i转换为__fp16
                a_payload[i] = *((int32_t*)&temp_a); // 将__fp16转换为int16_t
                b_payload[i] = *((int32_t*)&temp_b); //
            }
            c[0] = 0; // 初始值为0
        }
        if (use_128b_adapter) {
            expand_payload_preserve_group_topology(a_payload, logical_size, a, size, topology_group_count_for_dtype(dtype));
            expand_payload_preserve_group_topology(b_payload, logical_size, b, size, topology_group_count_for_dtype(dtype));
        }
        d[0] = mymac(a, b, c, size, dtype, float_type);
        
        d_nv[0] = maybe_reference_mac(a, b, c, size, dtype, float_type, d[0]);

        if (d[0] != d_nv[0]) {
            record_mac_mismatch("TF32", dtype, vecnum, d[0], d_nv[0]);
            if (desc) {
                dump_descriptor_mismatch_input(*desc,
                                               vecnum,
                                               a_payload,
                                               logical_size,
                                               a,
                                               size,
                                               b_payload,
                                               logical_size,
                                               b,
                                               size,
                                               nullptr,
                                               nullptr,
                                               0,
                                               c[0],
                                               d[0],
                                               d_nv[0],
                                               float_type);
            }
            printf("TF32 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvmac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
        }
        df->a->write_int32_array_to_file(a_payload, logical_size, 1, 0);
        df->b->write_int32_array_to_file(b_payload, logical_size, 1, 0);
        df->c->write_int32_array_to_file(c, 1, 1, 0);
        df->d->write_int32_array_to_file(d, 1, 1, 0);
        if (d[0] == d_nv[0] && print_en) {
            // Optional: log success for debugging
            printf("TF32 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

        // df->a->write_int32_array_to_file(a, size, 1);
        // df->b->write_int32_array_to_file(b, size, 1);
        // df->c->write_int32_array_to_file(c, 1, 1);
        // df->d->write_int32_array_to_file(d, 1, 1);

    }
    return 0;
}
// int check_data_file(DataFile* df, int bitsize, int dtype, int gen_diff_file){
//     if (!df || !df->a || !df->b || !df->c || !df->d) return 1;

    
//     if(dtype == 0){
//         int size = bitsize / 8;
//         int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
//         int8_t* b = (int8_t*)malloc(size * sizeof(int8_t));
//         int32_t* c = (int32_t*)malloc(sizeof(int32_t));
//         int32_t* d = (int32_t*)malloc(sizeof(int32_t));

//         df->a->read_int8_array_from_file(a, size, 1);
//         df->b->read_int8_array_from_file(b, size, 1);
//         df->c->read_int32_array_from_file(c, 1, 1);
//         df->d->read_int32_array_from_file(d, 1, 1);

//         int32_t check_result = c[0];
//         int32_t undercheck_result = d[0];

//         for (int i = 0; i < size; ++i) {
//             check_result += (int32_t)a[i] * (int32_t)b[i];
//         }

//         free(a);
//         free(b);
//         free(c);
//         free(d);

//         if(check_result == undercheck_result){
//             return 0; // 成功
//         }
//         printf("Check       Result: %d\n", check_result);
//         printf("Under Check Result: %d\n", undercheck_result);
//         printf("\n");
//     }
//     else if(dtype == 1){
//         int size = bitsize / 16;
//         int16_t* a = (int16_t*)malloc(size * sizeof(int16_t));
//         int16_t* b = (int16_t*)malloc(size * sizeof(int16_t));
//         int32_t* c = (int32_t*)malloc(sizeof(int32_t));
//         int32_t* d = (int32_t*)malloc(sizeof(int32_t));

//         df->a->read_int16_array_from_file(a, size, 1);
//         df->b->read_int16_array_from_file(b, size, 1);
//         df->c->read_int32_array_from_file(c, 1, 1);
//         df->d->read_int32_array_from_file(d, 1, 1);

//         float check_result = 0;
//         float file_result = *((float*)&d[0]);

//         float add_float_max = my_float_abs(*((float*)&c[0]));

//         for (int i = 0; i < size; ++i) {
//             __fp16 temp_a = *((__fp16*)&a[i]);
//             __fp16 temp_b = *((__fp16*)&b[i]);
//             // check_result += (float)(temp_a * temp_b);
//             float temp_mul = ((float)temp_a) * ((float)temp_b); // 计算FP16乘积
//             check_result += temp_mul; // 将FP16乘积加到总和中
//             if (my_float_abs(temp_mul) > add_float_max) {
//                 add_float_max = my_float_abs(temp_mul);
//             }
//         }

//         check_result += *((float*)&c[0]); // 将int32_t转换为float

//         int32_t* d_new = (int32_t*)malloc(sizeof(int32_t));
//         d_new[0] = mymac(a, b, c, size, dtype);

//         float undercheck_result = *((float*)&d_new[0]);
//         float new_diff_percent = (check_result - undercheck_result) / add_float_max;

//         int32_t exception = get_exceptioncode(check_result);
//         int32_t undercheck_exception = get_exceptioncode(undercheck_result);

//         // 全0特殊情况，diff_percent为0，只检查输出结果是不是0
//         if(add_float_max == 0){
//             if(undercheck_result != 0){
//                 printf("Check       Result: %f\n", check_result);
//                 printf("Under Check Result: %f\n", undercheck_result);

//                 free(a);
//                 free(b);
//                 free(c);
//                 free(d);
//                 free(d_new);
//                 return 2;
//             }

//             free(a);
//             free(b);
//             free(c);
//             free(d);
//             free(d_new);
//             return 0;
//         }

//         // 检查异常码是否一致
//         if(exception != 0){
//             if(exception != undercheck_exception){
//                 printf("Check       Result: %f\n", check_result);
//                 printf("Under Check Result: %f\n", undercheck_result);
//                 printf("Exception code mismatch: %d vs %d\n", exception, undercheck_exception);

//                 free(a);
//                 free(b);
//                 free(c);
//                 free(d);
//                 free(d_new);
//                 return 2;
//             }

//             free(a);
//             free(b);
//             free(c);
//             free(d);
//             free(d_new);
//             return 3;
//         }

//         if((new_diff_percent > NEW_DIFF_PERCENT || new_diff_percent < -NEW_DIFF_PERCENT)&& gen_diff_file == 1){
//             FILE* fp = fopen("diff_fp16.txt", "w");
//             if(fp){
//                 fprintf(fp, "a\n");
//                 for(int i = 0; i < 2 * size; i++){
//                     fprintf(fp, "%.8f\n", *(float*)&a[i]);
//                 }
//                 fprintf(fp, "\n");
//                 fprintf(fp, "b\n");
//                 for(int i = 0; i < 2 * size; i++){
//                     fprintf(fp, "%.8f\n", *(float*)&b[i]);
//                 }
//                 fprintf(fp, "\n");
//                 fprintf(fp, "c\n");
//                 fprintf(fp, "%.8f\n", *(float*)&c[0]);

//                 fprintf(fp, "\n");
//                 fprintf(fp, "undercheck\n");
//                 fprintf(fp, "%.8f\n", *(float*)&d[0]);

//                 fprintf(fp, "\n");
//                 fprintf(fp, "golden\n");
//                 fprintf(fp, "%.8f\n", check_result);
//             }
//             fclose(fp);
//         }


//         if(new_diff_percent > NEW_DIFF_PERCENT || new_diff_percent < -NEW_DIFF_PERCENT){
//             printf("Check       Result: %a\n", check_result);
//             printf("Under Check Result: %a\n", undercheck_result);
//             printf("new_diff_percent: %f\n", new_diff_percent);

//             free(a);
//             free(b);
//             free(c);
//             free(d);
//             free(d_new);
//             return 2;
//         }
//     }
//     else if(dtype == 2){
//         int size = bitsize / 16;
//         int16_t* a = (int16_t*)malloc(size * sizeof(int16_t));
//         int16_t* b = (int16_t*)malloc(size * sizeof(int16_t));
//         int32_t* c = (int32_t*)malloc(sizeof(int32_t));
//         int32_t* d = (int32_t*)malloc(sizeof(int32_t));

//         df->a->read_int16_array_from_file(a, size, 1);
//         df->b->read_int16_array_from_file(b, size, 1);
//         df->c->read_int32_array_from_file(c, 1, 1);
//         df->d->read_int32_array_from_file(d, 1, 1);

//         double check_result_double = 0;
//         float file_result = *((float*)&d[0]);

//         double add_float_max = (double)my_float_abs(*((float*)&c[0]));

//         for (int i = 0; i < size; ++i) {
//             int32_t temp_int_a = (int32_t)a[i] << 16; // 将int16_t转换为int32_t
//             int32_t temp_int_b = (int32_t)b[i] << 16;
//             float temp_a = *((float*)&temp_int_a); // 将int32_t转换为float
//             float temp_b = *((float*)&temp_int_b);
//             double temp_mul = (double)temp_a * (double)temp_b; // 计算BF16乘积
//             check_result_double += temp_mul; // 将FP16乘积加到总和中
//             if (my_double_abs(temp_mul) > add_float_max) {
//                 add_float_max = my_double_abs(temp_mul);
//             }
//         }

//         check_result_double += (double)*((float*)&c[0]); // 将int32_t转换为float
//         float check_result = (float)check_result_double; // 将double转换为float

//         int32_t* d_new = (int32_t*)malloc(sizeof(int32_t));
//         d_new[0] = mymac(a, b, c, size, dtype);

//         float undercheck_result = *((float*)&d_new[0]);
//         float new_diff_percent = (check_result - undercheck_result) / add_float_max;

//         int32_t exception = get_exceptioncode(check_result);
//         int32_t undercheck_exception = get_exceptioncode(undercheck_result);

//         // 全0特殊情况，diff_percent为0，只检查输出结果是不是0
//         if(add_float_max == 0){
//             if(undercheck_result != 0){
//                 printf("Check       Result: %f\n", check_result);
//                 printf("Under Check Result: %f\n", undercheck_result);

//                 free(a);
//                 free(b);
//                 free(c);
//                 free(d);
//                 free(d_new);
//                 return 2;
//             }

//             free(a);
//             free(b);
//             free(c);
//             free(d);
//             free(d_new);
//             return 0;
//         }

//         // 检查异常码是否一致
//         if(exception != 0){
//             if(exception != undercheck_exception){
//                 printf("Check       Result: %f\n", check_result);
//                 printf("Under Check Result: %f\n", undercheck_result);
//                 printf("Exception code mismatch: %d vs %d\n", exception, undercheck_exception);

//                 free(a);
//                 free(b);
//                 free(c);
//                 free(d);
//                 free(d_new);
//                 return 2;
//             }

//             free(a);
//             free(b);
//             free(c);
//             free(d);
//             free(d_new);
//             return 3;
//         }

//         if((new_diff_percent > NEW_DIFF_PERCENT || new_diff_percent < -NEW_DIFF_PERCENT) && gen_diff_file == 1){
//             FILE* fp = fopen("diff_bf16.txt", "w");
//             if(fp){
//                 fprintf(fp, "a\n");
//                 for(int i = 0; i < 2 * size; i++){
//                     fprintf(fp, "%.8f\n", *(float*)&a[i]);
//                 }
//                 fprintf(fp, "\n");
//                 fprintf(fp, "b\n");
//                 for(int i = 0; i < 2 * size; i++){
//                     fprintf(fp, "%.8f\n", *(float*)&b[i]);
//                 }
//                 fprintf(fp, "\n");
//                 fprintf(fp, "c\n");
//                 fprintf(fp, "%.8f\n", *(float*)&c[0]);

//                 fprintf(fp, "\n");
//                 fprintf(fp, "undercheck\n");
//                 fprintf(fp, "%.8f\n", *(float*)&d[0]);

//                 fprintf(fp, "\n");
//                 fprintf(fp, "golden\n");
//                 fprintf(fp, "%.8f\n", check_result);
//             }
//             fclose(fp);
//         }

//         if(new_diff_percent > NEW_DIFF_PERCENT || new_diff_percent < -NEW_DIFF_PERCENT){
//             printf("Check       Result: %f\n", check_result);
//             printf("Under Check Result: %f\n", undercheck_result);
//             printf("new_diff_percent: %f\n", new_diff_percent);

//             free(a);
//             free(b);
//             free(c);
//             free(d);
//             free(d_new);
//             return 2;
//         }
//     }
//     else if(dtype == 3){
//         int size = bitsize / 32;
//         int32_t* a = (int32_t*)malloc(size * sizeof(int32_t));
//         int32_t* b = (int32_t*)malloc(size * sizeof(int32_t));
//         int32_t* c = (int32_t*)malloc(sizeof(int32_t));
//         int32_t* d = (int32_t*)malloc(sizeof(int32_t));

//         df->a->read_int32_array_from_file(a, size, 1);
//         df->b->read_int32_array_from_file(b, size, 1);
//         df->c->read_int32_array_from_file(c, 1, 1);
//         df->d->read_int32_array_from_file(d, 1, 1);


//         double check_result_double = 0;
//         float file_result = *((float*)&d[0]);

//         double add_float_max = (double)my_float_abs(*((float*)&c[0]));

//         for (int i = 0; i < size; ++i) {
//             int32_t temp_int_a = a[i] & 0xFFFFE000; // 将int16_t转换为int32_t
//             int32_t temp_int_b = b[i] & 0xFFFFE000;
//             float temp_a = *((float*)&temp_int_a); // 将int32_t转换为float
//             float temp_b = *((float*)&temp_int_b);
//             double temp_mul = (double)temp_a * (double)temp_b; // 计算BF16乘积
//             check_result_double += temp_mul; // 将FP16乘积加到总和中
//             if (my_double_abs(temp_mul) > add_float_max) {
//                 add_float_max = my_double_abs(temp_mul);
//             }
//         }

//         check_result_double += (double)*((float*)&c[0]); // 将int32_t转换为float
//         float check_result = (float)check_result_double; // 将double转换为float

//         int32_t* d_new = (int32_t*)malloc(sizeof(int32_t));
//         d_new[0] = mymac(a, b, c, size, dtype);

//         float undercheck_result = *((float*)&d_new[0]);
//         float new_diff_percent = (check_result - undercheck_result) / add_float_max;

//         int32_t exception = get_exceptioncode(check_result);
//         int32_t undercheck_exception = get_exceptioncode(undercheck_result);

//         // 全0特殊情况，diff_percent为0，只检查输出结果是不是0
//         if(add_float_max == 0){
//             if(undercheck_result != 0){
//                 printf("Check       Result: %f\n", check_result);
//                 printf("Under Check Result: %f\n", undercheck_result);

//                 free(a);
//                 free(b);
//                 free(c);
//                 free(d);
//                 free(d_new);
//                 return 2;
//             }

//             free(a);
//             free(b);
//             free(c);
//             free(d);
//             free(d_new);
//             return 0;
//         }

//         // 检查异常码是否一致
//         if(exception != 0){
//             if(exception != undercheck_exception){
//                 printf("Check       Result: %f\n", check_result);
//                 printf("Under Check Result: %f\n", undercheck_result);
//                 printf("Exception code mismatch: %d vs %d\n", exception, undercheck_exception);

//                 free(a);
//                 free(b);
//                 free(c);
//                 free(d);
//                 free(d_new);
//                 return 2;
//             }

//             free(a);
//             free(b);
//             free(c);
//             free(d);
//             free(d_new);
//             return 3;
//         }

//         if((new_diff_percent > NEW_DIFF_PERCENT || new_diff_percent < NEW_DIFF_PERCENT) && gen_diff_file == 1){
//             FILE* fp = fopen("diff_tf32.txt", "w");
//             if(fp){
//                 fprintf(fp, "a\n");
//                 for(int i = 0; i < 2 * size; i++){
//                     fprintf(fp, "%.8f\n", *(float*)&a[i]);
//                 }
//                 fprintf(fp, "\n");
//                 fprintf(fp, "b\n");
//                 for(int i = 0; i < 2 * size; i++){
//                     fprintf(fp, "%.8f\n", *(float*)&b[i]);
//                 }
//                 fprintf(fp, "\n");
//                 fprintf(fp, "c\n");
//                 fprintf(fp, "%.8f\n", *(float*)&c[0]);

//                 fprintf(fp, "\n");
//                 fprintf(fp, "undercheck\n");
//                 fprintf(fp, "%.8f\n", *(float*)&d[0]);

//                 fprintf(fp, "\n");
//                 fprintf(fp, "golden\n");
//                 fprintf(fp, "%.8f\n", check_result);
//             }
//             fclose(fp);
//         }

//         if(new_diff_percent > NEW_DIFF_PERCENT || new_diff_percent < -NEW_DIFF_PERCENT){
//             printf("Check       Result: %f\n", check_result);
//             printf("Under Check Result: %f\n", undercheck_result);
//             printf("new_diff_percent: %f\n", new_diff_percent);

//             free(a);
//             free(b);
//             free(c);
//             free(d);
//             free(d_new);
//             return 2;
//         }
//     }

    
//     return 0;
// }
