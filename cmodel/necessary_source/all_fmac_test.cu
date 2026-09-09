#include "gendata.h"
#include "mma_descriptor.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace std;

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>

#if NVIDIA_ON
#include <cuda_runtime.h>
#endif

#ifndef OUTPUT_DATA_DIR
#define OUTPUT_DATA_DIR "output_data"
#endif

#ifndef SCALE_OUTPUT_DATA_DIR
#define SCALE_OUTPUT_DATA_DIR "scale_output_data"
#endif

enum class SeedMode {
    Legacy,
    PerDtype,
    PerVec,
};

static const char* seed_mode_name(SeedMode mode) {
    switch (mode) {
        case SeedMode::Legacy: return "legacy";
        case SeedMode::PerDtype: return "per_dtype";
        case SeedMode::PerVec: return "per_vec";
    }
    return "unknown";
}

static bool parse_seed_mode(const char* text, SeedMode* mode) {
    if (strcmp(text, "legacy") == 0) {
        *mode = SeedMode::Legacy;
        return true;
    }
    if (strcmp(text, "per_dtype") == 0) {
        *mode = SeedMode::PerDtype;
        return true;
    }
    if (strcmp(text, "per_vec") == 0) {
        *mode = SeedMode::PerVec;
        return true;
    }
    return false;
}

static uint32_t mix_seed_component(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

static uint32_t make_repro_seed(uint32_t base_seed, int dtype, int vecnum) {
    uint32_t mixed = mix_seed_component(base_seed ^ (0x9e3779b9U * static_cast<uint32_t>(dtype + 1)));
    if (vecnum >= 0) {
        mixed ^= 0x85ebca6bU * static_cast<uint32_t>(vecnum + 1);
        mixed = mix_seed_component(mixed);
    }
    return mixed;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --dtype N              run only one dtype\n");
    printf("  --all-dtypes           run all supported instruction dtypes\n");
    printf("  --all-128b-dtypes      run one representative instruction per 128b input format\n");
    printf("  --list-dtypes          print all supported dtype ids\n");
    printf("  --bitsize N            input vector width in bits (default: 512)\n");
    printf("  --int4-k N             exact RTL int4 dot-product length; dtype must be 15-18\n");
    printf("  --vec-start N          first vec index to run (default: 0)\n");
    printf("  --vec-count N          how many vecs to run (default: 1000000)\n");
    printf("  --replay-vec N         replay one vector from existing dump files\n");
    printf("  --seed N               base RNG seed (default: current fixed seed)\n");
    printf("  --seed-mode MODE       one of: legacy, per_dtype, per_vec\n");
    printf("  --output-fp16          use fp16 accumulator/output when the dtype supports it\n");
    printf("  --blocksize16          mxf4 128b mode scale policy; enable per-16-data-block scales\n");
    printf("  --validate-reference   force reference nvmac validation even when NVIDIA_ON=0\n");
    printf("  --stop-on-mismatch     stop immediately after the first mismatch\n");
    printf("  --print-every N        progress print interval (default: 50000, 0 disables)\n");
    printf("  --help                 show this help message\n");
}

static void print_supported_dtypes() {
    int legacy_ids[] = {
        MMA_DTYPE_FP16, MMA_DTYPE_BF16, MMA_DTYPE_TF32,
        MMA_DTYPE_NVFP4, MMA_DTYPE_MXFP4, MMA_DTYPE_MXINT4
    };
    int rtl_int4_ids[] = {
        MMA_DTYPE_INT4_INT4_RTL, MMA_DTYPE_INT4_UINT4_RTL,
        MMA_DTYPE_UINT4_INT4_RTL, MMA_DTYPE_UINT4_UINT4_RTL
    };

    printf("Supported dtypes:\n");
    for (int id : legacy_ids) {
        printf("  %3d  %s\n", id, mma_dtype_name(id));
    }
    for (int id : rtl_int4_ids) {
        printf("  %3d  %s\n", id, mma_dtype_name(id));
    }
    for (int id = MMA_DTYPE_F8F6F4_MIXED_BASE;
         id < MMA_DTYPE_F8F6F4_MIXED_BASE + MMA_DTYPE_F8F6F4_MIXED_COUNT;
         ++id) {
        printf("  %3d  %s\n", id, mma_dtype_name(id));
    }
    for (int id = MMA_DTYPE_MXF8F6F4_MIXED_BASE;
         id < MMA_DTYPE_MXF8F6F4_MIXED_BASE + MMA_DTYPE_MXF8F6F4_MIXED_COUNT;
         ++id) {
        printf("  %3d  %s\n", id, mma_dtype_name(id));
    }
    for (int id = MMA_DTYPE_IMMA_U8S8_BASE;
         id < MMA_DTYPE_IMMA_U8S8_BASE + MMA_DTYPE_IMMA_U8S8_COUNT;
         ++id) {
        printf("  %3d  %s\n", id, mma_dtype_name(id));
    }
    for (int id = MMA_DTYPE_IMMA_U4S4_BASE;
         id < MMA_DTYPE_IMMA_U4S4_BASE + MMA_DTYPE_IMMA_U4S4_COUNT;
         ++id) {
        printf("  %3d  %s\n", id, mma_dtype_name(id));
    }
}

static void append_range(vector<int>* out, int base, int count) {
    for (int id = base; id < base + count; ++id) {
        out->push_back(id);
    }
}

static void append_default_float_dtypes(vector<int>* out) {
    int float_dtypes[] = {
        MMA_DTYPE_FP16, MMA_DTYPE_BF16, MMA_DTYPE_TF32,
        MMA_DTYPE_NVFP4, MMA_DTYPE_MXFP4, MMA_DTYPE_MXINT4,
        32, 38, 44, 50, 56
    };
    out->insert(out->end(), float_dtypes, float_dtypes + (sizeof(float_dtypes) / sizeof(int)));
}

static void append_all_supported_test_dtypes(vector<int>* out) {
    append_default_float_dtypes(out);
    append_range(out, MMA_DTYPE_INT4_INT4_RTL, 4);
    append_range(out, MMA_DTYPE_F8F6F4_MIXED_BASE, MMA_DTYPE_F8F6F4_MIXED_COUNT);
    append_range(out, MMA_DTYPE_MXF8F6F4_MIXED_BASE, MMA_DTYPE_MXF8F6F4_MIXED_COUNT);
    append_range(out, MMA_DTYPE_IMMA_U8S8_BASE, MMA_DTYPE_IMMA_U8S8_COUNT);
    append_range(out, MMA_DTYPE_IMMA_U4S4_BASE, MMA_DTYPE_IMMA_U4S4_COUNT);
}

static void append_128b_representative_test_dtypes(vector<int>* out) {
    append_default_float_dtypes(out);
    append_range(out, MMA_DTYPE_F8F6F4_MIXED_BASE, MMA_DTYPE_F8F6F4_MIXED_COUNT);
    append_range(out, MMA_DTYPE_MXF8F6F4_MIXED_BASE, MMA_DTYPE_MXF8F6F4_MIXED_COUNT);
    append_range(out, MMA_DTYPE_IMMA_U8S8_BASE, 8);   // m8n8k16 s8/u8 family
    append_range(out, MMA_DTYPE_IMMA_U4S4_BASE, 8);   // m8n8k32 s4/u4 family
}

static bool dtype_uses_scale(int dtype) {
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    return desc && desc->uses_scale;
}

static void create_no_scale_placeholder_file_if_needed(const vector<int>& dtypes_to_run, int required_vec_count) {
    if (!fmac_dump_output_enabled()) {
        return;
    }
    bool need_no_scale_file = false;
    for (int dtype : dtypes_to_run) {
        if (!dtype_uses_scale(dtype)) {
            need_no_scale_file = true;
            break;
        }
    }
    if (!need_no_scale_file) {
        return;
    }

    if (mkdir(SCALE_OUTPUT_DATA_DIR, 0775) != 0 && errno != EEXIST) {
        perror(SCALE_OUTPUT_DATA_DIR);
        return;
    }

    char placeholder_path[256];
    snprintf(placeholder_path, sizeof(placeholder_path), "%s/data_for_no_scale.bin.log", SCALE_OUTPUT_DATA_DIR);
    FILE* fp = fopen(placeholder_path, "wb");
    if (!fp) {
        perror(placeholder_path);
        return;
    }

    uint8_t zeros[4096] = {0};
    int remaining = required_vec_count;
    while (remaining > 0) {
        int chunk = remaining > (int)sizeof(zeros) ? (int)sizeof(zeros) : remaining;
        if (fwrite(zeros, sizeof(uint8_t), chunk, fp) != (size_t)chunk) {
            perror(placeholder_path);
            break;
        }
        remaining -= chunk;
    }
    fclose(fp);
}

static void destroy_data_file(DataFile* df) {
    if (!df) return;
    destroy_data_file_workspace(df);
    delete df->a;
    delete df->b;
    delete df->c;
    delete df->d;
    delete df->a_scale;
    delete df->b_scale;
    free(df);
}

static FileIntArrayManager::DataType dtype_accumulator_file_type(const MmaDescriptor& desc, bool float_type) {
    (void)desc;
    (void)float_type;
    return FileIntArrayManager::INT32;
}

static const char* output_shape_token_for_runtime(const MmaDescriptor& desc) {
    if (!mma_topology_128_input_enabled()) {
        return mma_shape_token(desc.shape);
    }
    if (desc.id == MMA_DTYPE_FP16 || desc.id == MMA_DTYPE_BF16) {
        return "m16n8k8";
    }
    if (desc.id == MMA_DTYPE_TF32) {
        return "m16n8k4";
    }
    return mma_shape_token(desc.shape);
}

static string build_dtype_output_prefix_base(const MmaDescriptor& desc, bool float_type) {
    char prefix[128];
    snprintf(prefix, sizeof(prefix), "%s_%s_%s_to_%s",
             output_shape_token_for_runtime(desc),
             mma_operand_file_token(desc, true),
             mma_operand_file_token(desc, false),
             mma_accumulator_kind_token(desc, float_type));
    return string(prefix);
}

static string build_dtype_output_prefix(const MmaDescriptor& desc, bool float_type) {
    string prefix = build_dtype_output_prefix_base(desc, float_type);
    if (mma_is_rtl_int4_dtype(desc.id) && mma_rtl_int4_k() > 0) {
        prefix += "_k" + to_string(mma_rtl_int4_k());
    }
    if (mma_dtype_uses_mxf4_topology(desc.id)) {
        prefix += mma_mxf4_blocksize16_enabled() ? "_b16" : "_b32";
    }
    return prefix;
}

static vector<string> build_dtype_replay_prefixes(const MmaDescriptor& desc, bool float_type) {
    vector<string> prefixes;
    auto add_unique_prefix = [&prefixes](const string& prefix) {
        for (const string& existing : prefixes) {
            if (existing == prefix) {
                return;
            }
        }
        prefixes.push_back(prefix);
    };

    add_unique_prefix(build_dtype_output_prefix(desc, float_type));
    // An explicit K identifies a distinct vector layout.  Do not fall back to
    // the unsuffixed files, because their per-vector byte stride may differ.
    if (mma_is_rtl_int4_dtype(desc.id) && mma_rtl_int4_k() > 0) {
        return prefixes;
    }
    add_unique_prefix(build_dtype_output_prefix_base(desc, float_type));
    add_unique_prefix(desc.file_stem);
    return prefixes;
}

static DataFile* create_descriptor_data_file_with_prefix(
    const MmaDescriptor& desc,
    const string& prefix,
    int is_create,
    bool float_type)
{
    string stem = string(OUTPUT_DATA_DIR) + "/" + prefix;
    FileIntArrayManager::DataType c_dtype = dtype_accumulator_file_type(desc, float_type);
    return create_data_file_ex(
        stem + "_a",
        stem + "_b",
        stem + "_c",
        stem + "_d",
        file_manager_dtype_for_operand_kind(desc.a_kind),
        file_manager_dtype_for_operand_kind(desc.b_kind),
        c_dtype,
        c_dtype,
        is_create,
        desc.uses_scale);
}

static size_t file_manager_data_type_bytes(FileIntArrayManager::DataType dtype) {
    switch (dtype) {
        case FileIntArrayManager::INT16: return 2;
        case FileIntArrayManager::INT32: return 4;
        default: return 1;
    }
}

static bool file_has_required_vectors(const char* path, size_t bytes_per_vec, int required_vec_count) {
    if (!path || bytes_per_vec == 0 || required_vec_count <= 0) {
        return false;
    }
    struct stat st;
    if (stat(path, &st) != 0) {
        return false;
    }
    return (st.st_size / static_cast<off_t>(bytes_per_vec)) >= required_vec_count;
}

static bool replay_data_files_exist(const DataFile* df, bool need_scale) {
    if (!df || !df->a || !df->b || !df->c || !df->d) {
        return false;
    }
    if (access(df->a->filename_double_bin, F_OK) != 0 ||
        access(df->b->filename_double_bin, F_OK) != 0 ||
        access(df->c->filename_double_bin, F_OK) != 0 ||
        access(df->d->filename_double_bin, F_OK) != 0) {
        return false;
    }
    if (need_scale) {
        if (!df->a_scale || !df->b_scale) {
            return false;
        }
        if (access(df->a_scale->filename_double_bin, F_OK) != 0 ||
            access(df->b_scale->filename_double_bin, F_OK) != 0) {
            return false;
        }
    }
    return true;
}

static bool replay_data_files_cover_vec(
    const DataFile* df,
    const MmaDescriptor& desc,
    int bitsize,
    bool float_type,
    int vecnum)
{
    if (!df || vecnum < 0) {
        return false;
    }
    int required_vec_count = vecnum + 1;
    FileIntArrayManager::DataType c_dtype = dtype_accumulator_file_type(desc, float_type);
    int a_logical_count = replay_logical_elem_count_for_desc_operand(desc, desc.a_kind, bitsize);
    int b_logical_count = replay_logical_elem_count_for_desc_operand(desc, desc.b_kind, bitsize);
    size_t a_bytes_per_vec =
        (desc.a_kind == MmaOperandKind::S4 || desc.a_kind == MmaOperandKind::U4) &&
                desc.id != MMA_DTYPE_MXINT4
            ? static_cast<size_t>((a_logical_count + 1) / 2)
            : static_cast<size_t>(a_logical_count) *
                  file_manager_data_type_bytes(file_manager_dtype_for_operand_kind(desc.a_kind));
    size_t b_bytes_per_vec =
        (desc.b_kind == MmaOperandKind::S4 || desc.b_kind == MmaOperandKind::U4) &&
                desc.id != MMA_DTYPE_MXINT4
            ? static_cast<size_t>((b_logical_count + 1) / 2)
            : static_cast<size_t>(b_logical_count) *
                  file_manager_data_type_bytes(file_manager_dtype_for_operand_kind(desc.b_kind));
    size_t c_bytes_per_vec = file_manager_data_type_bytes(c_dtype);

    if (!file_has_required_vectors(df->a->filename_double_bin, a_bytes_per_vec, required_vec_count) ||
        !file_has_required_vectors(df->b->filename_double_bin, b_bytes_per_vec, required_vec_count) ||
        !file_has_required_vectors(df->c->filename_double_bin, c_bytes_per_vec, required_vec_count) ||
        !file_has_required_vectors(df->d->filename_double_bin, c_bytes_per_vec, required_vec_count)) {
        return false;
    }
    if (desc.uses_scale) {
        size_t scale_bytes_per_vec = (size_t)valid_scale_count_for_desc_bitsize(desc, bitsize);
        return df->a_scale && df->b_scale &&
               file_has_required_vectors(df->a_scale->filename_double_bin, scale_bytes_per_vec, required_vec_count) &&
               file_has_required_vectors(df->b_scale->filename_double_bin, scale_bytes_per_vec, required_vec_count);
    }
    return true;
}

static DataFile* create_dtype_data_file(int dtype, int is_create, bool float_type) {
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (!desc) {
        return NULL;
    }

    vector<string> prefixes = build_dtype_replay_prefixes(*desc, float_type);
    for (const string& prefix : prefixes) {
        DataFile* df = create_descriptor_data_file_with_prefix(*desc, prefix, is_create, float_type);
        if (is_create || replay_data_files_exist(df, desc->uses_scale)) {
            return df;
        }
        destroy_data_file(df);
    }

    return create_descriptor_data_file_with_prefix(*desc, prefixes.back(), is_create, float_type);
}

static int replay_dtype_data_file_with_fallback(
    int dtype,
    int bitsize,
    int vecnum,
    bool float_type,
    bool print_en)
{
    const MmaDescriptor* desc = mma_get_descriptor(dtype);
    if (!desc) {
        return -1;
    }

    vector<string> prefixes = build_dtype_replay_prefixes(*desc, float_type);
    int first_replay_status = -1;
    bool tried_replay = false;
    for (const string& prefix : prefixes) {
        DataFile* df = create_descriptor_data_file_with_prefix(*desc, prefix, 0, float_type);
        if (!df) {
            continue;
        }

        if (!replay_data_files_cover_vec(df, *desc, bitsize, float_type, vecnum)) {
            destroy_data_file(df);
            continue;
        }

        int replay_status = replay_data_file(df, bitsize, dtype, vecnum, float_type, print_en);
        destroy_data_file(df);
        if (replay_status >= 0) {
            return replay_status;
        }

        if (!tried_replay) {
            first_replay_status = replay_status;
            tried_replay = true;
        }
    }

    return tried_replay ? first_replay_status : -1;
}

// ===== dtype 名字映射 =====
const char* get_dtype_name(int dtype){
    return mma_dtype_name(dtype);
}

#if NVIDIA_ON
static bool verify_nvidia_runtime() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count <= 0) {
        fprintf(stderr,
                "CUDA device check failed: %s. Run NVIDIA_ON=1 tests only where the GPU device nodes are available.\n",
                cudaGetErrorString(status));
        return false;
    }

    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, 0);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA device property query failed: %s\n", cudaGetErrorString(status));
        return false;
    }

    printf("CUDA device 0: %s (compute capability %d.%d)\n",
           prop.name, prop.major, prop.minor);
    return true;
}
#endif

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    int is_rand = 1;  // 1表示随机生成数据，0表示顺序生成数据
    int bitsize = 8 * 64; // 数据位数
    int total_vecnum = 1000;
    bool float_type = 0; //0表示目标格式为fp32，1表示目标格式为fp16
    int target_dtype = -1; // -1表示测试所有dtype，其他值表示只测试指定dtype
    int vec_start = 0;
    int vec_count = total_vecnum;  // 测试向量组数 ，默认1000000
    int print_every = 50000;
    bool stop_on_mismatch = false;
    bool run_all_dtypes = false;
    bool run_all_128b_dtypes = false;
    uint32_t base_seed = current_random_seed();
    SeedMode seed_mode = SeedMode::Legacy;
    int replay_vec = -1;
    int blocksize16 = 0;
    int int4_k = 0;
    bool int4_k_specified = false;

    // 输入参数处理
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) {
            target_dtype = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--all-dtypes") == 0) {
            run_all_dtypes = true;
        } else if (strcmp(argv[i], "--all-128b-dtypes") == 0) {
            run_all_128b_dtypes = true;
        } else if (strcmp(argv[i], "--list-dtypes") == 0) {
            print_supported_dtypes();
            return 0;
        } else if (strcmp(argv[i], "--bitsize") == 0 && i + 1 < argc) {
            bitsize = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--int4-k") == 0 && i + 1 < argc) {
            int4_k = atoi(argv[++i]);
            int4_k_specified = true;
        } else if (strcmp(argv[i], "--vec-start") == 0 && i + 1 < argc) {
            vec_start = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vec-count") == 0 && i + 1 < argc) {
            vec_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--replay-vec") == 0 && i + 1 < argc) {
            replay_vec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            base_seed = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 0));
        } else if (strcmp(argv[i], "--seed-mode") == 0 && i + 1 < argc) {
            if (!parse_seed_mode(argv[++i], &seed_mode)) {
                fprintf(stderr, "Unsupported seed mode: %s\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "--output-fp16") == 0) {
            float_type = true;
        } else if (strcmp(argv[i], "--blocksize16") == 0) {
            blocksize16 = 1;
        } else if (strcmp(argv[i], "--validate-reference") == 0) {
            set_reference_validation_enabled(true);
        } else if (strcmp(argv[i], "--stop-on-mismatch") == 0) {
            stop_on_mismatch = true;
        } else if (strcmp(argv[i], "--print-every") == 0 && i + 1 < argc) {
            print_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (bitsize <= 0 || (!int4_k_specified && (bitsize % 32) != 0) ||
        vec_start < 0 || vec_count <= 0 || print_every < 0 || replay_vec < -1 ||
        (blocksize16 != 0 && blocksize16 != 1) ||
        (int4_k_specified && int4_k <= 0)) {
        fprintf(stderr, "Invalid bitsize, vec range, or print interval.\n");
        return 1;
    }

#if NVIDIA_ON
    if (!verify_nvidia_runtime()) {
        return 1;
    }
#endif

    mma_set_mxf4_blocksize16(blocksize16);
    mma_set_rtl_int4_k(0);

    if (target_dtype != -1 && mma_dtype_is_deprecated(target_dtype)) {
        fprintf(stderr, "dtype %d has been removed; use dtype %d instead.\n",
                target_dtype,
                mma_dtype_deprecated_replacement(target_dtype));
        return 1;
    }

    if (int4_k_specified) {
        if (!mma_is_rtl_int4_dtype(target_dtype)) {
            fprintf(stderr, "--int4-k requires exactly one RTL int4 dtype (15-18).\n");
            return 1;
        }
        mma_set_rtl_int4_k(int4_k);
    }

    if (replay_vec >= 0 && target_dtype == -1) {
        fprintf(stderr, "--replay-vec requires --dtype.\n");
        return 1;
    }

    if (replay_vec < 0 &&
        seed_mode == SeedMode::Legacy &&
        (target_dtype != -1 || vec_start != 0 || vec_count != total_vecnum)) {
        printf("Note: legacy RNG mode preserves the original cross-dtype stream. "
               "Filtering dtype or vec range changes that stream. "
               "Use --seed-mode per_dtype or --seed-mode per_vec for isolated repro.\n");
    }

    vector<int> dtypes_to_run;
    if (target_dtype != -1) {
        dtypes_to_run.push_back(target_dtype);
    } else if (run_all_128b_dtypes) {
        append_128b_representative_test_dtypes(&dtypes_to_run);
    } else if (run_all_dtypes) {
        append_all_supported_test_dtypes(&dtypes_to_run);
    } else {
        append_default_float_dtypes(&dtypes_to_run);
    }

    reset_random_generator(base_seed);
    bool stop_requested = false;
    int overall_mismatch_count = 0;
    bool overall_error = false;

    if (replay_vec >= 0) {
        if (!mma_get_descriptor(target_dtype)) {
            fprintf(stderr, "Failed to open replay files for dtype %d\n", target_dtype);
            return 1;
        }
        int replay_status = replay_dtype_data_file_with_fallback(
            target_dtype, bitsize, replay_vec, float_type, 1);
        return replay_status == 0 ? 0 : 1;
    }

    create_no_scale_placeholder_file_if_needed(dtypes_to_run, vec_start + vec_count);
    for (int dtype : dtypes_to_run) {
        const char* dtype_name = get_dtype_name(dtype);
        if (seed_mode == SeedMode::PerDtype) {
            reset_random_generator(make_repro_seed(base_seed, dtype, -1));
        }

        DataFile* df = create_dtype_data_file(dtype, 1, float_type);
        if (!df) {
            printf("%s Failed to create data\n", dtype_name);
            continue;
        }

        bool all_correct = true;
        int fail_index = -1;
        int mismatch_count = 0;
        int vec_end = vec_start + vec_count;

        for (int vecnum = vec_start; vecnum < vec_end; vecnum++) {
            if (seed_mode == SeedMode::PerVec) {
                reset_random_generator(make_repro_seed(base_seed, dtype, vecnum));
            }

            int result = gen_data_file(df, bitsize, dtype, is_rand, float_type, 0, vecnum);
            if (result != 0) {
                all_correct = false;
                fail_index = vecnum;
                overall_error = true;
                stop_requested = stop_on_mismatch;
                break;
            }

            if (last_call_had_mismatch()) {
                const MismatchRecord& mismatch = get_last_mismatch_record();
                all_correct = false;
                if (fail_index < 0) {
                    fail_index = vecnum;
                }
                mismatch_count++;
                if (mismatch_count == 1) {
                    printf("First mismatch summary: dtype=%s(%d) vec=%d tag=%s "
                           "my=0x%08x nv=0x%08x seed_mode=%s base_seed=%u\n",
                           dtype_name,
                           dtype,
                           vecnum,
                           mismatch.tag ? mismatch.tag : "unknown",
                           static_cast<uint32_t>(mismatch.my_result),
                           static_cast<uint32_t>(mismatch.nv_result),
                           seed_mode_name(seed_mode),
                           base_seed);
                    if (seed_mode == SeedMode::PerVec) {
                        printf("Direct repro: ./all_fmac_test --dtype %d --seed %u "
                               "--seed-mode per_vec --vec-start %d --vec-count 1 "
                               "--stop-on-mismatch\n",
                               dtype,
                               base_seed,
                               vecnum);
                    }
                }
                if (stop_on_mismatch) {
                    stop_requested = true;
                    break;
                }
            }

            int completed = vecnum - vec_start + 1;
            if (print_every > 0 && completed % print_every == 0) {
                printf("%s test progress: %d vectors\n", dtype_name, completed);
            }
        }

        overall_mismatch_count += mismatch_count;

        if (all_correct) {
            printf("%s Format test vec[%d, %d) finished with no mismatches\n",
                   dtype_name, vec_start, vec_end);
        } else {
            printf("%s Format test vec[%d, %d) first mismatch at vec %d, mismatch_count=%d\n",
                   dtype_name, vec_start, vec_end, fail_index, mismatch_count);
        }

        destroy_data_file(df);
        if (stop_requested) {
            break;
        }
    }

    fmac_trace_close_all_files();
    return (overall_error || overall_mismatch_count != 0) ? 1 : 0;
}
