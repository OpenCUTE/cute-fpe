#include "fmac.h"
#include "FileManager.h"
#include "myrandom.h"
#include "nv_mac.h"
#include <cstdint>
#include <cstring>
#include <string>  // ← 添加这一行

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
    FileIntArrayManager* a; // 文件管理器
    FileIntArrayManager* a_scale;
    FileIntArrayManager* b; // 文件管理器
    FileIntArrayManager* b_scale;
    FileIntArrayManager* c; // 文件管理器
    FileIntArrayManager* d; // 文件管理器
} DataFile;

DataFile* create_data_file(string a_fname, string b_fname, string c_fname, string d_fname, int dtype, int is_create, bool float_type = 0) {
    DataFile* df = (DataFile*)malloc(sizeof(DataFile));
    if (!df) return NULL;
    if(dtype >= 7 && dtype <= 12) {
        df->a_scale = new FileIntArrayManager("scale_" + a_fname, FileIntArrayManager::INT8, is_create);
        df->b_scale = new FileIntArrayManager("scale_" + b_fname, FileIntArrayManager::INT8, is_create);
    }
    if((dtype == 0) || (dtype == 4) || (dtype == 5) || (dtype == 6) || (dtype == 7) || (dtype == 8) || (dtype == 9) || (dtype == 10) || 
       (dtype == 11) || (dtype == 12) || (dtype == 13) || (dtype == 14) || (dtype == 15) || (dtype == 16)){
        df->a = new FileIntArrayManager(a_fname, FileIntArrayManager::INT8, is_create);
        df->b = new FileIntArrayManager(b_fname, FileIntArrayManager::INT8, is_create);
    }
    else if((dtype == 1) || (dtype == 2)){
        df->a = new FileIntArrayManager(a_fname, FileIntArrayManager::INT16, is_create);
        df->b = new FileIntArrayManager(b_fname, FileIntArrayManager::INT16, is_create);
    }
    else{
        df->a = new FileIntArrayManager(a_fname, FileIntArrayManager::INT32, is_create);
        df->b = new FileIntArrayManager(b_fname, FileIntArrayManager::INT32, is_create);
    }
    
    if(!float_type){
        df->c = new FileIntArrayManager(c_fname, FileIntArrayManager::INT32, is_create);
        df->d = new FileIntArrayManager(d_fname, FileIntArrayManager::INT32, is_create);
    } 
    else{
        df->c = new FileIntArrayManager(c_fname, FileIntArrayManager::INT16, is_create);
        df->d = new FileIntArrayManager(d_fname, FileIntArrayManager::INT16, is_create);
    }
    return df;
}

int gen_data_file(DataFile* df, int bitsize, int dtype, int is_rand, bool float_type = 0, bool print_en = 0) {
    if (!df || !df->a || !df->b || !df->c || !df->d) return -1;

    if ((dtype == 7 || dtype == 8 || dtype == 9 || dtype == 10 || dtype == 11 || dtype == 12) && (!df->a_scale || !df->b_scale)) return -1;

    if ((dtype == 0) || (dtype == 4) || (dtype == 5) || (dtype == 6)) { // INT8
        int size = bitsize / 8; // INT8每个元素1字节
        int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* b = (int8_t*)malloc(size * sizeof(int8_t));
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));

        if(is_rand){
            for (int i = 0; i < size; ++i) {
                // 随机生成-128到127的值
                a[i] = rand() % 256 - 128; 
                b[i] = rand() % 256 - 128;
            }
            c[0] = rand();
        }
        else {
            for (int i = 0; i < size; ++i){
                a[i] = i;
                b[i] = i;
            }
            c[0] = 0; // 初始值为0
        }
        // int32_t temp_zero = 0;
        d[0] = mymac(a, b, c, size, dtype);

        df->a->write_int8_array_to_file(a, size, 1);
        df->b->write_int8_array_to_file(b, size, 1);
        df->c->write_int32_array_to_file(c, 1, 1);
        df->d->write_int32_array_to_file(d, 1, 1);

        free(a);
        free(b);
        free(c);
        free(d);
    }
    else if (dtype == 1) {
        int size = bitsize / 16; // INT16每个元素2字节，INT32每个元素4字节
        int16_t* a = (int16_t*)malloc(size * sizeof(int16_t));
        int16_t* b = (int16_t*)malloc(size * sizeof(int16_t));
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d_nv = (int32_t*)malloc(sizeof(int32_t));

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                a[i] = generate_normalized_fp16(); // 将__fp16转换为int16_t
                b[i] = generate_normalized_fp16(); // 将__fp16转换为int16_t
            }
            if(!float_type){
                c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            }
            else {
                c[0] = (uint16_t)generate_normalized_fp16(); // 先去掉符号扩展,只保留低16位
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
            memcpy(&a[0], &val, sizeof(uint16_t));

            // val = 0x94d6;
            // memcpy(&b[0], &val, sizeof(uint16_t));
            // val = 0x4fdc;
            // memcpy(&b[0], &val, sizeof(uint16_t));
            val = 0x22ae;
            memcpy(&b[0], &val, sizeof(uint16_t));
            if(!float_type){
                // c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            }
            else {
                uint16_t val16 = 0x8009;
                c[0] = (int32_t)val16;
            }
        }
        d[0] = mymac(a, b, c, size, dtype, float_type);

        d_nv[0] = nvmac(a, b, c, size, dtype, float_type);

        if (d[0] != d_nv[0]) {
            // Formatted output: hex (raw bits) + decimal (numeric value) for debugging
            printf("FB16 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvfp16mac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
            df->a->write_int16_array_to_file(a, size, 1);
            df->b->write_int16_array_to_file(b, size, 1);
            if(!float_type){
                df->c->write_int32_array_to_file(c, 1, 1);
                df->d->write_int32_array_to_file(d, 1, 1);
            }
            else {
                int16_t c16 = (int16_t)c[0];
                int16_t d16 = (int16_t)d[0];

                df->c->write_int16_array_to_file(&c16,1,1);
                df->d->write_int16_array_to_file(&d16,1,1);
            }
        } else if(print_en) {
            // Optional: log success for debugging
            printf("TF32 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

        free(a);
        free(b);
        free(c);
        free(d);
        free(d_nv);
    }
    else if (dtype == 2) {
        int size = bitsize / 16; // INT16每个元素2字节，INT32每个元素4字节
        int16_t* a = (int16_t*)malloc(size * sizeof(int16_t));
        int16_t* b = (int16_t*)malloc(size * sizeof(int16_t));
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d_nv = (int32_t*)malloc(sizeof(int32_t));

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                a[i] = generate_normalized_bf16(); // 将__fp16转换为int16_t
                b[i] = generate_normalized_bf16(); // 将__fp16转换为int16_t
            }
            c[0] = generate_normalized_fp32(); // 将float转换为int32_t
        } else if(is_rand != 0){
            gen_exception_bf16(a, b, c, size, is_rand);
        } else{
            for (int i = 0; i < size; ++i){
                float temp_a_f = i; // 将i转换为__fp16
                float temp_b_f = i; // 将i转换为__fp16
                int32_t temp_a_32 = *((int32_t*)&temp_a_f) >> 16; // 将float转换为int32_t
                int32_t temp_b_32 = *((int32_t*)&temp_b_f) >> 16; // 将float转换为int32_t
                int16_t temp_a = (int16_t)(temp_a_32 & 0xFFFF); // 取低16位
                int16_t temp_b = (int16_t)(temp_b_32 & 0xFFFF); // 取低16位
                // 将int32_t转换为int16
                a[i] = temp_a;
                b[i] = temp_b;
            }
            c[0] = 0; // 初始值为0
        }
        d[0] = mymac(a, b, c, size, dtype, float_type);

        d_nv[0] = nvmac(a, b, c, size, dtype, float_type);

        if (d[0] != d_nv[0]) {
            // Formatted output: hex (raw bits) + decimal (numeric value) for debugging
            printf("BF32 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvtf32mac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
            df->a->write_int16_array_to_file(a, size, 1);
            df->b->write_int16_array_to_file(b, size, 1);
            df->c->write_int32_array_to_file(c, 1, 1);
            df->d->write_int32_array_to_file(d, 1, 1);
        } else if(print_en) {
            // Optional: log success for debugging
            printf("BF16 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

        free(a);
        free(b);
        free(c);
        free(d);
        free(d_nv);
    }
    else if (dtype == 3) {
        int size = bitsize / 32; // INT16每个元素2字节，INT32每个元素4字节
        int32_t* a = (int32_t*)malloc(size * sizeof(int32_t));
        int32_t* b = (int32_t*)malloc(size * sizeof(int32_t));
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d_nv = (int32_t*)malloc(sizeof(int32_t));

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                a[i] = generate_normalized_fp32() & 0xFFFFE000; // 将__fp16转换为int16_t
                b[i] = generate_normalized_fp32() & 0xFFFFE000; // 将__fp16转换为int16_t
                // a[i] = 0x7F802000;
                // b[i] = 0x00100000;
            }
            c[0] = generate_normalized_fp32(); // 将float转换为int32_t
        } else if(is_rand != 0){
            gen_exception_tf32(a, b, c, size, is_rand);
        } else{
            for (int i = 0; i < size; ++i){
                float temp_a = (float)i; // 将i转换为__fp16
                float temp_b = (float)i; // 将i转换为__fp16
                a[i] = *((int32_t*)&temp_a); // 将__fp16转换为int16_t
                b[i] = *((int32_t*)&temp_b); //
            }
            c[0] = 0; // 初始值为0
        }
        d[0] = mymac(a, b, c, size, dtype, float_type);
        
        d_nv[0] = nvmac(a, b, c, size, dtype, float_type);

        if (d[0] != d_nv[0]) {
            // Formatted output: hex (raw bits) + decimal (numeric value) for debugging
            printf("TF32 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvtf32mac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
            df->a->write_int32_array_to_file(a, size, 1);
            df->b->write_int32_array_to_file(b, size, 1);
            df->c->write_int32_array_to_file(c, 1, 1);
            df->d->write_int32_array_to_file(d, 1, 1);
        } else if(print_en) {
            // Optional: log success for debugging
            printf("TF32 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

        // df->a->write_int32_array_to_file(a, size, 1);
        // df->b->write_int32_array_to_file(b, size, 1);
        // df->c->write_int32_array_to_file(c, 1, 1);
        // df->d->write_int32_array_to_file(d, 1, 1);

        free(a);
        free(b);
        free(c);
        free(d);
        free(d_nv);
    }
    else if (dtype == 7 || dtype == 8) {       // mxfp8e4m3 与 mxfp8e5m2
        int size = bitsize / 8;
        int scale_size = bitsize / 4 / 16;
        int scale_valid_num = bitsize / 8 / 32;
        // unsigned int a_bits[4] = {0x56235f15, 0x95cbbdc8, 0x1022d5d2, 0xdc3620c5};
        // unsigned int b_bits[4] = {0x70ef27d5, 0x77993919, 0x0a1b2e40, 0xdd5e2412};
        // unsigned int c_bits[1] = {0xca4df250};
        int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* b = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* a_scale = (int8_t*)malloc(2 * scale_size * sizeof(int8_t));
        int8_t* b_scale = (int8_t*)malloc(2 * scale_size * sizeof(int8_t));
        int8_t* a_scale_to_file = a_scale + scale_size;
        int8_t* b_scale_to_file = b_scale + scale_size;
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        // memcpy((void *) a, (void *)a_bits, sizeof(a_bits));
        // memcpy((void *) b, (void *)b_bits, sizeof(b_bits));
        // memcpy((void *) c, (void *)c_bits, sizeof(c));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d_nv = (int32_t*)malloc(sizeof(int32_t));

        for (int i = 0; i < scale_size; ++i) {
            a_scale[i] = 0;
            b_scale[i] = 0;
        }

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                if (dtype == 7)
                {
                    a[i] = generate_normalized_e4m3() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e4m3() ; // 将__fp16转换为int16_t}
                } else {
                    a[i] = generate_normalized_e5m2() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e5m2() ; // 将__fp16转换为int16_t}
                }
            }
            c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            for (int i = 0; i < scale_valid_num; ++i) {
                a_scale[i] = generate_normalized_ue8m0(); // 随机生成-128到127
                b_scale[i] = generate_normalized_ue8m0(); // 随机生成-128到127
            }
        } else if(is_rand != 0){
            gen_exception_fp8(a, b, c, size, is_rand);
        } else{
            for (int i = 0; i < scale_valid_num; ++i) {
                a_scale[i] = 127; // 初始值为1
                b_scale[i] = 127; // 初始值为1
            }
            for (int i = 0; i < size; ++i){
                int temp_a = 64; // 将i转换为__fp16
                int temp_b = 64; // 将i转换为__fp16
                a[i] = *((int8_t*)&temp_a); // 将__fp16转换为int16_t
                b[i] = *((int8_t*)&temp_b); //
                printf("a[%d]: %02x, b[%d]: %02x\n", i, (uint8_t)a[i], i, (uint8_t)b[i]);
            }
            // c[0] = 0; // 初始值为0
            c[0] = 0x3f800000;
        }

        d[0] = mymac_scale(a, a_scale, b, b_scale, c, size, dtype);
        
        d_nv[0] = nvmac_scale(a, a_scale, b, b_scale, c, size, dtype);

        if (d[0] != d_nv[0]) {
            // Formatted output: hex (raw bits) + decimal (numeric value) for debugging
            printf("FP8 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvfp8mac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
            
            for (int i = 0; i < scale_size; ++i) {
                a_scale_to_file[scale_size - 1 - i] = a_scale[i];
                b_scale_to_file[scale_size - 1 - i] = b_scale[i];
            }
            
            df->a->write_int8_array_to_file(a, size, 1);
            df->b->write_int8_array_to_file(b, size, 1);
            df->c->write_int32_array_to_file(c, 1, 1);
            df->d->write_int32_array_to_file(d, 1, 1);
            df->a_scale->write_int8_array_to_file(a_scale_to_file, scale_size, 1);
            df->b_scale->write_int8_array_to_file(b_scale_to_file, scale_size, 1);
        } else if(print_en) {
            // Optional: log success for debugging
            printf("FP6FP4 MAC results match: d[0] = 0x%08x\n", d[0]);
        }
        free(a_scale);
        free(b_scale);

        free(a);
        free(b);
        free(c);
        free(d);
        free(d_nv);

    }else if (dtype == 13 || dtype == 14) {
        int size = bitsize / 8;
        int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* b = (int8_t*)malloc(size * sizeof(int8_t));
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d_nv = (int32_t*)malloc(sizeof(int32_t));

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                if (dtype == 11) {
                    a[i] = generate_normalized_e4m3() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e4m3() ; // 将__fp16转换为int16_t}
                } else {
                    a[i] = generate_normalized_e5m2() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e5m2() ; // 将__fp16转换为int16_t}
                }
            }

            if(!float_type){
                c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            }
            else {
                c[0] = (uint16_t)generate_normalized_fp16(); // 先去掉符号扩展,只保留低16位
            }
            
        } else if(is_rand != 0){
            gen_exception_fp8(a, b, c, size, is_rand);
        } else{
            uint8_t val;

            // val = 0xc8a8;
            // memcpy(&a[0], &val, sizeof(uint16_t));
            // val = 0x1bec;
            // memcpy(&a[0], &val, sizeof(uint16_t));
            val = 0xd4;
            memcpy(&a[0], &val, sizeof(uint8_t));

            // val = 0x94d6;
            // memcpy(&b[0], &val, sizeof(uint16_t));
            // val = 0x4fdc;
            // memcpy(&b[0], &val, sizeof(uint16_t));
            val = 0x18;
            memcpy(&b[0], &val, sizeof(uint8_t));
            if(!float_type){
                // c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            }
            else {
                uint16_t val16 = 0x8009;
                c[0] = (int32_t)val16;
            }
        }
        d[0] = mymac(a, b, c, size, dtype, float_type);
        d_nv[0] = nvmac(a, b, c, size, dtype, float_type);

        if (d[0] != d_nv[0]) {
            // Formatted output: hex (raw bits) + decimal (numeric value) for debugging
            printf("FP8 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvfp8mac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
            df->a->write_int8_array_to_file(a, size, 1);
            df->b->write_int8_array_to_file(b, size, 1);
            if(!float_type){
                df->c->write_int32_array_to_file(c, 1, 1);
                df->d->write_int32_array_to_file(d, 1, 1);
            }
            else {
                int16_t c16 = (int16_t)c[0];
                int16_t d16 = (int16_t)d[0];

                df->c->write_int16_array_to_file(&c16,1,1);
                df->d->write_int16_array_to_file(&d16,1,1);
            }
        } else if(print_en) {
            // Optional: log success for debugging
            printf("TF32 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

        free(a);
        free(b);
        free(c);
        free(d);
        free(d_nv);
    }
    //mxfp6
    else if (dtype == 11 || dtype == 12 || dtype == 17) {       // mxfp6e2m3 与 mxfp6e3m2 与mxfp4e2m1
        int size = bitsize / 8;
        int scale_size = bitsize / 4 / 16;
        int scale_valid_num = bitsize / 8 / 32;
        // unsigned int a_bits[4] = {0x56235f15, 0x95cbbdc8, 0x1022d5d2, 0xdc3620c5};
        // unsigned int b_bits[4] = {0x70ef27d5, 0x77993919, 0x0a1b2e40, 0xdd5e2412};
        // unsigned int c_bits[1] = {0xca4df250};
        int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* b = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* a_scale = (int8_t*)malloc(2 * scale_size * sizeof(int8_t));
        int8_t* b_scale = (int8_t*)malloc(2 * scale_size * sizeof(int8_t));
        int8_t* a_scale_to_file = a_scale + scale_size;
        int8_t* b_scale_to_file = b_scale + scale_size;
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        // memcpy((void *) a, (void *)a_bits, sizeof(a_bits));
        // memcpy((void *) b, (void *)b_bits, sizeof(b_bits));
        // memcpy((void *) c, (void *)c_bits, sizeof(c));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d_nv = (int32_t*)malloc(sizeof(int32_t));

        for (int i = 0; i < scale_size; ++i) {
            a_scale[i] = 0;
            b_scale[i] = 0;
        }

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                if (dtype == 11)
                {
                    a[i] = generate_normalized_e2m3() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e2m3() ; // 将__fp16转换为int16_t}
                } else if(dtype == 17) {
                    a[i] = generate_normalized_e2m1() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e2m1() ; // 将__fp16转换为int16_t}
                } else {
                    a[i] = generate_normalized_e3m2() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e3m2() ; // 将__fp16转换为int16_t}
                }
            }
            c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            for (int i = 0; i < scale_valid_num; ++i) {
                a_scale[i] = generate_normalized_ue8m0(); // 随机生成-128到127
                b_scale[i] = generate_normalized_ue8m0(); // 随机生成-128到127
            }
        } else if(is_rand != 0){
            gen_exception_fp8(a, b, c, size, is_rand);
        } else{
            for (int i = 0; i < scale_valid_num; ++i) {
                a_scale[i] = 127; // 初始值为1
                b_scale[i] = 127; // 初始值为1
            }
            for (int i = 0; i < size; ++i){
                int temp_a = 64; // 将i转换为__fp16
                int temp_b = 64; // 将i转换为__fp16
                a[i] = *((int8_t*)&temp_a); // 将__fp16转换为int16_t
                b[i] = *((int8_t*)&temp_b); //
                printf("a[%d]: %02x, b[%d]: %02x\n", i, (uint8_t)a[i], i, (uint8_t)b[i]);
            }
            // c[0] = 0; // 初始值为0
            c[0] = 0x3f800000;
        }
        
        d[0] = mymac_scale(a, a_scale, b, b_scale, c, size, dtype);

        d_nv[0] = nvmac_scale(a, a_scale, b, b_scale, c, size, dtype);

        if (d[0] != d_nv[0]) {
            // Formatted output: hex (raw bits) + decimal (numeric value) for debugging
            printf("FP6FP4 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvfp6fp4mac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
            
            for (int i = 0; i < scale_size; ++i) {
                a_scale_to_file[scale_size - 1 - i] = a_scale[i];
                b_scale_to_file[scale_size - 1 - i] = b_scale[i];
            }
            
            df->a->write_int8_array_to_file(a, size, 1);
            df->b->write_int8_array_to_file(b, size, 1);
            df->c->write_int32_array_to_file(c, 1, 1);
            df->d->write_int32_array_to_file(d, 1, 1);
            df->a_scale->write_int8_array_to_file(a_scale_to_file, scale_size, 1);
            df->b_scale->write_int8_array_to_file(b_scale_to_file, scale_size, 1);

        } else if(print_en) {
            // Optional: log success for debugging
            printf("FP6FP4 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

        free(a_scale);
        free(b_scale);

        free(a);
        free(b);
        free(c);
        free(d);
        free(d_nv);
        
    }else if (dtype == 15 || dtype == 16 || dtype == 18) {
        int size = bitsize / 8;
        int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* b = (int8_t*)malloc(size * sizeof(int8_t));
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d_nv = (int32_t*)malloc(sizeof(int32_t));

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                if (dtype == 15)
                {
                    a[i] = generate_normalized_e2m3() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e2m3() ; // 将__fp16转换为int16_t}
                } else if(dtype == 18) {
                    a[i] = generate_normalized_e2m1() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e2m1() ; // 将__fp16转换为int16_t}
                } else {
                    a[i] = generate_normalized_e3m2() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e3m2() ; // 将__fp16转换为int16_t}
                }
            }

            if(!float_type){
                c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            }
            else {
                c[0] = (uint16_t)generate_normalized_fp16(); // 先去掉符号扩展,只保留低16位
            }
            
        } else if(is_rand != 0){
            gen_exception_fp6(a, b, c, size, is_rand);
        } else{
            for (int i = 0; i < size; ++i){
                int temp_a = 64; // 将i转换为__fp16
                int temp_b = 64; // 将i转换为__fp16
                a[i] = *((int8_t*)&temp_a); // 将__fp16转换为int16_t
                b[i] = *((int8_t*)&temp_b); //
                printf("a[%d]: %02x, b[%d]: %02x\n", i, (uint8_t)a[i], i, (uint8_t)b[i]);
            }
            // c[0] = 1.0; // 初始值为0
            if(!float_type){
                c[0] = 0x3f800000;
            }
            else {
                c[0] = 0x00003C00;
            }
        }
        d[0] = mymac(a, b, c, size, dtype, float_type);
        d_nv[0] = nvmac(a, b, c, size, dtype, float_type);

        if (d[0] != d_nv[0]) {
            // Formatted output: hex (raw bits) + decimal (numeric value) for debugging
            printf("FP6FP4 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvfp6fp4mac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
            df->a->write_int8_array_to_file(a, size, 1);
            df->b->write_int8_array_to_file(b, size, 1);
            if(!float_type){
                df->c->write_int32_array_to_file(c, 1, 1);
                df->d->write_int32_array_to_file(d, 1, 1);
            }
            else {
                int16_t c16 = (int16_t)c[0];
                int16_t d16 = (int16_t)d[0];

                df->c->write_int16_array_to_file(&c16,1,1);
                df->d->write_int16_array_to_file(&d16,1,1);
            }
        } else if(print_en) {
            // Optional: log success for debugging
            printf("FP6FP4 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

        free(a);
        free(b);
        free(c);
        free(d);
        free(d_nv);
    }
    else if (dtype == 9) {
        int size = bitsize / 8;
        //计算fp4个数，然后我们按照16个一组，来进行scale个数的计算
        int scale_size = bitsize / 4 / 16;
        // unsigned int a_bits[4] = {0x56235f15, 0x95cbbdc8, 0x1022d5d2, 0xdc3620c5};
        // unsigned int b_bits[4] = {0x70ef27d5, 0x77993919, 0x0a1b2e40, 0xdd5e2412};
        // unsigned int c_bits[1] = {0xca4df250};
        int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* b = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* a_scale = (int8_t*)malloc(2 * scale_size * sizeof(int8_t));
        int8_t* b_scale = (int8_t*)malloc(2 * scale_size * sizeof(int8_t));
        int8_t* a_scale_to_file = a_scale + scale_size;
        int8_t* b_scale_to_file = b_scale + scale_size;
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        // memcpy((void *) a, (void *)a_bits, sizeof(a_bits));
        // memcpy((void *) b, (void *)b_bits, sizeof(b_bits));
        // memcpy((void *) c, (void *)c_bits, sizeof(c));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d_nv = (int32_t*)malloc(sizeof(int32_t));


        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                a[i] = generate_normalized_2e2m1(); 
                b[i] = generate_normalized_2e2m1(); 
            }
            c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            for (int i = 0; i < scale_size; ++i) {
                a_scale[i] = generate_normalized_e4m3(); // 随机生成-128到127
                b_scale[i] = generate_normalized_e4m3(); // 随机生成-128到127
            }
        } else if(is_rand != 0){
            gen_exception_fp8(a, b, c, size, is_rand);
        } else
        {
            for (int i = 0; i < size; ++i) {
                a[i] = 0x00; 
                b[i] = 0x00; 
            }
            a[0] = 0xda; 
            b[0] = 0x19; 
            c[0] = 0x447ca75a; // 将float转换为int32_t

            for (int i = 0; i < scale_size; ++i) {
                a_scale[i] = 0x00; // 随机生成-128到127
                b_scale[i] = 0x00; // 随机生成-128到127
            }

            // a_scale[0] = 0xad; // 随机生成-128到127
            a_scale[0] = 0xb0; // 随机生成-128到127
            a_scale[1] = 0xe9; // 随机生成-128到127
            a_scale[2] = 0xb8; // 随机生成-128到127
            a_scale[3] = 0xb0; // 随机生成-128到127

            b_scale[0] = 0x1c; // 随机生成-128到127
            // b_scale[0] = 0xea; // 随机生成-128到127
            b_scale[1] = 0x81; // 随机生成-128到127
            b_scale[2] = 0xe8; // 随机生成-128到127
            b_scale[3] = 0x1c; // 随机生成-128到127
        }
        d[0] = mymac_scale(a, a_scale, b, b_scale, c, size, dtype);

        d_nv[0] = nvmac_scale(a, a_scale, b, b_scale, c, size, dtype);

        if (d[0] != d_nv[0]) {
            // Formatted output: hex (raw bits) + decimal (numeric value) for debugging
            printf("NVFP4 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA nvfp4mac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
            
            for (int i = 0; i < scale_size; ++i) {
                a_scale_to_file[scale_size - 1 - i] = a_scale[i];
                b_scale_to_file[scale_size - 1 - i] = b_scale[i];
            }
            
            df->a->write_int8_array_to_file(a, size, 1);
            df->b->write_int8_array_to_file(b, size, 1);
            df->c->write_int32_array_to_file(c, 1, 1);
            df->d->write_int32_array_to_file(d, 1, 1);
            df->a_scale->write_int8_array_to_file(a_scale_to_file, scale_size, 1);
            df->b_scale->write_int8_array_to_file(b_scale_to_file, scale_size, 1);

        } else if(print_en) {
            // Optional: log success for debugging
            printf("NVFP4 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

        free(a_scale);
        free(b_scale);

        free(a);
        free(b);
        free(c);
        free(d);
        free(d_nv);

    }else if (dtype == 10) {        // mxfp4
        int size = bitsize / 8;
        int scale_size = bitsize / 4 / 16;
        int scale_valid_num = bitsize / 4 / 32;
        int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* b = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* a_scale = (int8_t*)malloc(2 * scale_size * sizeof(int8_t));
        int8_t* b_scale = (int8_t*)malloc(2 * scale_size * sizeof(int8_t));
        int8_t* a_scale_to_file = a_scale + scale_size;
        int8_t* b_scale_to_file = b_scale + scale_size;
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d_nv = (int32_t*)malloc(sizeof(int32_t));

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                a[i] = generate_normalized_2e2m1(); 
                b[i] = generate_normalized_2e2m1(); 
            }
            c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            for (int i = 0; i < scale_valid_num; ++i) {
                a_scale[i] = generate_normalized_ue8m0(); // 随机生成-128到127
                b_scale[i] = generate_normalized_ue8m0(); // 随机生成-128到127
            }
        } else if(is_rand != 0){
            gen_exception_fp8(a, b, c, size, is_rand);
        } else
        {
            for (int i = 0; i < size; ++i) {
                a[i] = 0x00; 
                b[i] = 0x00; 
            }
            a[0] = 0x92; 
            b[0] = 0x0f; 
            c[0] = 0x59297454; // 将float转换为int32_t
            
            for (int i = 0; i < scale_valid_num; ++i) {
                a_scale[i] = 0x00; // 随机生成-128到127
                b_scale[i] = 0x00; // 随机生成-128到127
            }
        }
        d[0] = mymac_scale(a, a_scale, b, b_scale, c, size, dtype);

        d_nv[0] = nvmac_scale(a, a_scale, b, b_scale, c, size, dtype);

        if (d[0] != d_nv[0]) {
            // Formatted output: hex (raw bits) + decimal (numeric value) for debugging
            printf("MXFP4 MAC results mismatch!\n");
            printf("Custom mymac: d[0] = 0x%08x (decimal: %d)\n", d[0], d[0]);
            printf("NVIDIA mxfp4mac: d_nv[0] = 0x%08x (decimal: %d)\n", d_nv[0], d_nv[0]);
            
            for (int i = 0; i < scale_valid_num; ++i) {
                a_scale_to_file[scale_size - 1 - i] = a_scale[i];
                b_scale_to_file[scale_size - 1 - i] = b_scale[i];
            }
            
            df->a->write_int8_array_to_file(a, size, 1);
            df->b->write_int8_array_to_file(b, size, 1);
            df->c->write_int32_array_to_file(c, 1, 1);
            df->d->write_int32_array_to_file(d, 1, 1);
            df->a_scale->write_int8_array_to_file(a_scale_to_file, scale_size, 1);
            df->b_scale->write_int8_array_to_file(b_scale_to_file, scale_size, 1);

        } else if(print_en) {
            // Optional: log success for debugging
            printf("MXFP4 MAC results match: d[0] = 0x%08x\n", d[0]);
        }

        free(a_scale);
        free(b_scale);

        free(a);
        free(b);
        free(c);
        free(d);
        free(d_nv);

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
