#include "fmac.h"
#include "FileManager.h"
#include "myrandom.h"
#include <cstdint>
#include <cstring>

using namespace std;
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

DataFile* create_data_file(string a_fname, string b_fname, string c_fname, string d_fname, int dtype, int is_create) {
    DataFile* df = (DataFile*)malloc(sizeof(DataFile));
    if (!df) return NULL;
    if(dtype >= 7 || dtype <= 10) {
        df->a_scale = new FileIntArrayManager("scale_" + a_fname, FileIntArrayManager::INT8, is_create);
        df->b_scale = new FileIntArrayManager("scale_" + b_fname, FileIntArrayManager::INT8, is_create);
    }
    if((dtype == 0) || (dtype == 4) || (dtype == 5) || (dtype == 6) || (dtype == 7 || (dtype == 8) || (dtype == 9) || (dtype == 10) || (dtype == 11) || (dtype == 12))){
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
    df->c = new FileIntArrayManager(c_fname, FileIntArrayManager::INT32, is_create);
    df->d = new FileIntArrayManager(d_fname, FileIntArrayManager::INT32, is_create);
    return df;
}

int gen_data_file(DataFile* df, int bitsize, int dtype, int is_rand) {
    if (!df || !df->a || !df->b || !df->c || !df->d) return -1;

    if ((dtype == 7 || dtype == 8 || dtype == 9 || dtype == 10) && (!df->a_scale || !df->b_scale)) return -1;

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
        int32_t temp_zero = 0;
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

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                a[i] = generate_normalized_fp16(); // 将__fp16转换为int16_t
                b[i] = generate_normalized_fp16(); // 将__fp16转换为int16_t
            }
            c[0] = generate_normalized_fp32(); // 将float转换为int32_t
        } else if(is_rand != 0){
            gen_exception_fp16(a, b, c, size, is_rand);
        } else{
            // for (int i = 0; i < size; ++i){
            //     __fp16 temp_a = (__fp16)i; // 将i转换为__fp16
            //     __fp16 temp_b = (__fp16)i; // 将i转换为__fp16
            //     a[i] = *((int16_t*)&temp_a); // 将__fp16转换为int16_t
            //     b[i] = *((int16_t*)&temp_b); //
            // }
            // c[0] = 0; // 初始值为0
        }
        d[0] = mymac(a, b, c, size, dtype);

        df->a->write_int16_array_to_file(a, size, 1);
        df->b->write_int16_array_to_file(b, size, 1);
        df->c->write_int32_array_to_file(c, 1, 1);
        df->d->write_int32_array_to_file(d, 1, 1);

        free(a);
        free(b);
        free(c);
        free(d);
    }
    else if (dtype == 2) {
        int size = bitsize / 16; // INT16每个元素2字节，INT32每个元素4字节
        int16_t* a = (int16_t*)malloc(size * sizeof(int16_t));
        int16_t* b = (int16_t*)malloc(size * sizeof(int16_t));
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));

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
        d[0] = mymac(a, b, c, size, dtype);

        df->a->write_int16_array_to_file(a, size, 1);
        df->b->write_int16_array_to_file(b, size, 1);
        df->c->write_int32_array_to_file(c, 1, 1);
        df->d->write_int32_array_to_file(d, 1, 1);

        free(a);
        free(b);
        free(c);
        free(d);
    }
    else if (dtype == 3) {
        int size = bitsize / 32; // INT16每个元素2字节，INT32每个元素4字节
        int32_t* a = (int32_t*)malloc(size * sizeof(int32_t));
        int32_t* b = (int32_t*)malloc(size * sizeof(int32_t));
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                a[i] = generate_normalized_fp32() & 0xFFFFE000; // 将__fp16转换为int16_t
                b[i] = generate_normalized_fp32() & 0xFFFFE000; // 将__fp16转换为int16_t
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
        d[0] = mymac(a, b, c, size, dtype);

        df->a->write_int32_array_to_file(a, size, 1);
        df->b->write_int32_array_to_file(b, size, 1);
        df->c->write_int32_array_to_file(c, 1, 1);
        df->d->write_int32_array_to_file(d, 1, 1);

        free(a);
        free(b);
        free(c);
        free(d);
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
                a_scale[i] = generate_normalized_e8m0(); // 随机生成-128到127
                b_scale[i] = generate_normalized_e8m0(); // 随机生成-128到127
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

        free(a_scale);
        free(b_scale);

        free(a);
        free(b);
        free(c);
        free(d);
    }else if (dtype == 11 || dtype == 12) {
        int size = bitsize / 8;
        int8_t* a = (int8_t*)malloc(size * sizeof(int8_t));
        int8_t* b = (int8_t*)malloc(size * sizeof(int8_t));
        int32_t* c = (int32_t*)malloc(sizeof(int32_t));
        int32_t* d = (int32_t*)malloc(sizeof(int32_t));

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                if (dtype == 11)
                {
                    a[i] = generate_normalized_e4m3() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e4m3() ; // 将__fp16转换为int16_t}
                } else {
                    a[i] = generate_normalized_e5m2() ; // 将__fp16转换为int16_t
                    b[i] = generate_normalized_e5m2() ; // 将__fp16转换为int16_t}
                }
            }
            c[0] = generate_normalized_fp32(); // 将float转换为int32_t
        } else if(is_rand != 0){
            gen_exception_fp8(a, b, c, size, is_rand);
        } else{
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
        d[0] = mymac(a, b, c, size, dtype);

        df->a->write_int8_array_to_file(a, size, 1);
        df->b->write_int8_array_to_file(b, size, 1);
        df->c->write_int32_array_to_file(c, 1, 1);
        df->d->write_int32_array_to_file(d, 1, 1);

        free(a);
        free(b);
        free(c);
        free(d);
    }else if (dtype == 9) {
        int size = bitsize / 8;
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
            for (int i = 0; i < scale_size; ++i) {
                a_scale[i] = 0x38; // 初始值为1
                b_scale[i] = 0x38; // 初始值为1
            }
            for (int i = 0; i < size; ++i){
                int temp_a = 0x44; // 将i转换为__fp16
                int temp_b = 0x44; // 将i转换为__fp16
                a[i] = *((int8_t*)&temp_a); // 将__fp16转换为int16_t
                b[i] = *((int8_t*)&temp_b); //
                printf("a[%d]: %02x, b[%d]: %02x\n", i, (uint8_t)a[i], i, (uint8_t)b[i]);
            }
            // c[0] = 0; // 初始值为0
            c[0] = 0x3f800000;
        }
        d[0] = mymac_scale(a, a_scale, b, b_scale, c, size, dtype);

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

        free(a_scale);
        free(b_scale);

        free(a);
        free(b);
        free(c);
        free(d);
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

        if(is_rand == 1){
            for (int i = 0; i < size; ++i) {
                a[i] = generate_normalized_2e2m1(); 
                b[i] = generate_normalized_2e2m1(); 
            }
            c[0] = generate_normalized_fp32(); // 将float转换为int32_t
            for (int i = 0; i < scale_valid_num; ++i) {
                a_scale[i] = generate_normalized_e8m0(); // 随机生成-128到127
                b_scale[i] = generate_normalized_e8m0(); // 随机生成-128到127
            }
        } else if(is_rand != 0){
            gen_exception_fp8(a, b, c, size, is_rand);
        } else
        {
            for (int i = 0; i < scale_valid_num; ++i) {
                a_scale[i] = 127 + i; // 初始值为1
                b_scale[i] = 127 + i; // 初始值为1
            }
            for (int i = 0; i < size; ++i){
                int temp_a = 0x44; // 全部为2
                int temp_b = 0x44; // 全部为2
                a[i] = *((int8_t*)&temp_a); // 将__fp16转换为int16_t
                b[i] = *((int8_t*)&temp_b); //
                printf("a[%d]: %02x, b[%d]: %02x\n", i, (uint8_t)a[i], i, (uint8_t)b[i]);
            }
            // 初始值为1
            c[0] = 0x3f800000;
        }
        d[0] = mymac_scale(a, a_scale, b, b_scale, c, size, dtype);

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

        free(a_scale);
        free(b_scale);

        free(a);
        free(b);
        free(c);
        free(d);
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

int main(int argc, char* argv[]){
    // 示例：创建数据文件并生成数据
    int regendata = 1; // 1表示生成数据，0表示读取数据，2表示不重新生成abc，只重新生成d用于调试
    int dtype = 3; // 0: INT8, 1: FP16, 2: BF16, 3: TF32, 4: i8u8, 5: u8i8, 6: u8u8, 7: MXFP8e4m3, 9:NVFP4
    int is_rand = 1; // 1表示随机生成数据，0表示顺序生成数据
    int bitsize = 512; // 数据位数
    int total_vecnum = 10000; // 总向量数量 1000000
    int gen_diff = 0; // 1表示生成差异文件，0表示不生成
    

    if(argc > 1){
        dtype = atoi(argv[1]);
        printf("Using dtype from arg: %d\n", dtype);
    }

    if(regendata){
        DataFile* df = NULL;
        if(dtype == 0){
            df = create_data_file("int8_a.txt", "int8_b.txt", "int8_c.txt", "int8_d.txt", dtype, 1);
        }
        else if(dtype == 1){
            df = create_data_file("fp16_a.txt", "fp16_b.txt", "fp16_c.txt", "fp16_d.txt", dtype, 1);
        }
        else if(dtype == 2){
            df = create_data_file("bf16_a.txt", "bf16_b.txt", "bf16_c.txt", "bf16_d.txt", dtype, 1);
        }
        else if(dtype == 3){
            df = create_data_file("tf32_a.txt", "tf32_b.txt", "tf32_c.txt", "tf32_d.txt", dtype, 1);
        }
        else if(dtype == 4){
            df = create_data_file("i8ui8_a.txt", "i8ui8_b.txt", "i8ui8_c.txt", "i8ui8_d.txt", dtype, 1);
        }
        else if(dtype == 5){
            df = create_data_file("ui8i8_a.txt", "ui8i8_b.txt", "ui8i8_c.txt", "ui8i8_d.txt", dtype, 1);
        }
        else if(dtype == 6){
            df = create_data_file("ui8ui8_a.txt", "ui8ui8_b.txt", "ui8ui8_c.txt", "ui8ui8_d.txt", dtype, 1);
        }
        else if(dtype == 7){
            df = create_data_file("e4m3_a.txt", "e4m3_b.txt", "e4m3_c.txt", "e4m3_d.txt", dtype, 1);
        }
        else if(dtype == 8){
            df = create_data_file("e5m2_a.txt", "e5m2_b.txt", "e5m2_c.txt", "e5m2_d.txt", dtype, 1);
        }
        else if(dtype == 9){
            df = create_data_file("nvfp4_a.txt", "nvfp4_b.txt", "nvfp4_c.txt", "nvfp4_d.txt", dtype, 1);
        }
        else if(dtype == 10){
            df = create_data_file("mxfp4_a.txt", "mxfp4_b.txt", "mxfp4_c.txt", "mxfp4_d.txt", dtype, 1);
        }
        else if(dtype == 11){
            df = create_data_file("fp8e4m3_a.txt", "fp8e4m3_b.txt", "fp8e4m3_c.txt", "fp8e4m3_d.txt", dtype, 1);
        }
        else if(dtype == 12){
            df = create_data_file("fp8e5m2_a.txt", "fp8e5m2_b.txt", "fp8e5m2_c.txt", "fp8e5m2_d.txt", dtype, 1);
        }
        else{
            printf("Invalid dtype: %d\n", dtype);
            return -1;
        }

         // 检查DataFile是否创建成功
        if (!df) {
            printf("Failed to create DataFile\n");
            return -1;
        }
        
        for(int vecnum = 0; vecnum < total_vecnum; vecnum++){
            // printf("Generating data for vector %d\n", vecnum);
            int result = gen_data_file(df, bitsize, dtype, is_rand); // 生成128位数据
            if (result != 0) {
                printf("Data generation failed with error code: %d\n", result);

            }
        }

        // 清理资源
        delete df->a;
        delete df->b;
        delete df->c;
        delete df->d;
        free(df);
    }
    // DataFile* df2 = NULL;
    //    if(dtype == 0){
    //         df2 = create_data_file("int8_a.txt", "int8_b.txt", "int8_c.txt", "int8_d.txt", dtype, 0);
    //     }
    //     else if(dtype == 1){
    //         df2 = create_data_file("fp16_a.txt", "fp16_b.txt", "fp16_c.txt", "fp16_d.txt", dtype, 0);
    //     }
    //     else if(dtype == 2){
    //         df2 = create_data_file("bf16_a.txt", "bf16_b.txt", "bf16_c.txt", "bf16_d.txt", dtype, 0);
    //     }
    //     else{
    //         df2 = create_data_file("tf32_a.txt", "tf32_b.txt", "tf32_c.txt", "tf32_d.txt", dtype, 0);
    //     }
    // if (!df2) {
    //     printf("Failed to create DataFile for reading\n");
    //     return -1;
    // }

    // int error_number = 0;
    // int exception_number = 0;
    // for(int vecnum = 0; vecnum < total_vecnum; vecnum++){
    //     int check = check_data_file(df2, bitsize, dtype, gen_diff); // 检查数据文件
    //     if(check == 1){
    //         printf("file is empty or not exist\n");
    //     }
    //     else if(check == 2){
    //         printf("[%d]: Data check failed, results differ significantly.\n", vecnum);
    //         // gen_data_file(df2, bitsize, dtype, is_rand, 1); // 重新生成数据
    //         printf("\n");
    //         error_number++;
    //     }
    //     else if(check == 3)
    //         exception_number++;
    // }

    // printf("Total errors: %d out of %d\n", error_number, total_vecnum);
    // printf("Total exceptions: %d out of %d\n", exception_number, total_vecnum);

    // delete df2->a;
    // delete df2->b;
    // delete df2->c;
    // delete df2->d;
    // free(df2);
    return 0;
}