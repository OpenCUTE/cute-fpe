#include <verilated.h>
#include <Vtop.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include "FloatDecode.h"
#include "FileManager.h"

using namespace std;
//mode 0是从头读/写，1是从上次读写位置继续读/写
// #define FM_DEBUG


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
    if(dtype == 7 || dtype == 8 || dtype == 9 || dtype == 10) {
        df->a_scale = new FileIntArrayManager("scale_" + a_fname, FileIntArrayManager::INT8, is_create);
        df->b_scale = new FileIntArrayManager("scale_" + b_fname, FileIntArrayManager::INT8, is_create);
    }
    if((dtype == 0) || (dtype == 4) || (dtype == 5) || (dtype == 6) || (dtype == 7)){
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

int create_all_types(int dtype, DataFile** df){
    if (dtype == 0){
        *df = create_data_file("int8_a.txt", "int8_b.txt", "int8_c.txt", "int8_d.txt", dtype, 0);
    }
    else if (dtype == 1){
        *df = create_data_file("fp16_a.txt", "fp16_b.txt", "fp16_c.txt", "fp16_d.txt", dtype, 0);
        // *df = create_data_file("fp16_atest.txt", "fp16_btest.txt", "fp16_ctest.txt", "fp16_dtest.txt", dtype, 0);
    }
    else if (dtype == 2){
        *df = create_data_file("bf16_a.txt", "bf16_b.txt", "bf16_c.txt", "bf16_d.txt", dtype, 0);
    }
    else if (dtype == 3){
        *df = create_data_file("tf32_a.txt", "tf32_b.txt", "tf32_c.txt", "tf32_d.txt", dtype, 0);
    }
    else if (dtype == 4){
        *df = create_data_file("i8ui8_a.txt", "i8ui8_b.txt", "i8ui8_c.txt", "i8ui8_d.txt", dtype, 0);
    }
    else if (dtype == 5){
        *df = create_data_file("ui8i8_a.txt", "ui8i8_b.txt", "ui8i8_c.txt", "ui8i8_d.txt", dtype, 0);
    }
    else if (dtype == 6){
        *df = create_data_file("ui8ui8_a.txt", "ui8ui8_b.txt", "ui8ui8_c.txt", "ui8ui8_d.txt", dtype, 0);
    }
    else if (dtype == 7){
        *df = create_data_file("e4m3_a.txt", "e4m3_b.txt", "e4m3_c.txt", "e4m3_d.txt", dtype, 0);
    }
    else if (dtype == 8){
        *df = create_data_file("e5m2_a.txt", "e5m2_b.txt", "e5m2_c.txt", "e5m2_d.txt", dtype, 0);
    }
    else if (dtype == 9){
        *df = create_data_file("nvfp4_a.txt", "nvfp4_b.txt", "nvfp4_c.txt", "nvfp4_d.txt", dtype, 0);
    }
    else if(dtype == 10){
        *df = create_data_file("mxfp4_a.txt", "mxfp4_b.txt", "mxfp4_c.txt", "mxfp4_d.txt", dtype, 0);
    }
    else if(dtype == 11){
        *df = create_data_file("fp8e4m3_a.txt", "fp8e4m3_b.txt", "fp8e4m3_c.txt", "fp8e4m3_d.txt", dtype, 0);
    }
    else if(dtype == 12){
        *df = create_data_file("fp8e5m2_a.txt", "fp8e5m2_b.txt", "fp8e5m2_c.txt", "fp8e5m2_d.txt", dtype, 0);
    }
    else{
        printf("Invalid dtype: %d\n", dtype);
        return -1;
    }
    return 0;
}

int check_sim(DataFile* df, int bitsize, int dtype, Vtop* top, int cycles, int tolerance){
    int result = 0;
    if (!df || !df->a || !df->b || !df->c || !df->d) {
        return 1;
    }
    if (bitsize != 256 && bitsize != 512) {
        printf("Unsupported bitsize: %d\n", bitsize);
        return 1;
    }
    int exception_error = 0;
    int data_error = 0;

    int size= bitsize / 32;
    int scale_size = bitsize / 4 / 16 * 8 / 32;

    int32_t* a = (int32_t*)malloc(size * sizeof(int32_t));
    int32_t* b = (int32_t*)malloc(size * sizeof(int32_t));
    int32_t* a_scale = (int32_t*)malloc(scale_size * sizeof(int32_t));
    int32_t* b_scale = (int32_t*)malloc(scale_size * sizeof(int32_t));
    int32_t* c = (int32_t*)malloc(sizeof(int32_t));
    int32_t* d = (int32_t*)malloc(sizeof(int32_t));

    char* filename = "error.txt";
    FILE* fp = fopen(filename, "w");

    //循环部分
    printf("cycles : %d\n", cycles);
    for(int j = 0; j < cycles; j ++){

        uint32_t *a_vec = top->io_AVector_bits.data();
        uint32_t *b_vec = top->io_BVector_bits.data();
        // uint32_t *a_scale_vec = top->io_AScale_bits.data();
        // uint32_t *b_scale_vec = top->io_BScale_bits.data();

        df->a->read_int32_array_from_file(a, size, 1);
        df->b->read_int32_array_from_file(b, size, 1);
        if (df->a_scale && df->b_scale) {
            df->a_scale->read_int32_array_from_file((int32_t *)a_scale, scale_size, 1);
            df->b_scale->read_int32_array_from_file((int32_t *)b_scale, scale_size, 1);
        }
        // printf("a_scale[0]: %08x, b_scale[0]: %08x\n", a_scale[0], b_scale[0]);
        df->c->read_int32_array_from_file(c, 1, 1);

        if (bitsize == 512) {
            top->io_AScale_bits = (long unsigned int) a_scale[0] << 32 | (a_scale[1] & 0xFFFFFFFF);
            // printf("AScale bits: %lx\n", top->io_AScale_bits);
            top->io_BScale_bits = (long unsigned int) b_scale[0] << 32 | (b_scale[1] & 0xFFFFFFFF);
        }
        else {
            top->io_AScale_bits = *(uint32_t *)a_scale;
            top->io_BScale_bits = *(uint32_t *)b_scale;
        }
        top->clock = 1;
        top->reset = 0;
        top->io_AVector_valid = 1;
        top->io_BVector_valid = 1;
        top->io_AScale_valid = 1;
        top->io_BScale_valid = 1;
        top->io_CAdd_valid = 1;
        top->io_DResult_ready = 1;
        top->io_opcode = dtype;

        // top->io_AVector_bits = *(uint32_t *)a;
        // top->io_BVector_bits = *(uint32_t *)b;


        // top->io_AVector_bits = *(uint64_t *)a;
        // top->io_BVector_bits = *(uint64_t *)b;

        for(int i = 0; i < size; i ++){
            a_vec[i] = ((uint32_t *) a)[i];
            b_vec[i] = ((uint32_t *) b)[i];
        }

        // for(int i = 0; i < scale_size; i ++){
        //     a_scale_vec[i] = ((uint32_t *) a_scale)[i];
        //     b_scale_vec[i] = ((uint32_t *) b_scale)[i];
        // }


        top->io_CAdd_bits = *(uint32_t *)c;
        top->eval();
        top->clock = 0;
        top->eval();

        uint32_t DRes = top->io_DResult_bits;
        if(top->io_DResult_valid == 1){
            df->d->read_int32_array_from_file(d, 1, 1);
            uint32_t golden_result = *(uint32_t *)d;
            // printf("Cycle %d: DResult = %08x\n", j, DRes);
            // printf("Cycle %d: golden  = %08x\n", j, golden_result);
            float FDRes = *(float *)&DRes;
            float FGolden = *(float *)&golden_result;
            int32_t FDResExcept = get_exceptioncode(FDRes);
            int32_t FGoldenExcept = get_exceptioncode(FGolden);
            if(FDResExcept != FGoldenExcept){
                exception_error ++;
                printf("Cycle %d: Mismatch! Expected: %x, Got: %x\n", j, golden_result, DRes);
                // printf("golden:%a\n", *(float*)&golden_result);
                // printf("result:%a\n", *(float*)&(DRes));
                // printf("FDResExcept: %d\n", FDResExcept);
                // printf("FGoldenExcept: %d\n", FGoldenExcept);
                // int16_t * fp16_a = (int16_t *)a;
                // int16_t * fp16_b = (int16_t *)b;
                // FloatDecode a0 = decode_fp16(fp16_a[0]);
                // FloatDecode a1 = decode_fp16(fp16_a[1]);
                // FloatDecode b0 = decode_fp16(fp16_b[0]);
                // FloatDecode b1 = decode_fp16(fp16_b[1]);
                // FloatDecode cdecode = decode_fp32(c[0]);
                // printf("product0 exp: %x\n", a0.exponent + b0.exponent);
                // printf("product1 exp: %x\n", a1.exponent + b1.exponent);
                // printf("product1 exp: %x\n", a1.exponent);
                // printf("product1 exp: %x\n", b1.exponent);
                // printf("c exp : %x\n", cdecode.exponent);
                // printf("product0 exp: %x\n", a0.exponent + b0.exponent + 255);
                // printf("product1 exp: %x\n", a1.exponent + b1.exponent + 255);
                // printf("c exp : %x\n", cdecode.exponent + 255);
                // printf("product0 exp: %d\n", a0.exponent + b0.exponent);
                // printf("product1 exp: %d\n", a1.exponent + b1.exponent);
                // printf("c exp : %d\n", cdecode.exponent);
                // free(a);
                // free(b);
                // free(c);
                // free(d);
                // return result;
            }
            else if(((int32_t)DRes - (int32_t)golden_result > tolerance) || ((int32_t)DRes - (int32_t)golden_result < -tolerance)){
                data_error ++;

                int8_t * fp8_a = (int8_t *)a;
                int8_t * fp8_b = (int8_t *)b;
                printf("fp8_a\n");
                for (int i = 0; i < bitsize / 8; i++) {
                    printf("%02x", fp8_a[i] & 0xFF);
                }
                printf("\n");
                for (int i = 0; i < bitsize / 8; i++) {
                    printf("%02x", fp8_b[i] & 0xFF);
                    
                }
                printf("\n");
                printf("c:%x\n", c[0]);
                printf("golden:%x\n", golden_result);
                printf("result:%x\n", DRes);
                // int16_t * fp16_a = (int16_t *)a;
                // int16_t * fp16_b = (int16_t *)b;
                // FloatDecode a0 = decode_fp16(fp16_a[0]);
                // FloatDecode a1 = decode_fp16(fp16_a[1]);
                // FloatDecode b0 = decode_fp16(fp16_b[0]);
                // FloatDecode b1 = decode_fp16(fp16_b[1]);
                // FloatDecode cdecode = decode_fp32(c[0]);
                // int32_t product0_sign = (a0.sign ^ b0.sign)? -1 : 1;
                // int32_t product1_sign = (a1.sign ^ b1.sign)? -1 : 1;
                // int32_t c_sign = cdecode.sign ? -1 : 1;
                // printf("%04x", fp16_a[0] & 0xFFFF);
                // printf("%04x\n", fp16_a[1]& 0xFFFF);
                // printf("%04x", fp16_b[0]& 0xFFFF);
                // printf("%04x\n", fp16_b[1]& 0xFFFF);
                // printf("%08x\n", c[0]);
                // printf("%08x\n", d[0]);
                // printf("DRes - golden = %d\n", (int32_t)DRes - (int32_t)golden_result);
                printf("Cycle %d: Mismatch! Expected: %x, Got: %x\n", j, golden_result, DRes);
                // printf("golden:%a\n", *(float*)&golden_result);
                // printf("result:%a\n", *(float*)&(DRes));

                // printf("golden float:%f\n", *(float*)&golden_result);
                // printf("result float:%f\n", *(float*)&(DRes));
                // printf("a0 sig: %x\n", a0.mantissa);
                // printf("a1 sig: %x\n", a1.mantissa);
                // printf("b0 sig: %x\n", b0.mantissa);
                // printf("b1 sig: %x\n", b1.mantissa);
                // printf("c  sig: %x\n", cdecode.mantissa);
                // printf("product0 sig: %x\n", a0.mantissa * b0.mantissa * product0_sign);
                // printf("product1 sig: %x\n", a1.mantissa * b1.mantissa * product1_sign);
                // printf("c sig : %x\n", cdecode.mantissa * c_sign);
                // printf("product0 exp: %x\n", a0.exponent + b0.exponent);
                // printf("product1 exp: %x\n", a1.exponent + b1.exponent);
                // printf("c exp : %x\n", cdecode.exponent);
                // // printf("product0 exp: %x\n", a0.exponent + b0.exponent + 255);
                // // printf("product1 exp: %x\n", a1.exponent + b1.exponent + 255);
                // // printf("c exp : %x\n", cdecode.exponent + 255);
                // printf("product0 exp: %d\n", a0.exponent + b0.exponent);
                // printf("product1 exp: %d\n", a1.exponent + b1.exponent);
                // printf("c exp : %d\n", cdecode.exponent);

                // int32_t * tf32_a = (int32_t *)a;
                // int32_t * tf32_b = (int32_t *)b;
                // FloatDecode a0 = decode_tf32(tf32_a[0]);
                // FloatDecode a1 = decode_tf32(tf32_a[1]);
                // FloatDecode b0 = decode_tf32(tf32_b[0]);
                // FloatDecode b1 = decode_tf32(tf32_b[1]);
                // FloatDecode cdecode = decode_fp32(c[0]);
                // int32_t product0_sign = (a0.sign ^ b0.sign)? -1 : 1;
                // int32_t product1_sign = (a1.sign ^ b1.sign)? -1 : 1;
                // int32_t c_sign = cdecode.sign ? -1 : 1;
                // printf("%04x", tf32_a[0]);
                // printf("%04x\n", tf32_a[1]);
                // printf("%04x", tf32_b[0]);
                // printf("%04x\n", tf32_b[1]);
                // printf("%08x\n", c[0]);
                // printf("%08x\n", d[0]);
                // printf("DRes - golden = %d\n", (int32_t)DRes - (int32_t)golden_result);
                // printf("Cycle %d: Mismatch! Expected: %x, Got: %x\n", j, golden_result, DRes);
                // printf("golden:%a\n", *(float*)&golden_result);
                // printf("result:%a\n", *(float*)&(DRes));

                // printf("golden float:%f\n", *(float*)&golden_result);
                // printf("result float:%f\n", *(float*)&(DRes));
                // printf("a0 sig: %x\n", a0.mantissa);
                // printf("a1 sig: %x\n", a1.mantissa);
                // printf("b0 sig: %x\n", b0.mantissa);
                // printf("b1 sig: %x\n", b1.mantissa);
                // printf("c  sig: %x\n", cdecode.mantissa);
                // printf("product0 sig: %x\n", a0.mantissa * b0.mantissa * product0_sign);
                // printf("product1 sig: %x\n", a1.mantissa * b1.mantissa * product1_sign);
                // printf("c sig : %x\n", cdecode.mantissa * c_sign);
                // printf("product0 exp: %x\n", a0.exponent + b0.exponent);
                // printf("product1 exp: %x\n", a1.exponent + b1.exponent);
                // printf("c exp : %x\n", cdecode.exponent);
                // // printf("product0 exp: %x\n", a0.exponent + b0.exponent + 255);
                // // printf("product1 exp: %x\n", a1.exponent + b1.exponent + 255);
                // // printf("c exp : %x\n", cdecode.exponent + 255);
                // printf("product0 exp: %d\n", a0.exponent + b0.exponent);
                // printf("product1 exp: %d\n", a1.exponent + b1.exponent);
                // printf("c exp : %d\n", cdecode.exponent);

                // free(a);
                // free(b);
                // free(c);
                // free(d);
                // return result;
            } else {
                result ++;
            }
        }else
            printf("Cycle %d: Result is Invalid\n",j);
    }
    if (fp) fclose(fp);
    printf("exc : %d\n", exception_error);
    printf("data : %d\n", data_error);
    free(a);
    free(b);
    free(c);
    free(d);
    return result;
}

int main(int argc, char* argv[]){
    int dtype = 9;
    int cycles = 10005; // 设置模拟周期数 1000000
    int bitsize = 512;
    int tolerance = 1;

    if(argc > 1){
        dtype = atoi(argv[1]);
        printf("Using dtype from arg: %d\n", dtype);
    }

    DataFile* df = NULL;
    
    int create = create_all_types(dtype, &df);
    if(df == NULL)
        printf("df NULL\n");
    Vtop* top = new Vtop;
    top->reset = 1; // 设置复位信号
    top->clock = 1;
    top->eval(); // 评估初始状态
    top->clock = 0; // 释放复位信号
    top->eval(); // 评估复位后的状态
    int result = check_sim(df, bitsize, dtype, top, cycles, tolerance);
    printf("correct: %d / %d\n", result, cycles);
    if(result == cycles)
        printf("Pass!!!\n");
    return 0;
}
