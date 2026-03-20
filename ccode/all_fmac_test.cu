// #include "fmac.h"
// #include "FileManager.h"
// #include "myrandom.h"
#include "gendata.h"
#include <cstdint>
#include <cstring>

using namespace std;

int main(int argc, char* argv[]){
    // 示例：创建数据文件并生成数据
    int regendata = 1; // 1表示生成数据，0表示读取数据，2表示不重新生成abc，只重新生成d用于调试
    // 0: INT8,  1 : FP16, 2 : BF16, 3: TF32, 4: i8u8, 5: u8i8, 6: u8u8, 7: MXFP8e4m3（带scale）, 8:MXFP8E5M2
    // 9: NVFP4, 10: MXFP4,11: MXFP6E2M3, 12:MXFP6E3M2, 13:FP8E4M3(不带scale), 14:FP8E5M2, 15: FP6E2M3, 16:FP6E3M2
    // 17:mxFP4E2M1, 18:FP4E2M1
    int dtype = 9; 
    int is_rand = 1; // 1表示随机生成数据，0表示顺序生成数据
    int bitsize = 4*64; // 数据位数
    int total_vecnum = 1000000; // 总向量数量 1000000
    // int gen_diff = 0; // 1表示生成差异文件，0表示不生成
    bool float_type = 0; //0表示目标格式为fp32，1表示目标格式为fp16
    

    if(argc > 1){
        dtype = atoi(argv[1]);
        printf("Using dtype from arg: %d\n", dtype);
    }

    if(regendata){
        DataFile* df = NULL;
        if(dtype == 0){
            df = create_data_file("output_data/int8_a.txt", "output_data/int8_b.txt", "output_data/int8_c.txt", "output_data/int8_d.txt", dtype, 1);
        }
        else if(dtype == 1){
            df = create_data_file("output_data/fp16_a.txt", "output_data/fp16_b.txt", "output_data/fp16_c.txt", "output_data/fp16_d.txt", dtype, 1, float_type);
        }
        else if(dtype == 2){
            df = create_data_file("output_data/bf16_a.txt", "output_data/bf16_b.txt", "output_data/bf16_c.txt", "output_data/bf16_d.txt", dtype, 1);
        }
        else if(dtype == 3){
            df = create_data_file("output_data/tf32_a.txt", "output_data/tf32_b.txt", "output_data/tf32_c.txt", "output_data/tf32_d.txt", dtype, 1);
        }
        else if(dtype == 4){
            df = create_data_file("output_data/i8ui8_a.txt", "output_data/i8ui8_b.txt", "output_data/i8ui8_c.txt", "output_data/i8ui8_d.txt", dtype, 1);
        }
        else if(dtype == 5){
            df = create_data_file("output_data/ui8i8_a.txt", "output_data/ui8i8_b.txt", "output_data/ui8i8_c.txt", "output_data/ui8i8_d.txt", dtype, 1);
        }
        else if(dtype == 6){
            df = create_data_file("output_data/ui8ui8_a.txt", "output_data/ui8ui8_b.txt", "output_data/ui8ui8_c.txt", "output_data/ui8ui8_d.txt", dtype, 1);
        }
        else if(dtype == 7){
            df = create_data_file("output_data/e4m3_a.txt", "output_data/e4m3_b.txt", "output_data/e4m3_c.txt", "output_data/e4m3_d.txt", dtype, 1);
        }
        else if(dtype == 8){
            df = create_data_file("output_data/e5m2_a.txt", "output_data/e5m2_b.txt", "output_data/e5m2_c.txt", "output_data/e5m2_d.txt", dtype, 1);
        }
        else if(dtype == 9){
            df = create_data_file("output_data/nvfp4_a.txt", "output_data/nvfp4_b.txt", "output_data/nvfp4_c.txt", "output_data/nvfp4_d.txt", dtype, 1);
        }
        else if(dtype == 10){
            df = create_data_file("output_data/mxfp4_a.txt", "output_data/mxfp4_b.txt", "output_data/mxfp4_c.txt", "output_data/mxfp4_d.txt", dtype, 1);
        }
        else if(dtype == 11){
            df = create_data_file("output_data/e2m3_a.txt", "output_data/e2m3_b.txt", "output_data/e2m3_c.txt", "output_data/e2m3_d.txt", dtype, 1);
        }
        else if(dtype == 12){
            df = create_data_file("output_data/e3m2_a.txt", "output_data/e3m2_b.txt", "output_data/e3m2_c.txt", "output_data/e3m2_d.txt", dtype, 1);
        }
        else if(dtype == 13){
            df = create_data_file("output_data/fp8e4m3_a.txt", "output_data/fp8e4m3_b.txt", "output_data/fp8e4m3_c.txt", "output_data/fp8e4m3_d.txt", dtype, 1, float_type);
        }
        else if(dtype == 14){
            df = create_data_file("output_data/fp8e5m2_a.txt", "output_data/fp8e5m2_b.txt", "output_data/fp8e5m2_c.txt", "output_data/fp8e5m2_d.txt", dtype, 1, float_type);
        }
        else if(dtype == 15){
            df = create_data_file("output_data/fp6e2m3_a.txt", "output_data/fp6e2m3_b.txt", "output_data/fp6e2m3_c.txt", "output_data/fp6e2m3_d.txt", dtype, 1, float_type);
        }
        else if(dtype == 16){
            df = create_data_file("output_data/fp6e3m2_a.txt", "output_data/fp6e3m2_b.txt", "output_data/fp6e3m2_c.txt", "output_data/fp6e3m2_d.txt", dtype, 1, float_type);
        }
        else if(dtype == 17){
            df = create_data_file("output_data/e2m1_a.txt", "output_data/e2m1_b.txt", "output_data/e2m1_c.txt", "output_data/e2m1_d.txt", dtype, 1);
        }
        else if(dtype == 18){
            df = create_data_file("output_data/fp4e2m1_a.txt", "output_data/fp4e2m1_b.txt", "output_data/fp4e2m1_c.txt", "output_data/fp4e2m1_d.txt", dtype, 1, float_type);
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
        
        bool print_en = 0;

        for(int vecnum = 0; vecnum < total_vecnum; vecnum++){
            // printf("Generating data for vector %d\n", vecnum);
            int result = gen_data_file(df, bitsize, dtype, is_rand, float_type, print_en); // 生成128位数据
            if (result != 0) {
                printf("Data generation failed with error code: %d\n", result);
            }

            //每一万次打印一下
            if ((vecnum + 1) % 10000 == 0) { // 注意：vecnum从0开始，+1后对应第N次
                print_en = true;
                printf("the %d-th test: \n", vecnum + 1); // 修正：补充逗号，+1让计数从1开始更直观
            } else {
                print_en = false;
            }
        }

        // 清理资源
        delete df->a;
        delete df->b;
        delete df->c;
        delete df->d;
        free(df);
    }

    return 0;
}