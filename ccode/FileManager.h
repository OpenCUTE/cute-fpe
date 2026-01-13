#include <stdio.h>
#include <cstdint>
#include <string>

using namespace std;
//mode 0是从头读/写，1是从上次读写位置继续读/写
// #define FM_DEBUG

class FileIntArrayManager {
public:
    enum DataType { INT32, INT16, INT8 };

    char filename[256] = {0}; // 公共文件名属性

private:
    FILE* fp_read = nullptr; // 用于跨多次读取时保存文件指针
    DataType dtype;          // 控制读写的数据类型

public:
    // 构造函数：设置文件名、数据类型，并创建空文件
    FileIntArrayManager(string fname, DataType t = INT32, int create = 0) : dtype(t) {
        // printf("%s\n", fname.c_str());
        snprintf(filename, sizeof(filename), "%s", fname.c_str());
        FILE* fp = NULL;
        if(create){
            FILE* fp = fopen(filename, "w");
        }
        else{
            FILE* fp = fopen(filename, "r");
        }
        
        if (fp) fclose(fp);
    }

    // 写int32数组
    void write_int32_array_to_file(int32_t* arr, int size, int mode) {
        if (filename[0] == '\0' || dtype != INT32) return;
        const char* fmode = (mode == 1) ? "a" : "w";
        FILE* fp = fopen(filename, fmode);
        if (!fp) {
            perror("fopen failed");
            return;
        }
        for (int i = 0; i < size; ++i) {
            fprintf(fp, "%08x", arr[i]);
        }
        fclose(fp);
    }

    // 写int16数组
    void write_int16_array_to_file(int16_t* arr, int size, int mode) {
        if (filename[0] == '\0' || dtype != INT16) return;
        const char* fmode = (mode == 1) ? "a" : "w";
        FILE* fp = fopen(filename, fmode);
        if (!fp) {
            perror("fopen failed");
            return;
        }
        for (int i = 0; i < size; ++i) {
            fprintf(fp, "%04x", (uint16_t)arr[i]);
        }
        fclose(fp);
    }

    // 写int8数组
    void write_int8_array_to_file(int8_t* arr, int size, int mode) {
        if (filename[0] == '\0' || dtype != INT8) return;
        const char* fmode = (mode == 1) ? "a" : "w";
        FILE* fp = fopen(filename, fmode);
        if (!fp) {
            perror("fopen failed");
            return;
        }
        for (int i = 0; i < size; ++i) {
            fprintf(fp, "%02x", (uint8_t)arr[i]);
        }
        fclose(fp);
    }

    // 读int32数组
    int read_int32_array_from_file(int* arr, int max_size, int mode) {
        if (filename[0] == '\0'/* || dtype != INT32*/) return -1;
        if (mode == 0) {
            if (fp_read) fclose(fp_read);
            fp_read = fopen(filename, "r");
        } else if (!fp_read) {
            fp_read = fopen(filename, "r");
        }
        if (!fp_read) {
            perror("fopen failed");
            return -1;
        }
        int count = 0;
        char buf[9];
        buf[8] = '\0';
        while (count < max_size && fread(buf, 1, 8, fp_read) == 8) {
            sscanf(buf, "%x", &arr[count]);
            float temp_float = *((float*)&arr[count]);
            #ifdef FM_DEBUG
                printf("count:%d read fp32: %#.6a\t, float: %e as float:%f\n", count, temp_float, temp_float, temp_float);
            #endif
            count++;
        }
        if (feof(fp_read)) {
            fclose(fp_read);
            fp_read = NULL;
        }
        return count;
    }

    // 读int16数组
    int read_int16_array_from_file(int16_t* arr, int max_size, int mode) {
        int check_print_mode = 1;
        if (filename[0] == '\0' || dtype != INT16) return -1;
        if (mode == 0) {
            if (fp_read) fclose(fp_read);
            fp_read = fopen(filename, "r");
        } else if (!fp_read) {
            fp_read = fopen(filename, "r");
        }
        if (!fp_read) {
            perror("fopen failed");
            return -1;
        }
        int count = 0;
        char buf[5];
        buf[4] = '\0';
        while (count < max_size && fread(buf, 1, 4, fp_read) == 4) {
            unsigned int tmp;
            sscanf(buf, "%x", &tmp);
            arr[count] = (int16_t)tmp;
            if(check_print_mode == 1){
                // __fp16 temp_fp16 = *((__fp16*)&arr[count]);
                // float temp_float = (float)temp_fp16;
                // #ifdef FM_DEBUG
                //     // printf("count:%d read fp16: %04x, float: %f\n", count, arr[count], temp_float);
                //     printf("count:%d read fp16: %#.3a\t, float: %e as float:%f\n", count, temp_float, temp_float, temp_float);
                // #endif
            }
            else if(check_print_mode == 2){
                int32_t temp_int32 = ((int32_t)arr[count]) << 16;
                float temp_float = *((float*)&temp_int32);
                #ifdef FM_DEBUG
                    // printf("count:%d read bf16: %04x, float: %f\n", count, arr[count], temp_float);
                    printf("count:%d read bf16: %#.2a\t, float: %e as float:%f\n", count, temp_float, temp_float, temp_float);
                #endif
            }

            count++;
        }
        if (feof(fp_read)) {
            fclose(fp_read);
            fp_read = NULL;
        }
        return count;
    }

    // 读int8数组
    int read_int8_array_from_file(int8_t* arr, int max_size, int mode) {
        if (filename[0] == '\0' || dtype != INT8) return -1;
        if (mode == 0) {
            if (fp_read) fclose(fp_read);
            fp_read = fopen(filename, "r");
        } else if (!fp_read) {
            fp_read = fopen(filename, "r");
        }
        if (!fp_read) {
            perror("fopen failed");
            return -1;
        }
        int count = 0;
        char buf[3];
        buf[2] = '\0';
        while (count < max_size && fread(buf, 1, 2, fp_read) == 2) {
            unsigned int tmp;
            sscanf(buf, "%x", &tmp);
            arr[count] = (int8_t)tmp;
            count++;
        }
        if (feof(fp_read)) {
            fclose(fp_read);
            fp_read = NULL;
        }
        return count;
    }

    // 析构时关闭文件指针
    ~FileIntArrayManager() {
        if (fp_read) fclose(fp_read);
    }
};