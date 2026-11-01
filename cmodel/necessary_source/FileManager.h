#pragma once

#include <stdio.h>
#include <cstdint>
#include <string>
#include <cstring>
#include <bitset>
#include <unistd.h>
#include "FloatDecode.h"

using namespace std;
//mode 0是从头读/写，1是从上次读写位置继续读/写
// #define FM_DEBUG

#ifndef FMAC_DUMP_OUTPUT
#define FMAC_DUMP_OUTPUT 1
#endif

#ifndef FMAC_DUMP_ALL
#define FMAC_DUMP_ALL 0
#endif

static inline bool fmac_dump_output_enabled() {
    return FMAC_DUMP_OUTPUT != 0;
}

static inline bool fmac_dump_all_enabled() {
    return FMAC_DUMP_ALL != 0;
}

class FileIntArrayManager {
public:
    enum DataType { INT32, INT16, INT8 };
    enum DumpMask { DUMP_TEXT = 1, DUMP_RAWHEX = 2, DUMP_BIN = 4 };

    char filename[256] = {0}; // 公共文件名属性
    char filename_rawhex[256] = {0}; // 用于存储原始十六进制数据、指数、尾数、数据类型的文件
    char filename_double_bin[256] = {0}; // 用于存储double二进制数据的文件
    
    FloatDecode res; // 存储输入值的浮点解码结果

private:
    FILE* fp_read = nullptr; // 用于跨多次读取时保存文件指针
    FILE* fp_write = nullptr;
    FILE* fp_write_rawhex = nullptr;
    FILE* fp_write_double_bin = nullptr;
    DataType dtype;          // 控制读写的数据类型
    int dump_mask;

    bool dump_text_enabled() const { return fmac_dump_all_enabled() || (dump_mask & DUMP_TEXT); }
    bool dump_rawhex_enabled() const { return fmac_dump_all_enabled() || (dump_mask & DUMP_RAWHEX); }
    bool dump_bin_enabled() const { return fmac_dump_all_enabled() || (dump_mask & DUMP_BIN); }

    void close_write_handles() {
        if (fp_write) {
            fclose(fp_write);
            fp_write = nullptr;
        }
        if (fp_write_rawhex) {
            fclose(fp_write_rawhex);
            fp_write_rawhex = nullptr;
        }
        if (fp_write_double_bin) {
            fclose(fp_write_double_bin);
            fp_write_double_bin = nullptr;
        }
    }

    bool ensure_write_handles(int mode) {
        if (mode == 0) {
            close_write_handles();
        } else if ((!dump_text_enabled() || fp_write) &&
                   (!dump_rawhex_enabled() || fp_write_rawhex) &&
                   (!dump_bin_enabled() || fp_write_double_bin)) {
            return true;
        }

        const char* fmode_hex = (mode == 1) ? "a" : "w";
        const char* fmode_bin = (mode == 1) ? "ab" : "wb";
        if (dump_text_enabled()) fp_write = fopen(filename, fmode_hex);
        if (dump_rawhex_enabled()) fp_write_rawhex = fopen(filename_rawhex, fmode_hex);
        if (dump_bin_enabled()) fp_write_double_bin = fopen(filename_double_bin, fmode_bin);
        if ((dump_text_enabled() && !fp_write) ||
            (dump_rawhex_enabled() && !fp_write_rawhex) ||
            (dump_bin_enabled() && !fp_write_double_bin)) {
            perror("fopen failed");
            close_write_handles();
            return false;
        }

        if (fp_write) setvbuf(fp_write, NULL, _IOFBF, 1 << 20);
        if (fp_write_rawhex) setvbuf(fp_write_rawhex, NULL, _IOFBF, 1 << 20);
        if (fp_write_double_bin) setvbuf(fp_write_double_bin, NULL, _IOFBF, 1 << 20);
        return true;
    }

public:
    // 构造函数：设置文件名、数据类型，并创建空文件
    FileIntArrayManager(
        string fname,
        DataType t = INT32,
        int create = 0,
        int output_mask = DUMP_TEXT | DUMP_RAWHEX | DUMP_BIN)
        : dtype(t), dump_mask(output_mask) {
        snprintf(filename, sizeof(filename), "%s.txt.log", fname.c_str());
        snprintf(filename_rawhex, sizeof(filename_rawhex), "%s_rawhex.txt.log", fname.c_str());
        snprintf(filename_double_bin, sizeof(filename_double_bin), "%s_double.bin.log", fname.c_str());
        DecodeExceptionInit(&res.exception);
        if (create && !fmac_dump_output_enabled()) {
            return;
        }
        char legacy_filename_double_bin[256] = {0};
        snprintf(legacy_filename_double_bin, sizeof(legacy_filename_double_bin),
                 "%s_double_bin.txt.log", fname.c_str());

        FILE* fp = NULL;
        FILE* fp_rawhex = NULL;
        FILE* fp_double_bin = NULL;
        if(create){
            if (dump_text_enabled()) fp = fopen(filename, "w");
            if (dump_rawhex_enabled()) fp_rawhex = fopen(filename_rawhex, "w");
            if (dump_bin_enabled()) fp_double_bin = fopen(filename_double_bin, "wb");
        }
        else{
            fp = fopen(filename, "r");
            fp_rawhex = fopen(filename_rawhex, "r");
            fp_double_bin = fopen(filename_double_bin, "rb");
            if (!fp_double_bin && access(legacy_filename_double_bin, F_OK) == 0) {
                snprintf(filename_double_bin, sizeof(filename_double_bin), "%s",
                         legacy_filename_double_bin);
                fp_double_bin = fopen(filename_double_bin, "rb");
            }
        }
        
        if (fp) fclose(fp);
        if (fp_rawhex) fclose(fp_rawhex);
        if (fp_double_bin) fclose(fp_double_bin);
    }

    // 写int32数组
    void write_int32_array_to_file(int32_t* arr, int size, int mode, int data_type) {
        if (!fmac_dump_output_enabled()) return;
        if (filename[0] == '\0' || dtype != INT32) return;
        uint32_t b32_big_endian;

        if (!ensure_write_handles(mode)) {
            return;
        }
        bool want_text = dump_text_enabled();
        bool want_rawhex = dump_rawhex_enabled();
        bool want_bin = dump_bin_enabled();
        for (int i = 0; i < size; ++i) {
            string res_bit = bitset<32>(arr[i]).to_string();
            b32_big_endian = __builtin_bswap32(arr[i]);
            if (want_text) fprintf(fp_write, "%s\n", res_bit.c_str());
            if (want_rawhex && data_type == 8) {
                fprintf(fp_write_rawhex, "%08x %s int:%d uint:%u\n",
                        (uint32_t)arr[i],
                        res_bit.c_str(),
                        arr[i],
                        (uint32_t)arr[i]);
            } else if (want_rawhex) {
                res = decode(arr[i], data_type);
                string res_mant_bit = bitset<24>(res.mantissa).to_string();
                fprintf(fp_write_rawhex, "%08x %s exp:%04d mant:%s/%06x %s\n", arr[i], res_bit.c_str(), res.exponent, res_mant_bit.c_str(), (uint32_t)res.mantissa, exception_flag(&res.exception).c_str());
            }
            if (want_bin) fwrite (&b32_big_endian, sizeof(uint32_t), 1, fp_write_double_bin);
        }
    }

    // 写int16数组
    void write_int16_array_to_file(int16_t* arr, int size, int mode, int data_type) {
        if (!fmac_dump_output_enabled()) return;
        if (filename[0] == '\0' || dtype != INT16) return;
        uint16_t b16_big_endian;

        if (!ensure_write_handles(mode)) {
            return;
        }
        bool want_text = dump_text_enabled();
        bool want_rawhex = dump_rawhex_enabled();
        bool want_bin = dump_bin_enabled();
        for (int i = 0; i < size; ++i) {
            res = decode(arr[i], data_type);
            string res_mant_bit = bitset<12>(res.mantissa).to_string();
            string res_bit = bitset<16>(arr[i]).to_string();
            b16_big_endian = __builtin_bswap16((uint16_t)arr[i]);
            if (want_text) fprintf(fp_write, "%s\n", res_bit.c_str());
            if (want_rawhex) fprintf(fp_write_rawhex, "%04x %s exp:%04d mant:%s/%06x %s\n", (uint16_t)arr[i], res_bit.c_str(), res.exponent, res_mant_bit.c_str(), (uint32_t)res.mantissa, exception_flag(&res.exception).c_str());
            if (want_bin) fwrite (&b16_big_endian, sizeof(uint16_t), 1, fp_write_double_bin);
        }
    }

    // 写int8数组(当fp4_4b=1时代表为4-bit packed e2m1，对应mxf4nvf4指令)
    void write_int8_array_to_file(int8_t* arr, int size, int mode, int data_type, int fp4_4b) {
        if (!fmac_dump_output_enabled()) return;
        if (filename[0] == '\0' || dtype != INT8) return;
        string res_bit;
        bool pack_nibbles_in_bin = (data_type == 12 || data_type == 13); // 分别对应于写入 s4 或 u4 元素流
        if (!ensure_write_handles(mode)) {
            return;
        }
        bool want_text = dump_text_enabled();
        bool want_rawhex = dump_rawhex_enabled();
        bool want_bin = dump_bin_enabled();
        for (int i = 0; i < size; ++i) {
            res_bit = bitset<8>(arr[i]).to_string();
            if (data_type == 4) { // FP4e2m1特殊处理：解码前需要先将8-bit packed e2m1恢复到原位置
                if (fp4_4b ==1) {// 4-bit packed e2m1，此时输入进来的e2m1格式被紧凑排列到一个8bit格式中
                    int8_t low4  = static_cast<int8_t>(static_cast<uint8_t>(arr[i]) & 0x0f);// 取8bit中的低4bit
                    int8_t high4 = static_cast<int8_t>((static_cast<uint8_t>(arr[i]) >> 4) & 0x0f);// 取8bit中的高4bit
                    FloatDecode res_low4 = decode(low4, data_type);
                    FloatDecode res_high4 = decode(high4, data_type);
                    string low4_bit = bitset<4>(static_cast<uint8_t>(low4) & 0x0f).to_string();
                    string high4_bit = bitset<4>(static_cast<uint8_t>(high4) & 0x0f).to_string();
                    string res_low4_mant_bit = bitset<4>(res_low4.mantissa).to_string();
                    string res_high4_mant_bit = bitset<4>(res_high4.mantissa).to_string();
                    // txt.log按一个fp4数据一行输出，顺序与rawhex/bin保持一致：先高4bit，后低4bit。
                    if (want_text) {
                        fprintf(fp_write, "%s\n", high4_bit.c_str());
                        fprintf(fp_write, "%s\n", low4_bit.c_str());
                    }
                    // rawhex按别的dtype习惯保持“hex + raw bits”；mant:字段继续打印解码后的尾数。
                    // 先写入高4bit，再写入低4bit，这样能和.bin文件对上
                    if (want_rawhex) {
                        fprintf(fp_write_rawhex, "%2x %s exp:%04d mant:%s/%06x %s\n", (uint8_t)high4, high4_bit.c_str(), res_high4.exponent, res_high4_mant_bit.c_str(), (uint32_t)res_high4.mantissa, exception_flag(&res_high4.exception).c_str());
                        fprintf(fp_write_rawhex, "%2x %s exp:%04d mant:%s/%06x %s\n", (uint8_t)low4, low4_bit.c_str(), res_low4.exponent, res_low4_mant_bit.c_str(), (uint32_t)res_low4.mantissa, exception_flag(&res_low4.exception).c_str());
                    }
                } else {// 8-bit packed e2m1是放在8bit的正中间的，需要恢复原位置方便解码
                    int8_t tmp = static_cast<int8_t>((static_cast<uint8_t>(arr[i]) >> 2) & 0x0f);
                    res = decode(tmp, data_type);
                    string res_mant_bit = bitset<4>(res.mantissa).to_string();
                    if (want_rawhex) fprintf(fp_write_rawhex, "%02x %s exp:%04d mant:%s/%06x %s\n", (uint8_t)arr[i], res_bit.c_str(), res.exponent, res_mant_bit.c_str(), (uint32_t)res.mantissa, exception_flag(&res.exception).c_str());
                }
            } else if (data_type == 10) { // s8
                if (want_rawhex) fprintf(fp_write_rawhex, "%02x %s int:%d\n",
                        (uint8_t)arr[i],
                        res_bit.c_str(),
                        (int8_t)arr[i]);
            } else if (data_type == 11) { // u8
                if (want_rawhex) fprintf(fp_write_rawhex, "%02x %s uint:%u\n",
                        (uint8_t)arr[i],
                        res_bit.c_str(),
                        (uint8_t)arr[i]);
            } else if (data_type == 12) { // s4
                uint8_t raw4 = static_cast<uint8_t>(arr[i]) & 0x0f;
                string raw4_bit = bitset<4>(raw4).to_string();
                int32_t sval = (raw4 & 0x8) ? (int32_t)raw4 - 16 : (int32_t)raw4;
                if (want_rawhex) fprintf(fp_write_rawhex, "%01x %s int:%d\n",
                        raw4,
                        raw4_bit.c_str(),
                        sval);
            } else if (data_type == 13) { // u4
                uint8_t raw4 = static_cast<uint8_t>(arr[i]) & 0x0f;
                string raw4_bit = bitset<4>(raw4).to_string();
                if (want_rawhex) fprintf(fp_write_rawhex, "%01x %s uint:%u\n",
                        raw4,
                        raw4_bit.c_str(),
                        raw4);
            } else if (data_type == 8) { // generic integer byte
                if (want_rawhex) fprintf(fp_write_rawhex, "%02x %s int:%d uint:%u\n",
                        (uint8_t)arr[i],
                        res_bit.c_str(),
                        (int8_t)arr[i],
                        (uint8_t)arr[i]);
            } else {
                res = decode(arr[i], data_type);
                string res_mant_bit = bitset<4>(res.mantissa).to_string();
                if (want_rawhex) fprintf(fp_write_rawhex, "%02x %s exp:%04d mant:%s/%06x %s\n", (uint8_t)arr[i], res_bit.c_str(), res.exponent, res_mant_bit.c_str(), (uint32_t)res.mantissa, exception_flag(&res.exception).c_str());
            }

            if (want_text && !(data_type == 4 && fp4_4b == 1)) {
                fprintf(fp_write, "%s\n", res_bit.c_str());
            }
            if (want_bin && !pack_nibbles_in_bin) {
                fwrite (&arr[i], sizeof(int8_t), 1, fp_write_double_bin);
            }
        }
        if (want_bin && pack_nibbles_in_bin) {
            for (int i = 0; i < size; i += 2) {
                uint8_t low = static_cast<uint8_t>(arr[i]) & 0x0f;
                uint8_t high = (i + 1 < size) ? (static_cast<uint8_t>(arr[i + 1]) & 0x0f) : 0;
                // Element 0 occupies bits [3:0], element 1 occupies bits
                // [7:4].  This is the same low-nibble-first order used when
                // packing operands into PTX MMA registers.
                uint8_t packed = static_cast<uint8_t>(low | (high << 4));
                fwrite(&packed, sizeof(uint8_t), 1, fp_write_double_bin);
            }
        }
    }

    // 读int32数组
    int read_int32_array_from_file(int* arr, int max_size, int mode) {
        if (filename[0] == '\0' || dtype != INT32) return -1;
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
            float temp_float;
            memcpy(&temp_float, &arr[count], sizeof(float));
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
        close_write_handles();
    }
};
