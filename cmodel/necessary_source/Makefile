.PHONY: build build_static_runtime build_256b build_gen_data gen_data gen_data_fp32 gen_data_fp16 test check_gpu test_nv_fp32_128b test_nv_fp32_256b test_nv_128b test_nv_256b test_nv_fp16_128b test_nv_fp16_256b show_test_logs test_128b_reply test_128b_trace clean

# 当NVIDIA_ON=1时，使用NVCC编译带nvidia mma指令的程序，用于模拟器原型验证；否则使仅用CXX编译模拟器部分生成数据
NVIDIA_ON ?= 0
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= 120a
CUTLASS_DIR ?= $(HOME)/cutlass
CUTLASS_INCLUDE_DIR ?= $(CUTLASS_DIR)/include
CUTLASS_UTIL_INCLUDE_DIR ?= $(CUTLASS_DIR)/tools/util/include
NVCC ?= $(CUDA_HOME)/bin/nvcc
NVCCFLAGS ?= -std=c++17 -O2 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
INCLUDES ?= -I$(CUTLASS_INCLUDE_DIR) -I$(CUTLASS_UTIL_INCLUDE_DIR)
CXX ?= g++
CXXFLAGS ?= -x c++ -std=c++11 -O2
LDFLAGS ?=
STATIC_CXX_RUNTIME ?= 0
TARGET := all_fmac_test
TARGET_256B := all_fmac_test_256b
GEN_DATA_TARGET := all_fmac_test_gen_data
OUTPUT_DATA_DIR ?= output_data
TRACE_FILE ?= debug_data
SCALE_OUTPUT_DATA_DIR ?= scale_output_data
FMAC_DUMP_OUTPUT ?= 1
FMAC_DUMP_ALL ?= 0
TEST_NV_DUMP ?= 0
CPP_PATH_DEFINES := -DOUTPUT_DATA_DIR='"$(OUTPUT_DATA_DIR)"' -DTRACE_FILE='"$(TRACE_FILE)"' -DSCALE_OUTPUT_DATA_DIR='"$(SCALE_OUTPUT_DATA_DIR)"' -DFMAC_DUMP_OUTPUT=$(FMAC_DUMP_OUTPUT) -DFMAC_DUMP_ALL=$(FMAC_DUMP_ALL)
EXTRA_DEFINES ?=
OUTPUT_DATA_DIR_256B ?= output_data_256b
TRACE_FILE_256B ?= debug_data_256b
SCALE_OUTPUT_DATA_DIR_256B ?= scale_output_data_256b
GEN_OUTPUT_DATA_DIR ?= test_output_data
GEN_TRACE_FILE ?= test_debug_data
GEN_SCALE_OUTPUT_DATA_DIR ?= test_scale_output_data
DTYPES ?= 1 2 3 9 10 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 \
		  60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 97 99 101 103 129 131 133 135

DTYPES_0 ?= 1 2 3
DTYPES_1 ?= 9 10
DTYPES_2 ?= 32 33 34
DTYPES_3 ?= 35 36 37
DTYPES_4 ?= 38 39 40
DTYPES_5 ?= 41 42 43
DTYPES_6 ?= 44 45 46
DTYPES_7 ?= 47 48 49
DTYPES_8 ?= 50 51 52
DTYPES_9 ?= 53 54 55
DTYPES_10 ?= 56 57 58
DTYPES_11 ?= 59 60 61
DTYPES_12 ?= 62 63 64
DTYPES_13 ?= 65 66 67
DTYPES_14 ?= 68 69 70
DTYPES_15 ?= 71 72 73
DTYPES_16 ?= 74 75 76
DTYPES_17 ?= 77 78 79
DTYPES_18 ?= 80 81
DTYPES_24 ?= 97 99 101
DTYPES_25 ?= 103 129 131
DTYPES_26 ?= 133 135

DTYPES_FP16_0 ?= 1 32
DTYPES_FP16_1 ?= 33 34
DTYPES_FP16_2 ?= 35 36
DTYPES_FP16_3 ?= 37 38
DTYPES_FP16_4 ?= 39 40
DTYPES_FP16_5 ?= 41 42
DTYPES_FP16_6 ?= 43 44
DTYPES_FP16_7 ?= 45 46
DTYPES_FP16_8 ?= 47 48
DTYPES_FP16_9 ?= 49 50
DTYPES_FP16_10 ?= 51 52
DTYPES_FP16_11 ?= 53 54	
DTYPES_FP16_12 ?= 55 56

GEN_DATA_FP32_GROUPS ?= 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
GEN_DATA_FP16_GROUPS ?= 0 1 2 3 4 5 6 7 8 9 10 11 12
GEN_DATA_FP32_JOBS ?= $(words $(GEN_DATA_FP32_GROUPS))
GEN_DATA_FP16_JOBS ?= $(words $(GEN_DATA_FP16_GROUPS))
TEST_NV_FP32_GROUPS ?= $(GEN_DATA_FP32_GROUPS)
TEST_NV_FP32_128B_JOBS ?= 27
TEST_NV_FP32_256B_JOBS ?= 27
TEST_NV_FP16_GROUPS ?= $(GEN_DATA_FP16_GROUPS)
TEST_NV_FP16_128B_JOBS ?= $(words $(TEST_NV_FP16_GROUPS))
TEST_NV_FP16_256B_JOBS ?= $(words $(TEST_NV_FP16_GROUPS))
TEST_NV_FP32_128B_LOG_DIR ?= fmac_test_nv_fp32_128b_logs
TEST_NV_FP32_256B_LOG_DIR ?= fmac_test_nv_fp32_256b_logs
TEST_NV_FP16_128B_LOG_DIR ?= fmac_test_nv_fp16_128b_logs
TEST_NV_FP16_256B_LOG_DIR ?= fmac_test_nv_fp16_256b_logs

VEC_COUNT ?= 10000000
GEN_ARGS ?=
TEST_NV_256B_ARGS ?= --bitsize 256
GEN_ENV ?= TOPOLOGY_128_INPUT=1
ACCUM_OUTPUT ?= fp32
ACCUM_OUTPUT_FLAG := $(if $(filter fp16,$(ACCUM_OUTPUT)),--output-fp16,)

ifeq ($(NVIDIA_ON),1)
COMPILER := $(NVCC)
COMPILER_FLAGS := $(NVCCFLAGS) $(INCLUDES) -DNVIDIA_ON=1 $(CPP_PATH_DEFINES) $(EXTRA_DEFINES)
STATIC_CXX_RUNTIME_FLAGS := -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc
else
COMPILER := $(CXX)
COMPILER_FLAGS := $(CXXFLAGS) -DNVIDIA_ON=0 $(CPP_PATH_DEFINES) $(EXTRA_DEFINES)
STATIC_CXX_RUNTIME_FLAGS := -static-libstdc++ -static-libgcc
endif

ifeq ($(STATIC_CXX_RUNTIME),1)
LINK_FLAGS := $(STATIC_CXX_RUNTIME_FLAGS) $(LDFLAGS)
else
LINK_FLAGS := $(LDFLAGS)
endif

build:
	$(COMPILER) $(COMPILER_FLAGS) all_fmac_test.cu $(LINK_FLAGS) -o $(TARGET)

build_static_runtime:
	@$(MAKE) build STATIC_CXX_RUNTIME=1

build_256b:
	@mkdir -p $(OUTPUT_DATA_DIR_256B) $(TRACE_FILE_256B) $(SCALE_OUTPUT_DATA_DIR_256B)
	@$(COMPILER) $(filter-out $(CPP_PATH_DEFINES),$(COMPILER_FLAGS)) \
		-DOUTPUT_DATA_DIR='"$(OUTPUT_DATA_DIR_256B)"' \
		-DTRACE_FILE='"$(TRACE_FILE_256B)"' \
		-DSCALE_OUTPUT_DATA_DIR='"$(SCALE_OUTPUT_DATA_DIR_256B)"' \
		-DFMAC_DUMP_OUTPUT=$(FMAC_DUMP_OUTPUT) \
		-DFMAC_DUMP_ALL=$(FMAC_DUMP_ALL) \
		$(EXTRA_DEFINES) \
		all_fmac_test.cu $(LINK_FLAGS) -o $(TARGET_256B)

build_gen_data:
	@mkdir -p $(GEN_OUTPUT_DATA_DIR) $(GEN_TRACE_FILE) $(GEN_SCALE_OUTPUT_DATA_DIR)
	@$(COMPILER) $(filter-out $(CPP_PATH_DEFINES),$(COMPILER_FLAGS)) \
		-DOUTPUT_DATA_DIR='"$(GEN_OUTPUT_DATA_DIR)"' \
		-DTRACE_FILE='"$(GEN_TRACE_FILE)"' \
		-DSCALE_OUTPUT_DATA_DIR='"$(GEN_SCALE_OUTPUT_DATA_DIR)"' \
		-DFMAC_DUMP_OUTPUT=$(FMAC_DUMP_OUTPUT) \
		-DFMAC_DUMP_ALL=$(FMAC_DUMP_ALL) \
		$(EXTRA_DEFINES) \
		all_fmac_test.cu $(LINK_FLAGS) -o $(GEN_DATA_TARGET)

gen_data: build_gen_data
	@for dtype in $(DTYPES); do \
		echo "Generating dtype $$dtype into $(GEN_OUTPUT_DATA_DIR)"; \
		$(GEN_ENV) ./$(GEN_DATA_TARGET) --dtype $$dtype --vec-count $(VEC_COUNT) $(ACCUM_OUTPUT_FLAG) $(GEN_ARGS) || exit $$?; \
	done

gen_data_fp32: build_gen_data
	@$(MAKE) --no-print-directory -j$(GEN_DATA_FP32_JOBS) \
		$(addprefix gen_data_fp32_group_,$(GEN_DATA_FP32_GROUPS))

gen_data_fp16: build_gen_data
	@$(MAKE) --no-print-directory -j$(GEN_DATA_FP16_JOBS) \
		$(addprefix gen_data_fp16_group_,$(GEN_DATA_FP16_GROUPS))

gen_data_fp32_group_%:
	@dtypes="$(DTYPES_$*)"; \
	if [ -z "$$dtypes" ]; then \
		echo "Skipping empty FP32 group $*"; \
	else \
		echo "Generating FP32 group $*: $$dtypes into $(GEN_OUTPUT_DATA_DIR)"; \
		for dtype in $$dtypes; do \
			echo "  dtype $$dtype"; \
			$(GEN_ENV) ./$(GEN_DATA_TARGET) --dtype $$dtype --vec-count $(VEC_COUNT) $(GEN_ARGS) || exit $$?; \
		done; \
	fi

gen_data_fp16_group_%:
	@dtypes="$(DTYPES_FP16_$*)"; \
	if [ -z "$$dtypes" ]; then \
		echo "Skipping empty FP16 group $*"; \
	else \
		echo "Generating FP16 group $*: $$dtypes into $(GEN_OUTPUT_DATA_DIR)"; \
		for dtype in $$dtypes; do \
			echo "  dtype $$dtype"; \
			$(GEN_ENV) ./$(GEN_DATA_TARGET) --dtype $$dtype --vec-count $(VEC_COUNT) --output-fp16 $(GEN_ARGS) || exit $$?; \
		done; \
	fi

test: build
	./$(TARGET)

check_gpu:
	@command -v nvidia-smi >/dev/null || { echo "NVIDIA GPU check failed: nvidia-smi was not found."; exit 1; }
	@nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || { echo "NVIDIA GPU check failed: no CUDA device is available to this environment."; exit 1; }

test_nv_fp32_128b: check_gpu
	@mkdir -p $(TEST_NV_FP32_128B_LOG_DIR)
	@$(MAKE) build NVIDIA_ON=1 FMAC_DUMP_OUTPUT=$(TEST_NV_DUMP)
	@$(MAKE) --no-print-directory -j$(TEST_NV_FP32_128B_JOBS) \
		$(addprefix test_nv_fp32_128b_group_,$(TEST_NV_FP32_GROUPS))

test_nv_fp32_256b: check_gpu
	@mkdir -p $(TEST_NV_FP32_256B_LOG_DIR)
	@$(MAKE) build_256b NVIDIA_ON=1 FMAC_DUMP_OUTPUT=$(TEST_NV_DUMP)
	@$(MAKE) --no-print-directory -j$(TEST_NV_FP32_256B_JOBS) \
		$(addprefix test_nv_fp32_256b_group_,$(TEST_NV_FP32_GROUPS))

test_nv_fp16_128b: check_gpu
	@mkdir -p $(TEST_NV_FP16_128B_LOG_DIR)
	@$(MAKE) build NVIDIA_ON=1 FMAC_DUMP_OUTPUT=$(TEST_NV_DUMP)
	@$(MAKE) --no-print-directory -j$(TEST_NV_FP16_128B_JOBS) \
		$(addprefix test_nv_fp16_128b_group_,$(TEST_NV_FP16_GROUPS))

test_nv_fp16_256b: check_gpu
	@mkdir -p $(TEST_NV_FP16_256B_LOG_DIR)
	@$(MAKE) build_256b NVIDIA_ON=1 FMAC_DUMP_OUTPUT=$(TEST_NV_DUMP)
	@$(MAKE) --no-print-directory -j$(TEST_NV_FP16_256B_JOBS) \
		$(addprefix test_nv_fp16_256b_group_,$(TEST_NV_FP16_GROUPS))

test_nv_fp32_128b_group_%:
	@dtypes="$(DTYPES_$*)"; \
	log="$(TEST_NV_FP32_128B_LOG_DIR)/group_$*.log"; \
	if [ -z "$$dtypes" ]; then \
		echo "Skipping empty NV FP32 128b group $*" | tee "$$log"; \
	else \
		echo "Running NV FP32 128b group $*: $$dtypes" | tee "$$log"; \
			for dtype in $$dtypes; do \
				echo "  dtype $$dtype" | tee -a "$$log"; \
				FMAC_DUMP_MISMATCH_INPUT=1 TOPOLOGY_128_INPUT=1 ./$(TARGET) --dtype $$dtype --vec-count $(VEC_COUNT) $(ACCUM_OUTPUT_FLAG) $(GEN_ARGS) \
					2>&1 | tee -a "$$log" || exit $$?; \
			done; \
		fi

test_nv_fp32_256b_group_%:
	@dtypes="$(DTYPES_$*)"; \
	log="$(TEST_NV_FP32_256B_LOG_DIR)/group_$*.log"; \
	if [ -z "$$dtypes" ]; then \
		echo "Skipping empty NV FP32 256b group $*" | tee "$$log"; \
	else \
		echo "Running NV FP32 256b group $*: $$dtypes" | tee "$$log"; \
			for dtype in $$dtypes; do \
				echo "  dtype $$dtype" | tee -a "$$log"; \
				FMAC_DUMP_MISMATCH_INPUT=1 ./$(TARGET_256B) --dtype $$dtype --vec-count $(VEC_COUNT) $(ACCUM_OUTPUT_FLAG) $(TEST_NV_256B_ARGS) $(GEN_ARGS) \
					2>&1 | tee -a "$$log" || exit $$?; \
			done; \
		fi

test_nv_fp16_128b_group_%:
	@dtypes="$(DTYPES_FP16_$*)"; \
	log="$(TEST_NV_FP16_128B_LOG_DIR)/group_$*.log"; \
	if [ -z "$$dtypes" ]; then \
		echo "Skipping empty NV FP16 128b group $*" | tee "$$log"; \
	else \
		echo "Running NV FP16 128b group $*: $$dtypes" | tee "$$log"; \
			for dtype in $$dtypes; do \
				echo "  dtype $$dtype" | tee -a "$$log"; \
				FMAC_DUMP_MISMATCH_INPUT=1 TOPOLOGY_128_INPUT=1 ./$(TARGET) --dtype $$dtype --vec-count $(VEC_COUNT) --output-fp16 $(GEN_ARGS) \
					2>&1 | tee -a "$$log" || exit $$?; \
			done; \
		fi

test_nv_fp16_256b_group_%:
	@dtypes="$(DTYPES_FP16_$*)"; \
	log="$(TEST_NV_FP16_256B_LOG_DIR)/group_$*.log"; \
	if [ -z "$$dtypes" ]; then \
		echo "Skipping empty NV FP16 256b group $*" | tee "$$log"; \
	else \
		echo "Running NV FP16 256b group $*: $$dtypes" | tee "$$log"; \
			for dtype in $$dtypes; do \
				echo "  dtype $$dtype" | tee -a "$$log"; \
				FMAC_DUMP_MISMATCH_INPUT=1 ./$(TARGET_256B) --dtype $$dtype --vec-count $(VEC_COUNT) --output-fp16 $(TEST_NV_256B_ARGS) $(GEN_ARGS) \
					2>&1 | tee -a "$$log" || exit $$?; \
			done; \
		fi

show_test_logs:
	@for dir in $(TEST_NV_FP32_128B_LOG_DIR) $(TEST_NV_FP32_256B_LOG_DIR) $(TEST_NV_FP16_128B_LOG_DIR) $(TEST_NV_FP16_256B_LOG_DIR); do \
		echo "== $$dir =="; \
		if [ ! -d "$$dir" ]; then \
			echo "  missing"; \
			echo; \
			continue; \
		fi; \
		found=0; \
		success=0; \
		failed=0; \
		for log in "$$dir"/group_*.log; do \
			if [ ! -e "$$log" ]; then \
				continue; \
			fi; \
			found=1; \
			name=$$(basename "$$log"); \
			if grep -Eq "first mismatch at vec|results mismatch|mismatch_count=[1-9]" "$$log"; then \
				echo "  FAIL $$name"; \
				failed=$$((failed + 1)); \
			elif grep -q "finished with no mismatches" "$$log"; then \
				echo "  PASS $$name"; \
				success=$$((success + 1)); \
			else \
				echo "  INCOMPLETE $$name"; \
				failed=$$((failed + 1)); \
			fi; \
		done; \
		if [ "$$found" -eq 0 ]; then \
			echo "  no log files"; \
		fi; \
		echo "  summary: pass=$$success fail=$$failed"; \
		echo; \
	done

# 对已落盘的 128bit MXF8F6F4 向量进行 replay，并打开 trace 中间结果
test_128b_reply: build
	TOPOLOGY_128_INPUT=1 FMAC_TRACE_INTERMEDIATE=1 ./$(TARGET) --dtype 58 --replay-vec 0 $(ACCUM_OUTPUT_FLAG)

test_128b_trace: build
	TOPOLOGY_128_INPUT=1 FMAC_TRACE_INTERMEDIATE=1 ./$(TARGET) --all-128b-dtypes --vec-count 1000 $(ACCUM_OUTPUT_FLAG) | tee fmac_test_trace_1000.log

clean:
	@rm -f output_data/*.log
	@rm -f scale_output_data/*.log
	@rm -f debug_data/*.log
	@rm -f output_data_256b/*.log
	@rm -f scale_output_data_256b/*.log
	@rm -f debug_data_256b/*.log
	@rm -f $(TARGET)
	@rm -f $(TARGET_256B)
	@rm -f $(GEN_DATA_TARGET)
	@rm -f *.log
	@rm -f fmac_test_nv_fp32_128b_logs/*.log
	@rm -f fmac_test_nv_fp32_256b_logs/*.log
	@rm -f fmac_test_nv_fp16_128b_logs/*.log
	@rm -f fmac_test_nv_fp16_256b_logs/*.log

clean_log:
	@rm -f output_data/*.log
	@rm -f scale_output_data/*.log
	@rm -f debug_data/*.log
	@rm -f output_data_256b/*.log
	@rm -f scale_output_data_256b/*.log
	@rm -f debug_data_256b/*.log
	@rm -f *.log
	@rm -f fmac_test_nv_fp32_128b_logs/*.log
	@rm -f fmac_test_nv_fp32_256b_logs/*.log
	@rm -f fmac_test_nv_fp16_128b_logs/*.log
	@rm -f fmac_test_nv_fp16_256b_logs/*.log

clean_debug_log:
	@rm -f test_output_data/*.log
	@rm -f test_scale_output_data/*.log
	@rm -f test_debug_data/*.log
	@rm -f *.log
