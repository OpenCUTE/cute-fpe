ROOT := ..
CMODE_DIR := $(ROOT)/cmodel/necessary_source
RTL_DIR := $(ROOT)/rtl/fpe/rtl
RTL_SYNC := $(ROOT)/rtl/sync_pipe_to_fpe.py

DTYPE ?= 1
INT4_DTYPES := 15 16 17 18
CMODE_DTYPE_MAP_11 := 32
CMODE_DTYPE_MAP_12 := 38
CMODE_DTYPE_MAP_7 := 57
CMODE_DTYPE_MAP_8 := 63
CMODE_DTYPE_MAP_13 := 75
CMODE_DTYPE_MAP_14 := 69
CMODE_DTYPE ?= $(or $(CMODE_DTYPE_MAP_$(DTYPE)),$(DTYPE))
RTL_BITSIZE_FROM_DEFINES := $(shell awk '$$1=="`define" && $$2=="ReduceWidth" {print $$3; exit}' $(RTL_DIR)/defines.sv 2>/dev/null)
RTL_BITSIZE ?= $(or $(RTL_BITSIZE_FROM_DEFINES),512)
CMODE_BITSIZE ?= $(RTL_BITSIZE)
INT4_K ?= $(shell expr $(RTL_BITSIZE) / 4)
IS_INT4_DTYPE := $(filter $(INT4_DTYPES),$(DTYPE))
INT4_K_ARG := $(if $(IS_INT4_DTYPE),--int4-k $(INT4_K),)
VEC_COUNT ?= 100
SEED ?= 1
VEC_START ?= 0
RTL_VEC_START ?= 0
TIMEOUT_CYCLES ?= 2000
ULP_TOLERANCE ?= 0
STOP_ON_FAIL ?= 1
VERBOSE_PASS ?= 0
TRACE ?= 0
TRACE_DETAIL ?= 0
DUMP_WAVE ?= 0
CMODE_DUMP_ALL ?= 0
CMODE_TRACE_INTERMEDIATE ?= 0
DTYPES ?= 0 1 2 3 4 5 6 11 12 15 16 17 18 19
SMOKE_COUNT ?= 2
REGRESS_COUNT ?= 100
BATCH_COUNT ?= 100000
DEBUG_DIR ?= mismatch_debug

RUN_DIR ?= runs/dtype_$(DTYPE)_seed_$(SEED)_count_$(VEC_COUNT)
VECTOR_DIR := $(RUN_DIR)/vectors
EXISTING_VECTOR_DIR ?= $(VECTOR_DIR)
CMODE_OUTPUT_DIR := $(RUN_DIR)/cmode_output
CMODE_SCALE_DIR := $(RUN_DIR)/cmode_scale
CMODE_TRACE_DIR := $(RUN_DIR)/cmode_trace
CMODE_REPLAY_INPUT_DIR ?= $(CMODE_OUTPUT_DIR)
CMODE_REPLAY_SCALE_DIR ?= $(CMODE_SCALE_DIR)
CMODE_REPLAY_VEC ?= 0
CMODE_REPLAY_BIN ?= $(RUN_DIR)/all_fmac_replay
BUILD_DIR := build
SIM_DIR := $(BUILD_DIR)/verilator_trace_$(TRACE)_detail_$(TRACE_DETAIL)
SIM_BIN := $(SIM_DIR)/obj_dir/Vrtl_cmodel_compare_tb
FAIL_LOG := $(RUN_DIR)/fail_log.txt
WAVE_FILE := $(RUN_DIR)/tb_top.fst

RTL_SRCS := $(shell find $(RTL_DIR) -maxdepth 1 -name '*.sv' ! -name 'defines.sv' | sort)
SIM_SRCS := tb/rtl_cmodel_compare_tb.sv $(RTL_DIR)/defines.sv $(RTL_SRCS)
TRACE_DETAIL_FLAGS := $(if $(filter 1,$(TRACE_DETAIL)),--trace-depth 99 --trace-max-array 1024 --trace-max-width 4096 --trace-structs,)
TRACE_FLAGS := $(if $(filter 1,$(TRACE)),--trace-fst $(TRACE_DETAIL_FLAGS),)
CMODE_TRACE_ENV := $(if $(filter 1,$(CMODE_TRACE_INTERMEDIATE)),FMAC_TRACE_INTERMEDIATE=1 FMAC_TRACE_DIR=$(abspath $(CMODE_TRACE_DIR)),)

.PHONY: all run run-existing-vectors cmodel-replay run-dtypes vectors sim smoke regress debug-mismatch debug-mismatch-dtypes clean check-tools check-rtl-sync sync-rtl

all: run

check-tools:
	@command -v verilator >/dev/null || { echo "verilator not found"; exit 1; }
	@command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }

check-rtl-sync:
	python3 $(RTL_SYNC) --check

sync-rtl:
	python3 $(RTL_SYNC) --apply

$(RUN_DIR)/all_fmac_test: $(CMODE_DIR)/all_fmac_test.cu
	@mkdir -p $(RUN_DIR) $(CMODE_OUTPUT_DIR) $(CMODE_SCALE_DIR) $(CMODE_TRACE_DIR)
	c++ -x c++ -std=c++11 -O2 \
		-DNVIDIA_ON=0 \
		-DOUTPUT_DATA_DIR='"$(abspath $(CMODE_OUTPUT_DIR))"' \
		-DTRACE_FILE='"$(abspath $(CMODE_TRACE_DIR))"' \
		-DSCALE_OUTPUT_DATA_DIR='"$(abspath $(CMODE_SCALE_DIR))"' \
		-DFMAC_DUMP_OUTPUT=1 \
		-DFMAC_DUMP_ALL=$(CMODE_DUMP_ALL) \
		$(CMODE_DIR)/all_fmac_test.cu \
		-o $(RUN_DIR)/all_fmac_test

$(CMODE_REPLAY_BIN): $(CMODE_DIR)/all_fmac_test.cu
	@mkdir -p $(RUN_DIR) $(CMODE_TRACE_DIR)
	c++ -x c++ -std=c++11 -O2 \
		-DNVIDIA_ON=0 \
		-DOUTPUT_DATA_DIR='"$(abspath $(CMODE_REPLAY_INPUT_DIR))"' \
		-DTRACE_FILE='"$(abspath $(CMODE_TRACE_DIR))"' \
		-DSCALE_OUTPUT_DATA_DIR='"$(abspath $(CMODE_REPLAY_SCALE_DIR))"' \
		-DFMAC_DUMP_OUTPUT=1 \
		-DFMAC_DUMP_ALL=1 \
		$(CMODE_DIR)/all_fmac_test.cu \
		-o $(CMODE_REPLAY_BIN)

cmodel-replay: $(CMODE_REPLAY_BIN)
	@mkdir -p $(CMODE_TRACE_DIR)
	$(CMODE_TRACE_ENV) $(CMODE_REPLAY_BIN) \
		--dtype $(CMODE_DTYPE) \
		--bitsize $(CMODE_BITSIZE) $(INT4_K_ARG) \
		--replay-vec $(CMODE_REPLAY_VEC)

vectors: $(RUN_DIR)/all_fmac_test
	@mkdir -p $(RUN_DIR) $(CMODE_OUTPUT_DIR) $(CMODE_SCALE_DIR) $(CMODE_TRACE_DIR) $(VECTOR_DIR)
	$(CMODE_TRACE_ENV) $(RUN_DIR)/all_fmac_test \
		--dtype $(CMODE_DTYPE) \
		--vec-start $(VEC_START) \
		--vec-count $(VEC_COUNT) \
		--bitsize $(CMODE_BITSIZE) $(INT4_K_ARG) \
		--seed $(SEED) \
		--seed-mode per_vec \
		--print-every 0
	python3 scripts/prepare_vectors.py \
		--dtype $(DTYPE) \
		--vec-count $(VEC_COUNT) \
		--cmodel-bitsize $(CMODE_BITSIZE) \
		--rtl-bitsize $(RTL_BITSIZE) $(INT4_K_ARG) \
		--cmodel-output-dir $(CMODE_OUTPUT_DIR) \
		--cmodel-scale-dir $(CMODE_SCALE_DIR) \
		--out-dir $(VECTOR_DIR)

sim: check-tools check-rtl-sync $(SIM_BIN)

$(SIM_BIN): $(SIM_SRCS) Makefile
	@mkdir -p $(SIM_DIR)
	verilator --binary --main --timing --sv \
		-Wno-fatal \
		$(TRACE_FLAGS) \
		-I$(RTL_DIR) \
		-Mdir $(SIM_DIR)/obj_dir \
		--top-module rtl_cmodel_compare_tb \
		$(SIM_SRCS)

run: vectors sim
		$(SIM_BIN) \
			+VEC_COUNT=$(VEC_COUNT) \
			+VEC_START=$(RTL_VEC_START) \
			+OPCODE=$(DTYPE) \
			+TIMEOUT_CYCLES=$(TIMEOUT_CYCLES) \
			+ULP_TOLERANCE=$(ULP_TOLERANCE) \
			+STOP_ON_FAIL=$(STOP_ON_FAIL) \
			+VERBOSE_PASS=$(VERBOSE_PASS) \
			+A_HEX=$(VECTOR_DIR)/A.txt \
			+B_HEX=$(VECTOR_DIR)/B.txt \
			+C_HEX=$(VECTOR_DIR)/C.txt \
			+ASCALE_HEX=$(VECTOR_DIR)/A_scale.txt \
			+BSCALE_HEX=$(VECTOR_DIR)/B_scale.txt \
			+EXPECTED_HEX=$(VECTOR_DIR)/D.txt \
			+FAIL_LOG=$(FAIL_LOG) \
			+DUMP_WAVE=$(DUMP_WAVE) \
			+WAVE_FILE=$(WAVE_FILE)

run-existing-vectors: sim
	@mkdir -p $(RUN_DIR)
		$(SIM_BIN) \
			+VEC_COUNT=$(VEC_COUNT) \
			+VEC_START=$(RTL_VEC_START) \
			+OPCODE=$(DTYPE) \
			+TIMEOUT_CYCLES=$(TIMEOUT_CYCLES) \
			+ULP_TOLERANCE=$(ULP_TOLERANCE) \
			+STOP_ON_FAIL=$(STOP_ON_FAIL) \
			+VERBOSE_PASS=$(VERBOSE_PASS) \
			+A_HEX=$(EXISTING_VECTOR_DIR)/A.txt \
			+B_HEX=$(EXISTING_VECTOR_DIR)/B.txt \
			+C_HEX=$(EXISTING_VECTOR_DIR)/C.txt \
			+ASCALE_HEX=$(EXISTING_VECTOR_DIR)/A_scale.txt \
			+BSCALE_HEX=$(EXISTING_VECTOR_DIR)/B_scale.txt \
			+EXPECTED_HEX=$(EXISTING_VECTOR_DIR)/D.txt \
			+FAIL_LOG=$(FAIL_LOG) \
			+DUMP_WAVE=$(DUMP_WAVE) \
			+WAVE_FILE=$(WAVE_FILE)

run-dtypes:
	@for dtype in $(DTYPES); do \
		echo "== run dtype $$dtype count $(VEC_COUNT) seed $(SEED) =="; \
		$(MAKE) run DTYPE=$$dtype VEC_START=$(VEC_START) VEC_COUNT=$(VEC_COUNT) SEED=$(SEED) \
			CMODE_BITSIZE=$(CMODE_BITSIZE) RTL_BITSIZE=$(RTL_BITSIZE) \
			TIMEOUT_CYCLES=$(TIMEOUT_CYCLES) ULP_TOLERANCE=$(ULP_TOLERANCE) \
			STOP_ON_FAIL=$(STOP_ON_FAIL) TRACE=0 DUMP_WAVE=0 || exit $$?; \
	done

smoke:
	@for dtype in $(DTYPES); do \
		echo "== smoke dtype $$dtype count $(SMOKE_COUNT) seed $(SEED) =="; \
		$(MAKE) run DTYPE=$$dtype VEC_COUNT=$(SMOKE_COUNT) SEED=$(SEED) \
			CMODE_BITSIZE=$(CMODE_BITSIZE) RTL_BITSIZE=$(RTL_BITSIZE) \
			TIMEOUT_CYCLES=$(TIMEOUT_CYCLES) ULP_TOLERANCE=$(ULP_TOLERANCE) \
			STOP_ON_FAIL=$(STOP_ON_FAIL) TRACE=0 DUMP_WAVE=0 || exit $$?; \
	done

regress:
	@for dtype in $(DTYPES); do \
		echo "== regress dtype $$dtype count $(REGRESS_COUNT) seed $(SEED) =="; \
		$(MAKE) run DTYPE=$$dtype VEC_COUNT=$(REGRESS_COUNT) SEED=$(SEED) \
			CMODE_BITSIZE=$(CMODE_BITSIZE) RTL_BITSIZE=$(RTL_BITSIZE) \
			TIMEOUT_CYCLES=$(TIMEOUT_CYCLES) ULP_TOLERANCE=$(ULP_TOLERANCE) \
			STOP_ON_FAIL=$(STOP_ON_FAIL) TRACE=0 DUMP_WAVE=0 || exit $$?; \
	done

debug-mismatch: check-tools
	python3 scripts/capture_first_mismatch.py \
		--make "$(MAKE)" \
		--dtype $(DTYPE) \
		--seed $(SEED) \
		--vec-start $(VEC_START) \
		--vec-count $(VEC_COUNT) \
		--batch-count $(BATCH_COUNT) \
		--cmode-bitsize $(CMODE_BITSIZE) \
		--rtl-bitsize $(RTL_BITSIZE) \
		--timeout-cycles $(TIMEOUT_CYCLES) \
		--ulp-tolerance $(ULP_TOLERANCE) \
		--debug-dir $(DEBUG_DIR)

debug-mismatch-dtypes: check-tools
	@for dtype in $(DTYPES); do \
		echo "== debug-mismatch dtype $$dtype count $(VEC_COUNT) batch $(BATCH_COUNT) seed $(SEED) =="; \
		$(MAKE) debug-mismatch DTYPE=$$dtype VEC_START=$(VEC_START) VEC_COUNT=$(VEC_COUNT) SEED=$(SEED) \
			CMODE_BITSIZE=$(CMODE_BITSIZE) RTL_BITSIZE=$(RTL_BITSIZE) \
			BATCH_COUNT=$(BATCH_COUNT) TIMEOUT_CYCLES=$(TIMEOUT_CYCLES) \
			ULP_TOLERANCE=$(ULP_TOLERANCE) DEBUG_DIR=$(DEBUG_DIR) || exit $$?; \
		if [ -f "$(DEBUG_DIR)/first_mismatch.txt" ] && grep -q '^original_vec=' "$(DEBUG_DIR)/first_mismatch.txt"; then \
			echo "== mismatch found at dtype $$dtype; stopping =="; \
			cat "$(DEBUG_DIR)/first_mismatch.txt"; \
			exit 2; \
		fi; \
	done; \
	echo "PASS no mismatch found for DTYPES=$(DTYPES)"

clean:
	rm -rf runs build mismatch_debug output_data scale_output_data
