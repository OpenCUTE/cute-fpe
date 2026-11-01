# Builds the self-check and the nine probes for sm_120a (RTX 5090).
NVCC     ?= nvcc
ARCH     ?= -gencode arch=compute_120a,code=sm_120a
NVFLAGS  ?= -O2 -std=c++17 $(ARCH)
BUILD    := build
COMMON   := $(wildcard common/*.cuh common/*.hpp)

TESTS := selfcheck/selfcheck \
         tests/test1_output_uniformity/test1 \
         tests/test2_accumulation_order/test2 \
         tests/test3_hidden_precision/test3 \
         tests/test4_rounding/test4 \
         tests/test5_special_values/test5 \
         tests/test6_scale_order/test6 \
         tests/test7_exponent_floor/test7 \
         tests/test8_zero_exponent/test8 \
         tests/test9_fp4_fp6_low_bits/test9

BINS := $(addprefix $(BUILD)/,$(notdir $(TESTS)))

all: $(BINS)

define rule
$(BUILD)/$(notdir $(1)): $(1).cu $(COMMON) | $(BUILD)
	$(NVCC) $(NVFLAGS) -o $$@ $$<
endef
$(foreach t,$(TESTS),$(eval $(call rule,$(t))))

$(BUILD):
	mkdir -p $(BUILD)

run: all
	./run_all.sh

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
