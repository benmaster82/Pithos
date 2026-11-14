# pithos - GNU make build for Linux, macOS and MinGW.
#   make            optimized build (runtime CPU dispatch on x86-64)
#   make test       also builds tests
#   make NATIVE=1   tune for the build machine (-march=native / -mcpu=native)

CC      ?= cc
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
SRC      = src/main.c src/model.c src/gguf.c src/tokenizer.c src/sampler.c src/quant.c src/platform.c \
           src/quant_neon.c src/gpu_opencl.c src/gpu_d3d12.c src/gpu_vulkan.c src/json.c src/pcache.c src/server.c src/proc.c src/tools.c src/agent.c src/audio.c src/speech.c src/mp3.c
BUILD    = build
UNAME_S := $(shell uname -s 2>/dev/null)
UNAME_M := $(shell uname -m 2>/dev/null)

ifeq ($(OS),Windows_NT)
  EXE     = .exe
  LDLIBS  = -lm -lws2_32
  CFLAGS += -municode
else ifeq ($(UNAME_S),Darwin)
  LDLIBS  = -lm -lpthread
else
  LDLIBS  = -lm -lpthread -ldl
  CFLAGS += -D_GNU_SOURCE
endif

EXTRA_OBJ =
ifneq (,$(filter x86_64 amd64 AMD64,$(UNAME_M)))
  EXTRA_OBJ = $(BUILD)/quant_avx2.o $(BUILD)/quant_avx.o
endif
ifeq ($(OS),Windows_NT)
  EXTRA_OBJ = $(BUILD)/quant_avx2.o $(BUILD)/quant_avx.o
endif
ifneq (,$(filter aarch64 arm64,$(UNAME_M)))
  ifeq ($(UNAME_S),Darwin)
    ARCHFLAGS = -mcpu=apple-m1
  else ifneq (,$(shell grep -qw asimddp /proc/cpuinfo 2>/dev/null && grep -qw asimdhp /proc/cpuinfo && echo y))
    ARCHFLAGS = -march=armv8.2-a+dotprod+fp16
  else
    # no dotprod (Raspberry Pi 3 / 4, Cortex-A53 / A72...): plain ARMv8 NEON
    ARCHFLAGS = -march=armv8-a
  endif
endif
ifdef NATIVE
  ARCHFLAGS = $(if $(filter aarch64 arm64,$(UNAME_M)),-mcpu=native,-march=native)
endif

all: $(BUILD)/pithos$(EXE)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/quant_avx2.o: src/quant_avx2.c src/quant.h | $(BUILD)
	$(CC) $(CFLAGS) -mavx2 -mfma -mf16c -c $< -o $@

$(BUILD)/quant_avx.o: src/quant_avx.c src/quant.h | $(BUILD)
	$(CC) $(CFLAGS) -mavx -mf16c -c $< -o $@

$(BUILD)/pithos$(EXE): $(SRC) $(EXTRA_OBJ) src/*.h | $(BUILD)
	$(CC) $(CFLAGS) $(ARCHFLAGS) $(SRC) $(EXTRA_OBJ) -o $@ $(LDLIBS)

test: all
	$(CC) $(CFLAGS) $(ARCHFLAGS) tests/test_kernels.c src/gguf.c src/quant.c src/platform.c src/quant_neon.c $(EXTRA_OBJ) -o $(BUILD)/test_kernels$(EXE) $(LDLIBS)
	$(CC) $(CFLAGS) $(ARCHFLAGS) tests/test_gpu.c src/model.c src/gguf.c src/quant.c src/platform.c src/quant_neon.c src/gpu_opencl.c src/gpu_d3d12.c src/gpu_vulkan.c $(EXTRA_OBJ) -o $(BUILD)/test_gpu$(EXE) $(LDLIBS)
	$(CC) $(CFLAGS) $(ARCHFLAGS) tests/test_vk.c src/gpu_vulkan.c src/model.c src/gguf.c src/tokenizer.c src/quant.c src/platform.c src/quant_neon.c src/gpu_opencl.c src/gpu_d3d12.c $(EXTRA_OBJ) -o $(BUILD)/test_vk$(EXE) $(LDLIBS)
	$(CC) $(CFLAGS) tests/test_shell.c src/tools.c src/proc.c src/platform.c src/json.c -o $(BUILD)/test_shell$(EXE) $(LDLIBS)

# regenerate embedded sources (needs Python; outputs are committed)
gen:
	python3 tools/gen_unicode.py > src/unicode_data.h
	python3 tools/embed.py

clean:
	rm -rf $(BUILD)

.PHONY: all test gen clean
