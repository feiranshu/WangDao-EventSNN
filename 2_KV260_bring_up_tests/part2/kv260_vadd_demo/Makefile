TARGET ?= hw
PLATFORM ?=
VPP ?= v++
CXX ?= g++

BUILD_DIR := build/$(TARGET)
XO := $(BUILD_DIR)/krnl_vadd.xo
XCLBIN := $(BUILD_DIR)/kv260_vadd.xclbin
HOST := build/kv260_vadd

.PHONY: all xclbin host-native clean check-platform

all: xclbin

check-platform:
	@test -n "$(PLATFORM)" || (echo "ERROR: 请先设置 PLATFORM=/absolute/path/to/xilinx_kv260_base_202510_1.xpfm"; exit 1)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(XO): kernel/krnl_vadd.cpp | $(BUILD_DIR) check-platform
	$(VPP) -c -t $(TARGET) --platform $(PLATFORM) -k krnl_vadd --save-temps -o $@ $<

$(XCLBIN): $(XO) link.cfg check-platform
	$(VPP) -l -t $(TARGET) --platform $(PLATFORM) --config link.cfg --save-temps -o $@ $(XO)

xclbin: $(XCLBIN)

host-native: host/kv260_vadd.cpp
	mkdir -p build
	$(CXX) -std=c++17 -O2 -Wall -Wextra $< -o $(HOST) -lxrt_coreutil -pthread

clean:
	rm -rf build .Xil _x logs *.log *.jou
