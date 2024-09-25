# Define the compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++14 -O2 -arch=sm_90a
INCLUDES = -I.

# Targets
TARGETS = test_wgmma test_layout_transform_general test_wgmma_from_regs

# Header files
WGMMA_HEADER = wgmma.cuh
TRANSFORM_HEADER = layout_transform.cuh

# Default rule
all: $(TARGETS)

# Rule for test_wgmma
test_wgmma: test_single_instance_wgmma.cu $(WGMMA_HEADER) $(TRANSFORM_HEADER)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@

# Rule for test_layout_transform_general
test_layout_transform_general: test_layout_transform_general.cu $(TRANSFORM_HEADER)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@

# Rule for test_wgmma_from_regs
test_wgmma_from_regs: test_wgmma_from_regs.cu $(WGMMA_HEADER) $(TRANSFORM_HEADER)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@

# Clean up the build files
clean:
	rm -f $(TARGETS)

.PHONY: all clean