# Define the compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++14 -O2 -arch=sm_90a
INCLUDES = -I.

# Targets
TARGETS = test_wgmma test_layout_transform_general

# Source files
WGMMA_SRC = midwit-matmul/wgmma_kernel.cu
TRANSFORM_HEADER = layout_transform.cuh

# Object files
WGMMA_OBJ = wgmma_kernel.o

# Default rule
all: $(TARGETS)

# Rule for test_wgmma
test_wgmma: test_wgmma.cu $(WGMMA_OBJ) $(TRANSFORM_HEADER)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< $(WGMMA_OBJ) -o $@

# Rule for test_layout_transform_general
test_layout_transform_general: test_layout_transform_general.cu $(TRANSFORM_HEADER)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@

# Compile the WGMMA kernel source file to an object file
$(WGMMA_OBJ): $(WGMMA_SRC)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Clean up the build files
clean:
	rm -f $(TARGETS) $(WGMMA_OBJ)

.PHONY: all clean