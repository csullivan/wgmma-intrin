# Define the compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++11 -O2
TARGET = layout_transform

# Files
SRC = test_layout_transform_general.cu
OBJ = test_layout_transform.o

# Default rule
all: $(TARGET)

# Compile the CUDA source file to an object file
$(OBJ): $(SRC)
	$(NVCC) $(CXXFLAGS) -c $(SRC) -o $(OBJ)

# Link the object file to create the executable
$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $(TARGET)

# Clean up the build files
clean:
	rm -f $(OBJ) $(TARGET)
