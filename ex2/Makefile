.PHONY: all clean fclean re execs objs

# Directories
SRC_DIR = src
INC_DIR = inc
BIN_DIR = bin
BUILD_DIR = build

# Compilation
CC 		= mpicc
CFLAGS 	= -g -Wall -Wextra -I$(INC_DIR) -O3
# -D_POSIX_C_SOURCE=200809L 

# Commands
RM = rm -rf

# Explicitly define files
EXEC_NAME = main
# OBJS = main.o

# Automatically find all source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)

# Targets
TARGET_EXEC = $(EXEC_NAME:%=$(BIN_DIR)/%)
EXECUTABLES = $(patsubst $(SRC_DIR)/%.c,$(BIN_DIR)/%,$(SRC_FILES)) # to create executables directly for every .c
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRC_FILES))# one object file for every .c file

all: $(TARGET_EXEC)

debug: CFLAGS += -DDEBUG
debug: clean all

execs: $(EXECUTABLES)

objs: $(OBJS)

$(TARGET_EXEC): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OBJS) -o $@

# Build each executable from its corresponding .o
$(BIN_DIR)/%: $(BUILD_DIR)/%.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@

# Pattern rule for objects, built from their .c files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create build directory if missing 
$(BUILD_DIR):
	mkdir -p $@

# Create bin directory if missing 
$(BIN_DIR):
	mkdir -p $@

# Delete all binaries
clean:
	$(RM) $(BUILD_DIR)/* $(BIN_DIR)/* 

# Delete all built directories
fclean:
	$(RM) $(BUILD_DIR) $(BIN_DIR)

# Delete all binaries and re-build
re: clean all
# 	$(MAKE) all