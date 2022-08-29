# Build settings
export CC := $(shell which clang)
export CXX := $(shell which clang++)
BUILD_DIR := ./build
SRC_DIR := ./src
BUILD_COMMAND := cmake
BUILD_TYPE := Release
TARGET_EXEC := $(BUILD_DIR)/bin/fledge
CC_FORMAT_EXEC := clang-format
PY_FORMAT_EXEC := autopep8

define HELP_TEXT
Fledge Renderer

Targets:   
	- build:  Build all
	- clean:  Clean all
	- run:    Run with the default setting
	- format: Use autopep8 and clang-format to format
	- test:   Run all tests
	- help:   Print helper message
endef

# Prerequisites
OS = $(shell uname -s)
SHELL = /usr/bin/bash
ifeq ($(OS),Windows_NT)
	@echo "Currently this makefile cannot operate on windows"
endif

.PHONY: build
build: .FORCE
	@$(BUILD_COMMAND) -B $(BUILD_DIR) -DCMAKE_EXPORT_COMPILE_COMMANDS=True -GNinja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	@$(BUILD_COMMAND) --build $(BUILD_DIR)
	@ln -sf $(BUILD_DIR)/compile_commands.json ./  

.PHONY: clean
clean: .FORCE
	@$(BUILD_COMMAND) --build $(BUILD_DIR) --target clean

.PHONY: run
run:
	@$(TARGET_EXEC)

.PHONY: format
format: .FORCE
	@$(CC_FORMAT_EXEC) -i $(shell find -type f -regextype posix-extended -regex '$(SRC_DIR).*\.([c|h]p*|(ispc)|(inc))')
	@$(PY_FORMAT_EXEC) --in-place --recursive "$(SRC_DIR)"

.PHONY: test
test: build .FORCE
	@ctest --test-dir $(BUILD_DIR)

.PHONY: help
export HELP_TEXT
help: .FORCE
	@echo "$$HELP_TEXT"
	
.FORCE:
