# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/martin/.local/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/martin/.local/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/martin/Projects/Skola/CUDA/Krcma_program_4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/martin/Projects/Skola/CUDA/Krcma_program_4/build

# Include any dependencies generated for this target.
include CMakeFiles/Krcma_Ukol_4.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Krcma_Ukol_4.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Krcma_Ukol_4.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Krcma_Ukol_4.dir/flags.make

CMakeFiles/Krcma_Ukol_4.dir/main.cu.o: CMakeFiles/Krcma_Ukol_4.dir/flags.make
CMakeFiles/Krcma_Ukol_4.dir/main.cu.o: /home/martin/Projects/Skola/CUDA/Krcma_program_4/main.cu
CMakeFiles/Krcma_Ukol_4.dir/main.cu.o: CMakeFiles/Krcma_Ukol_4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martin/Projects/Skola/CUDA/Krcma_program_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/Krcma_Ukol_4.dir/main.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Krcma_Ukol_4.dir/main.cu.o -MF CMakeFiles/Krcma_Ukol_4.dir/main.cu.o.d -x cu -c /home/martin/Projects/Skola/CUDA/Krcma_program_4/main.cu -o CMakeFiles/Krcma_Ukol_4.dir/main.cu.o

CMakeFiles/Krcma_Ukol_4.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Krcma_Ukol_4.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Krcma_Ukol_4.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Krcma_Ukol_4.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.o: CMakeFiles/Krcma_Ukol_4.dir/flags.make
CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.o: /home/martin/Projects/Skola/CUDA/Krcma_program_4/image_rgb.cu
CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.o: CMakeFiles/Krcma_Ukol_4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martin/Projects/Skola/CUDA/Krcma_program_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.o -MF CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.o.d -x cu -c /home/martin/Projects/Skola/CUDA/Krcma_program_4/image_rgb.cu -o CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.o

CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.o: CMakeFiles/Krcma_Ukol_4.dir/flags.make
CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.o: /home/martin/Projects/Skola/CUDA/Krcma_program_4/histogram_generator.cu
CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.o: CMakeFiles/Krcma_Ukol_4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martin/Projects/Skola/CUDA/Krcma_program_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.o"
	/usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.o -MF CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.o.d -x cu -c /home/martin/Projects/Skola/CUDA/Krcma_program_4/histogram_generator.cu -o CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.o

CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.o: CMakeFiles/Krcma_Ukol_4.dir/flags.make
CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.o: /home/martin/Projects/Skola/CUDA/Krcma_program_4/argument_parser.cpp
CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.o: CMakeFiles/Krcma_Ukol_4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martin/Projects/Skola/CUDA/Krcma_program_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.o -MF CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.o.d -o CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.o -c /home/martin/Projects/Skola/CUDA/Krcma_program_4/argument_parser.cpp

CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/martin/Projects/Skola/CUDA/Krcma_program_4/argument_parser.cpp > CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.i

CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/martin/Projects/Skola/CUDA/Krcma_program_4/argument_parser.cpp -o CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.s

CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.o: CMakeFiles/Krcma_Ukol_4.dir/flags.make
CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.o: /home/martin/Projects/Skola/CUDA/Krcma_program_4/utils/pngio.cpp
CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.o: CMakeFiles/Krcma_Ukol_4.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/martin/Projects/Skola/CUDA/Krcma_program_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.o -MF CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.o.d -o CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.o -c /home/martin/Projects/Skola/CUDA/Krcma_program_4/utils/pngio.cpp

CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/martin/Projects/Skola/CUDA/Krcma_program_4/utils/pngio.cpp > CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.i

CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/martin/Projects/Skola/CUDA/Krcma_program_4/utils/pngio.cpp -o CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.s

# Object files for target Krcma_Ukol_4
Krcma_Ukol_4_OBJECTS = \
"CMakeFiles/Krcma_Ukol_4.dir/main.cu.o" \
"CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.o" \
"CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.o" \
"CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.o" \
"CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.o"

# External object files for target Krcma_Ukol_4
Krcma_Ukol_4_EXTERNAL_OBJECTS =

Krcma_Ukol_4: CMakeFiles/Krcma_Ukol_4.dir/main.cu.o
Krcma_Ukol_4: CMakeFiles/Krcma_Ukol_4.dir/image_rgb.cu.o
Krcma_Ukol_4: CMakeFiles/Krcma_Ukol_4.dir/histogram_generator.cu.o
Krcma_Ukol_4: CMakeFiles/Krcma_Ukol_4.dir/argument_parser.cpp.o
Krcma_Ukol_4: CMakeFiles/Krcma_Ukol_4.dir/utils/pngio.cpp.o
Krcma_Ukol_4: CMakeFiles/Krcma_Ukol_4.dir/build.make
Krcma_Ukol_4: /usr/lib/x86_64-linux-gnu/libpng.so
Krcma_Ukol_4: CMakeFiles/Krcma_Ukol_4.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/martin/Projects/Skola/CUDA/Krcma_program_4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable Krcma_Ukol_4"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Krcma_Ukol_4.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Krcma_Ukol_4.dir/build: Krcma_Ukol_4
.PHONY : CMakeFiles/Krcma_Ukol_4.dir/build

CMakeFiles/Krcma_Ukol_4.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Krcma_Ukol_4.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Krcma_Ukol_4.dir/clean

CMakeFiles/Krcma_Ukol_4.dir/depend:
	cd /home/martin/Projects/Skola/CUDA/Krcma_program_4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/martin/Projects/Skola/CUDA/Krcma_program_4 /home/martin/Projects/Skola/CUDA/Krcma_program_4 /home/martin/Projects/Skola/CUDA/Krcma_program_4/build /home/martin/Projects/Skola/CUDA/Krcma_program_4/build /home/martin/Projects/Skola/CUDA/Krcma_program_4/build/CMakeFiles/Krcma_Ukol_4.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Krcma_Ukol_4.dir/depend

