# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

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
CMAKE_COMMAND = /share/contrib-modules/Core/cmake/3.30.0/bin/cmake

# The command to remove a file.
RM = /share/contrib-modules/Core/cmake/3.30.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build

# Include any dependencies generated for this target.
include CMakeFiles/GhostExchange.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GhostExchange.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GhostExchange.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GhostExchange.dir/flags.make

CMakeFiles/GhostExchange.dir/GhostExchange.cc.o: CMakeFiles/GhostExchange.dir/flags.make
CMakeFiles/GhostExchange.dir/GhostExchange.cc.o: /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/GhostExchange.cc
CMakeFiles/GhostExchange.dir/GhostExchange.cc.o: CMakeFiles/GhostExchange.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GhostExchange.dir/GhostExchange.cc.o"
	/opt/rocm-6.2.0/bin/amdclang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/GhostExchange.dir/GhostExchange.cc.o -MF CMakeFiles/GhostExchange.dir/GhostExchange.cc.o.d -o CMakeFiles/GhostExchange.dir/GhostExchange.cc.o -c /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/GhostExchange.cc

CMakeFiles/GhostExchange.dir/GhostExchange.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GhostExchange.dir/GhostExchange.cc.i"
	/opt/rocm-6.2.0/bin/amdclang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/GhostExchange.cc > CMakeFiles/GhostExchange.dir/GhostExchange.cc.i

CMakeFiles/GhostExchange.dir/GhostExchange.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GhostExchange.dir/GhostExchange.cc.s"
	/opt/rocm-6.2.0/bin/amdclang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/GhostExchange.cc -o CMakeFiles/GhostExchange.dir/GhostExchange.cc.s

CMakeFiles/GhostExchange.dir/timer.c.o: CMakeFiles/GhostExchange.dir/flags.make
CMakeFiles/GhostExchange.dir/timer.c.o: /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/timer.c
CMakeFiles/GhostExchange.dir/timer.c.o: CMakeFiles/GhostExchange.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/GhostExchange.dir/timer.c.o"
	/opt/rocm-6.2.0/bin/amdclang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GhostExchange.dir/timer.c.o -MF CMakeFiles/GhostExchange.dir/timer.c.o.d -o CMakeFiles/GhostExchange.dir/timer.c.o -c /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/timer.c

CMakeFiles/GhostExchange.dir/timer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/GhostExchange.dir/timer.c.i"
	/opt/rocm-6.2.0/bin/amdclang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/timer.c > CMakeFiles/GhostExchange.dir/timer.c.i

CMakeFiles/GhostExchange.dir/timer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/GhostExchange.dir/timer.c.s"
	/opt/rocm-6.2.0/bin/amdclang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/timer.c -o CMakeFiles/GhostExchange.dir/timer.c.s

CMakeFiles/GhostExchange.dir/malloc2D.c.o: CMakeFiles/GhostExchange.dir/flags.make
CMakeFiles/GhostExchange.dir/malloc2D.c.o: /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/malloc2D.c
CMakeFiles/GhostExchange.dir/malloc2D.c.o: CMakeFiles/GhostExchange.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/GhostExchange.dir/malloc2D.c.o"
	/opt/rocm-6.2.0/bin/amdclang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/GhostExchange.dir/malloc2D.c.o -MF CMakeFiles/GhostExchange.dir/malloc2D.c.o.d -o CMakeFiles/GhostExchange.dir/malloc2D.c.o -c /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/malloc2D.c

CMakeFiles/GhostExchange.dir/malloc2D.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/GhostExchange.dir/malloc2D.c.i"
	/opt/rocm-6.2.0/bin/amdclang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/malloc2D.c > CMakeFiles/GhostExchange.dir/malloc2D.c.i

CMakeFiles/GhostExchange.dir/malloc2D.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/GhostExchange.dir/malloc2D.c.s"
	/opt/rocm-6.2.0/bin/amdclang $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/malloc2D.c -o CMakeFiles/GhostExchange.dir/malloc2D.c.s

# Object files for target GhostExchange
GhostExchange_OBJECTS = \
"CMakeFiles/GhostExchange.dir/GhostExchange.cc.o" \
"CMakeFiles/GhostExchange.dir/timer.c.o" \
"CMakeFiles/GhostExchange.dir/malloc2D.c.o"

# External object files for target GhostExchange
GhostExchange_EXTERNAL_OBJECTS =

GhostExchange: CMakeFiles/GhostExchange.dir/GhostExchange.cc.o
GhostExchange: CMakeFiles/GhostExchange.dir/timer.c.o
GhostExchange: CMakeFiles/GhostExchange.dir/malloc2D.c.o
GhostExchange: CMakeFiles/GhostExchange.dir/build.make
GhostExchange: /share/contrib-modules/openmpi/ompi5.0.3-ucc1.3.x-ucx1.16.x-rocm6.2.0/lib/libmpi.so
GhostExchange: CMakeFiles/GhostExchange.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable GhostExchange"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GhostExchange.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GhostExchange.dir/build: GhostExchange
.PHONY : CMakeFiles/GhostExchange.dir/build

CMakeFiles/GhostExchange.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GhostExchange.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GhostExchange.dir/clean

CMakeFiles/GhostExchange.dir/depend:
	cd /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2 /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2 /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build /home/gcapodag/repos/HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/Orig2/build/CMakeFiles/GhostExchange.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/GhostExchange.dir/depend

