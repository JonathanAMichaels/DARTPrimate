# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jonathan/Desktop/DARTPrimate/librealtest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jonathan/Desktop/DARTPrimate/librealtest/build

# Include any dependencies generated for this target.
include CMakeFiles/rs-capture.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rs-capture.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rs-capture.dir/flags.make

CMakeFiles/rs-capture.dir/rs-capture.cpp.o: CMakeFiles/rs-capture.dir/flags.make
CMakeFiles/rs-capture.dir/rs-capture.cpp.o: ../rs-capture.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jonathan/Desktop/DARTPrimate/librealtest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rs-capture.dir/rs-capture.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rs-capture.dir/rs-capture.cpp.o -c /home/jonathan/Desktop/DARTPrimate/librealtest/rs-capture.cpp

CMakeFiles/rs-capture.dir/rs-capture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rs-capture.dir/rs-capture.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jonathan/Desktop/DARTPrimate/librealtest/rs-capture.cpp > CMakeFiles/rs-capture.dir/rs-capture.cpp.i

CMakeFiles/rs-capture.dir/rs-capture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rs-capture.dir/rs-capture.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jonathan/Desktop/DARTPrimate/librealtest/rs-capture.cpp -o CMakeFiles/rs-capture.dir/rs-capture.cpp.s

CMakeFiles/rs-capture.dir/rs-capture.cpp.o.requires:

.PHONY : CMakeFiles/rs-capture.dir/rs-capture.cpp.o.requires

CMakeFiles/rs-capture.dir/rs-capture.cpp.o.provides: CMakeFiles/rs-capture.dir/rs-capture.cpp.o.requires
	$(MAKE) -f CMakeFiles/rs-capture.dir/build.make CMakeFiles/rs-capture.dir/rs-capture.cpp.o.provides.build
.PHONY : CMakeFiles/rs-capture.dir/rs-capture.cpp.o.provides

CMakeFiles/rs-capture.dir/rs-capture.cpp.o.provides.build: CMakeFiles/rs-capture.dir/rs-capture.cpp.o


# Object files for target rs-capture
rs__capture_OBJECTS = \
"CMakeFiles/rs-capture.dir/rs-capture.cpp.o"

# External object files for target rs-capture
rs__capture_EXTERNAL_OBJECTS =

rs-capture: CMakeFiles/rs-capture.dir/rs-capture.cpp.o
rs-capture: CMakeFiles/rs-capture.dir/build.make
rs-capture: /home/jonathan/anaconda3/lib/libpng.so
rs-capture: /home/jonathan/anaconda3/lib/libz.so
rs-capture: /home/jonathan/Desktop/Pangolin/build/src/libpangolin.so
rs-capture: /usr/local/cuda/lib64/libcudart_static.a
rs-capture: /usr/lib/x86_64-linux-gnu/librt.so
rs-capture: /usr/lib/x86_64-linux-gnu/libGLU.so
rs-capture: /usr/lib/x86_64-linux-gnu/libGL.so
rs-capture: /usr/lib/x86_64-linux-gnu/libGLEW.so
rs-capture: /usr/lib/x86_64-linux-gnu/libSM.so
rs-capture: /usr/lib/x86_64-linux-gnu/libICE.so
rs-capture: /usr/lib/x86_64-linux-gnu/libX11.so
rs-capture: /usr/lib/x86_64-linux-gnu/libXext.so
rs-capture: /usr/lib/x86_64-linux-gnu/librealsense2.so
rs-capture: /usr/lib/libOpenNI.so
rs-capture: /usr/local/lib/libOpenNI2.so
rs-capture: /home/jonathan/anaconda3/lib/libpng.so
rs-capture: /home/jonathan/anaconda3/lib/libz.so
rs-capture: /home/jonathan/anaconda3/lib/libjpeg.so
rs-capture: /home/jonathan/anaconda3/lib/libtiff.so
rs-capture: /usr/lib/x86_64-linux-gnu/libIlmImf.so
rs-capture: CMakeFiles/rs-capture.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jonathan/Desktop/DARTPrimate/librealtest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable rs-capture"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rs-capture.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rs-capture.dir/build: rs-capture

.PHONY : CMakeFiles/rs-capture.dir/build

CMakeFiles/rs-capture.dir/requires: CMakeFiles/rs-capture.dir/rs-capture.cpp.o.requires

.PHONY : CMakeFiles/rs-capture.dir/requires

CMakeFiles/rs-capture.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rs-capture.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rs-capture.dir/clean

CMakeFiles/rs-capture.dir/depend:
	cd /home/jonathan/Desktop/DARTPrimate/librealtest/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jonathan/Desktop/DARTPrimate/librealtest /home/jonathan/Desktop/DARTPrimate/librealtest /home/jonathan/Desktop/DARTPrimate/librealtest/build /home/jonathan/Desktop/DARTPrimate/librealtest/build /home/jonathan/Desktop/DARTPrimate/librealtest/build/CMakeFiles/rs-capture.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rs-capture.dir/depend

