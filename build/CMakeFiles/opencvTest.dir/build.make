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
CMAKE_COMMAND = /home/nano/cmake-3.30.0-rc3-linux-aarch64/bin/cmake

# The command to remove a file.
RM = /home/nano/cmake-3.30.0-rc3-linux-aarch64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nano/bru

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nano/bru/build

# Include any dependencies generated for this target.
include CMakeFiles/opencvTest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/opencvTest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/opencvTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencvTest.dir/flags.make

CMakeFiles/opencvTest.dir/main.cpp.o: CMakeFiles/opencvTest.dir/flags.make
CMakeFiles/opencvTest.dir/main.cpp.o: /home/nano/bru/main.cpp
CMakeFiles/opencvTest.dir/main.cpp.o: CMakeFiles/opencvTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/nano/bru/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencvTest.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencvTest.dir/main.cpp.o -MF CMakeFiles/opencvTest.dir/main.cpp.o.d -o CMakeFiles/opencvTest.dir/main.cpp.o -c /home/nano/bru/main.cpp

CMakeFiles/opencvTest.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/opencvTest.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nano/bru/main.cpp > CMakeFiles/opencvTest.dir/main.cpp.i

CMakeFiles/opencvTest.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/opencvTest.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nano/bru/main.cpp -o CMakeFiles/opencvTest.dir/main.cpp.s

# Object files for target opencvTest
opencvTest_OBJECTS = \
"CMakeFiles/opencvTest.dir/main.cpp.o"

# External object files for target opencvTest
opencvTest_EXTERNAL_OBJECTS =

opencvTest: CMakeFiles/opencvTest.dir/main.cpp.o
opencvTest: CMakeFiles/opencvTest.dir/build.make
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_alphamat.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudabgsegm.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudafeatures2d.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudaobjdetect.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudastereo.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_intensity_transform.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_mcc.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_rapid.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_sfm.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_signal.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_wechat_qrcode.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_xfeatures2d.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudacodec.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudaoptflow.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudalegacy.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudawarping.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudaimgproc.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudafilters.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudaarithm.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.10.0
opencvTest: /usr/lib/aarch64-linux-gnu/libopencv_cudev.so.4.10.0
opencvTest: CMakeFiles/opencvTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/nano/bru/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable opencvTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencvTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencvTest.dir/build: opencvTest
.PHONY : CMakeFiles/opencvTest.dir/build

CMakeFiles/opencvTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencvTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencvTest.dir/clean

CMakeFiles/opencvTest.dir/depend:
	cd /home/nano/bru/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nano/bru /home/nano/bru /home/nano/bru/build /home/nano/bru/build /home/nano/bru/build/CMakeFiles/opencvTest.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/opencvTest.dir/depend

