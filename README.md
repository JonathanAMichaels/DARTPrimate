DARTPrimate: Dense Articulated Real-time Tracking of Primates
=======

DARTPrimate is a C++ library for tracking arbitrary articulated models with a depth camera. It achieves real-time performance with the aid of a highly parallel CUDA 
implementation and state-of-the-art GPUs.

**Note:** This build is intended for Ubuntu 16.04.

Required Dependencies
---------------------

**CUDA 8.0:** Download Cuda 8.0 [here](https://developer.nvidia.com/cuda-80-ga2-download-archive) and follow installation instructions. Be sure to install the CUDA examples as well. Installing Cuda will require an nvidia graphics driver to be installed. We recommend first installing the proprietary drivers as follows instead of the graphics drivers included with Cuda:

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update
    sudo apt install nvidia-387

The exact version may depend on your card, but nvidia-387 is a safe choice for most popular cards.

**Eigen 3:** sudo apt-get install libeigen3-dev

**GNU libmatheval:** sudo apt-get install libmatheval-dev

**tinyxml:** sudo apt-get install libtinyxml-dev

**GLUT:** sudo apt-get install freeglut3-dev

**Pangolin [necessary for the GUI]:** Follow the instructions [here](https://github.com/stevenlovegrove/Pangolin)

**OpenCV (3.4.3):** Follow the instructions [here](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html). After creating the opencv and opencv_contrib directories, enter each one and manually specify the version:

    git checkout 3.4.3

At the cmake step, use the following command, replacing -DCUDA_ARCH_BIN=6.1 with the compute capability of your graphics card. For example, the compute capability of a GTX 1080 is 6.1, and the command should be:

	cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=ON \
    -D WITH_OPENMP=ON \
    -D WITH_IPP=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D WITH_CSTRIPES=ON \
    -D WITH_OPENCL=OFF \
    -D WITH_CUDA=ON \
    -D WITH_GTK=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -DCUDA_GENERATION="" \
    -DCUDA_ARCH_BIN=6.1 \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    ..


Optional Dependencies
---------------------

**libfreenect2 [necessary for Kinect v2 support]:** Follow the instructions [here](https://github.com/OpenKinect/libfreenect2)

**Open Asset Import Library [for mesh models]:** sudo apt-get install libassimp-dev

**KinectFusion [for creating meshes]:** A modified version supporting Kinect can be obtained [here](https://github.com/JonathanAMichaels/KinectFusionApp). *Note:* OpenCV must be installed with CUDA support, as described in the above section.


Installation
------------

	cd [DARTPrimate directory]
	mkdir build
	cd build
cmake should be run with the -DCC flag that matches the compute capability of your graphics card. For example, the compute capability of a GTX 1080 is 6.1, so the command should be:
	
	cmake -DCC="61" ..
then:

	make
	cd ../Application
	mkdir build
	cd build
	cmake -DCC="61" ..
	make

Example usage
------------

To run the application, navigate to the [DARTPrimate directory]/Application/build and execute

	./DARTPrimate ../config.toml
You must always include a path to a config file, in which all relevant parameters for the session are specified.

By default, the application is configured to use the Kinect v2. If you want to use an Intel RealSense device, the appropriate lines must be commented and changed in DARTPrimate.cpp before compiling the application.

Notes on using the library
------------

- Frames vs. Joints vs. SDFs: These are three separate but related ways in which
parts of a DART model can be referenced. A frame is a frame of reference in the
kinematic chain, a joint is a connection between two frames with a single degree
of freedom, and a signed distance function (SDF) implicitly stores all geometry
attached to a single frame. Every model has at least one frame (the root) but
need not have any joints or signed distance functions. Because loops are not
allowed, a model with N joints will have N+1 frames (and N+6 degrees of 
freedom). The number of SDFs is at most equal to the number of frames, but may
be less if there are frames with no geometry attached to them. Functions in the
Model class and subclasses that require indexing part of the model will indicate
in the parameter name whether the index is by joint, by frame, or by SDF.

Model file format
------------

DART models are stored as XML files which define the kinematic and geometric 
structure, optionally reference other mesh files to further describe the 
geometry. All models open with the "model" tag which has a single attribute, 
"version" describing the version of the DART XML format (current 1), like so:

    <model version ="1">
      [model here]
    </model>

The model can then optionally specify a number of parameters using the "param" 
tag, with attributes "name" (string) and "value" (floating point), like so:

    <param name="armLength" value="1.5"/>

These parameters can then be referenced when defining sizes, positions, or 
orientations, as described below. Parameters are useful if the same value 
appears multiple times in your specification (as is often the case) or if you 
would like to set the parameter values programatically.

After defining parameters, the model may contain a number of hierachically 
nested "frame" and "geom" tags, which specify new rigid body frames of reference 
or geometric objects, respectively. The "frame" tag requires four attributes,
"jointName" (string), "jointType" (currently accepts either "rotational" or
"prismatic", "jointMin" (floating point), and "jointMax" (floating point), the
last two of which define the joint limits. Additionally, the frame tag requires
three nested tags, "position", "orientation", and "axis", each of which require
three floating point attributes, "x", "y", and "z". An example might look like
this:

	<frame jointName="leftElbow" jointType="rotational" jointMin="0" jointMax="3.1416">
	    <position x="0" y="0" z="1.5" />
	    <orientation x="0" y="0" z="1.5708" />
	    <axis x="1" y="0" z="0"/>
	    [frame children here]
	</frame>

This snippet defines a new frame of reference relative to its parent (the XML
node directly above it in the hierarchy, or the root if the parent is the 
"model" tag). The transform from this frame of reference to the world is given
by:

T_w,f = T_w,p*Trans*R_z*R_y*R_x*R_axis(theta)

where T_w,p gives the transform from the parent to the world, Trans is a
translation-only transform given by the "position" tag, R_z, R_y, and R_x are
rotations about the z, y, and x axes (i.e. Euler angles) given by the
corresponding entries in the "orientation" tag, and R_axis is a rotation by
theta around the axis defined by the "axis" tag, with theta being given by the
articulated pose of the model.

Finally, geometry can be rigidly attached to any frame of reference in the model
by nesting a "geom" tag within a "frame" tag (or within the "model" tag for root
geometry). The geometry tag requires 13 attributes: "type" (currently accepts
"sphere","cylinder","cube", or "mesh"), "sx", "sy", and "sz", which define the
scaling of the geometry, "tx", "ty", and "tz", which define the translation of
the geometry root relative to the rigid body frame of reference, "rx", "ry" and
"rz", which define the orientation relative to the rigid body frame of reference
(also in Euler angles, as with the "frame" tag), and "red", "green" and "blue",
which define the geometry color, which is not used for tracking but will affect
how the model is rendered for debugging purposes. If "type" is set to "mesh",
there is one final attribute, "meshFile", which gives the location of the mesh
file, relative to the location of the XML file.

