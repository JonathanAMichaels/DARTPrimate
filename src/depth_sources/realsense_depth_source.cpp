#include "realsense_depth_source.h"

#include <iostream>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

namespace dart {

RealSenseDepthSource::RealSenseDepthSource() : DepthSource<ushort,uchar3>() {
    std::cout << "-1" << std::endl;
}

RealSenseDepthSource::~RealSenseDepthSource() {
    pipe.stop();
    if (_deviceDepth) {
        cudaFree(_deviceDepth);
    }
}

bool RealSenseDepthSource::initialize(const bool isLive) {

    _isLive = true

    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_DEPTH); // Enable default depth


    auto profile = pipe.start(cfg);
    auto sensor = profile.get_device().first<rs2::depth_sensor>();
    auto depth_scale = sensor.get_depth_scale();
    // TODO: At the moment the SDK does not offer a closed enum for D400 visual presets
    // (because they keep changing)
    // As a work-around we try to find the High-Density preset by name
    // We do this to reduce the number of black pixels
    // The hardware can perform hole-filling much better and much more power efficient then our software
    auto range = sensor.get_option_range(RS2_OPTION_VISUAL_PRESET);
    for (auto i = range.min; i < range.max; i += range.step)
        if (std::string(sensor.get_option_value_description(RS2_OPTION_VISUAL_PRESET, i)) == "High Density")
            sensor.set_option(RS2_OPTION_VISUAL_PRESET, i);
    auto stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics(); // Calibration data
    // Initialize a shared pointer to a device with the current device on the pipeline
    rs2::device device = pipe.get_active_profile().get_device();

    // If the device is sreaming live and not from a file
    if (!device.as<rs2::playback>())
    {
        frames = pipe.wait_for_frames(); // wait for next set of frames from the camera
        depth = frames.get_depth_frame(); // Find the depth data
    }
    std::cout << "0" << std::endl;
    _depthWidth = stream.width();
    _depthHeight = stream.height();


    _focalLength = make_float2(intrinsics.fx,intrinsics.fy);
    _principalPoint = make_float2(intrinsics.ppx,intrinsics.ppy);
    
    _depthData = (ushort *) (depth.get_data());
    _deviceDepth = _depthData;

    _hasTimestamps = true;
    _frame = 0;


    cudaMalloc(&_deviceDepth,_depthWidth*_depthHeight*sizeof(ushort));
   // _frameIndexOffset = _depthFrame.getFrameIndex();


}

const ushort * RealSenseDepthSource::getDepth() const {
    return _depthData;
}

const ushort * RealSenseDepthSource::getDeviceDepth() const {
    return _deviceDepth;
}

uint64_t RealSenseDepthSource::getDepthTime() const {
    return depth.get_timestamp();
}

void RealSenseDepthSource::setFrame(const uint frame) {

    if (_isLive)
        return;
    //advance();

}

void RealSenseDepthSource::advance() {

    _frame = _frame + 1;

    frames = pipe.wait_for_frames(); // wait for next set of frames from the camera
    depth = frames.get_depth_frame(); // Find the depth data

    _depthData = (ushort *) (depth.get_data());
    cudaMemcpy(_deviceDepth,_depthData,_depthWidth*_depthHeight*sizeof(ushort),cudaMemcpyHostToDevice);
}

}
