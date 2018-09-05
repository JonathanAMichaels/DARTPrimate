#include "realsense_depth_source.h"

#include <iostream>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

namespace dart {

RealSenseDepthSource::RealSenseDepthSource() : DepthSource<ushort,uchar3>() {
     // Declare a texture for the depth image on the GPU
    texture depth_image;

    // Declare frameset and frames which will hold the data from the camera
    rs2::frameset frames;
    rs2::frame depth;

     // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    rs2::config cfg;
}

RealSenseDepthSource::~RealSenseDepthSource() {

    
    pipe->stop();

}

bool RealSenseDepthSource::initialize(const bool isLive) {

    _isLive = isLive;
  
    // Create booleans to control GUI (recorded - allow play button, recording - show 'recording to file' text)
    bool recorded = false;
    bool recording = false;

   
    cfg.enable_stream(RS2_STREAM_DEPTH); // Enable default depth
    auto profile = pipe.start(cfg);

    auto sensor = profile.get_device().first<rs2::depth_sensor>();
    
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
    rs2::device device = pipe->get_active_profile().get_device();


    // If the device is sreaming live and not from a file
    if (!device.as<rs2::playback>())
    {
        frames = pipe->wait_for_frames(); // wait for next set of frames from the camera
        depth = frames.get_depth_frame(); // Find the depth data
    }


    if (_isLive) {



        _depthWidth = stream.width();
        _depthHeight = stream.height();

    } else {


        _depthWidth = depthMode.getResolutionX();
        _depthHeight = depthMode.getResolutionY();

    }

    _focalLength = make_float2(525.*_depthWidth/640,525.*_depthWidth/640);
    _principalPoint = make_float2(_depthWidth/2,_depthHeight/2);

    cudaMalloc(&_deviceDepth,_depthWidth*_depthHeight*sizeof(ushort));

    _hasTimestamps = true;
    _frame = 0;

   // _frameIndexOffset = _depthFrame.getFrameIndex();


}

const ushort * RealSenseDepthSource::getDepth() const {
    return (ushort *)_depthFrame.getData();
}

const ushort * RealSenseDepthSource::getDeviceDepth() const {
    return (ushort *)_deviceDepth;
}

const uchar3 * RealSenseDepthSource::getColor() const {
    return (uchar3 *)_colorFrame.getData();
}

uint64_t RealSenseDepthSource::getDepthTime() const {
    return _depthFrame.getTimestamp();
}

uint64_t RealSenseDepthSource::getColorTime() const {
    return _colorFrame.getTimestamp();
}

void RealSenseDepthSource::setFrame(const uint frame) {

    if (_isLive)
        return;

    RealSense::PlaybackControl * pc = _device.getPlaybackControl();

    pc->seek(_depthStream,frame + _frameIndexOffset);

    advance();

}

void RealSenseDepthSource::advance() {

    _depthStream.readFrame(&_depthFrame);
    _frame = _depthFrame.getFrameIndex() - _frameIndexOffset;
    if (_hasColor) {
        _colorStream.readFrame(&_colorFrame);
    }

    cudaMemcpy(_deviceDepth,_depthFrame.getData(),_depthWidth*_depthHeight*sizeof(ushort),cudaMemcpyHostToDevice);

}

}
