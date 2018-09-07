#ifndef REALSENSE_DEPTH_SOURCE_H
#define REALSENSE_DEPTH_SOURCE_H

#include "depth_source.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <vector_types.h>
#include "util/mirrored_memory.h"

namespace dart {

template <typename DepthType, typename ColorType>
class RealSenseDepthSource : public DepthSource<ushort,uchar3> {
public:
    RealSenseDepthSource();
    ~RealSenseDepthSource();

    bool initialize(const bool isLive = true);

    const DepthType * getDepth() const { return _depthData->hostPtr(); }
    
    const DepthType * getDeviceDepth() const { return _depthData->devicePtr(); }

  //  const uchar3 * getColor() const;

    ColorLayout getColorLayout() const { return LAYOUT_RGB; }

    uint64_t getDepthTime() const;

//    uint64_t getColorTime() const;

    void setFrame(const uint frame);

    void advance();

    bool hasRadialDistortionParams() const { return false; }

    float getScaleToMeters() const { return _scaleToMeters; }

private:
    std::shared_ptr<rs2::pipeline> pipe;

    

    rs2::frameset frames;
    rs2::frame depth;
    float _scaleToMeters;
    float _depth_scale;
    MirroredVector<DepthType> * _depthData;
};

// IMPLEMENTATION

template <typename DepthType, typename ColorType>
RealSenseDepthSource<DepthType,ColorType>::RealSenseDepthSource() : DepthSource<ushort,uchar3>() {

}

template <typename DepthType, typename ColorType>
RealSenseDepthSource<DepthType,ColorType>::~RealSenseDepthSource() {
    pipe->stop();
    delete _depthData;
}

template <typename DepthType, typename ColorType>
bool RealSenseDepthSource<DepthType,ColorType>::initialize(const bool isLive) {

    this->_frame = 0;
    this->_isLive = isLive;

    rs2::config cfg;
    if (this->_isLive)
    {
        cfg.enable_stream(RS2_STREAM_DEPTH); // Enable default depth
    }
    else
    {
        //rs2::playback playback = device.as<rs2::playback>();
        cfg.enable_device_from_file("/home/jonathan/Downloads/mocap-high-res.bag");
    }
    std::cout << "0" << std::endl;
    pipe = std::make_shared<rs2::pipeline>();
    std::cout << "1" << std::endl;
    auto profile = pipe->start(cfg); // File will be opened in read mode at this point
    std::cout << "2" << std::endl;
    rs2::device device = pipe->get_active_profile().get_device();  
    auto sensor = profile.get_device().first<rs2::depth_sensor>();

    this->_depth_scale = rs2::context().query_devices().front()
        .query_sensors().front().get_option(RS2_OPTION_DEPTH_UNITS);

    this->_scaleToMeters = (float)this->_depth_scale;
    
    auto stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics(); // Calibration data

    // If the device is sreaming live and not from a file
    if (this->_isLive)
    {
        frames = pipe->wait_for_frames(); // wait for next set of frames from the camera
    }
    else
    {
        pipe->poll_for_frames(&frames);
    }
    depth = frames.get_depth_frame(); // Find the depth data

    this->_depthWidth = stream.width();
    this->_depthHeight = stream.height();


    this->_focalLength = make_float2(intrinsics.fx,intrinsics.fy);
    this->_principalPoint = make_float2(intrinsics.ppx,intrinsics.ppy);


    _depthData = new MirroredVector<DepthType>(this->_depthWidth*this->_depthHeight);
    
      
    auto pixels = (uint16_t*)depth.get_data();    
    auto yy = this->_depthHeight-1;
    for (auto y = 0; y < this->_depthHeight; y++)
    {
        auto xx = this->_depthWidth-1;
        for (auto x = 0; x < this->_depthWidth; x++)
        {
            _depthData->hostPtr()[this->_depthWidth*y + x] = pixels[this->_depthWidth*yy + xx];
            xx -= 1;
        }
        yy -= 1;
    }

    _depthData->syncHostToDevice();

    return true;
}

template <typename DepthType, typename ColorType>
uint64_t RealSenseDepthSource<DepthType,ColorType>::getDepthTime() const {
    return depth.get_timestamp();
}

template <typename DepthType, typename ColorType>
void RealSenseDepthSource<DepthType,ColorType>::setFrame(const uint frame) {

    if (_isLive)
        return;
    //advance();

}

template <typename DepthType, typename ColorType>
void RealSenseDepthSource<DepthType,ColorType>::advance() {

    this->_frame++;

        // If the device is sreaming live and not from a file
    if (this->_isLive)
    {
        frames = pipe->wait_for_frames(); // wait for next set of frames from the camera
    }
    else
    {
        pipe->poll_for_frames(&frames);
    }
    depth = frames.get_depth_frame(); // Find the depth data
    
    auto pixels = (uint16_t*)depth.get_data();    
    auto yy = this->_depthHeight-1;
    for (auto y = 0; y < this->_depthHeight; y++)
    {
        auto xx = this->_depthWidth-1;
        for (auto x = 0; x < this->_depthWidth; x++)
        {
            _depthData->hostPtr()[this->_depthWidth*y + x] = pixels[this->_depthWidth*yy + xx];
            xx -= 1;
        }
        yy -= 1;
    }

    _depthData->syncHostToDevice();
}



}

#endif // REALSENSE_DEPTH_SOURCE_H
