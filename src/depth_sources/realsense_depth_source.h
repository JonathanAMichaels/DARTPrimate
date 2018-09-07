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
    rs2::pipeline pipe;
    rs2::frameset frames;
    rs2::frame depth;
    float _scaleToMeters;
    float _depth_scale;
    MirroredVector<DepthType> * _depthData;
};

// IMPLEMENTATION

template <typename DepthType, typename ColorType>
RealSenseDepthSource<DepthType,ColorType>::RealSenseDepthSource() : DepthSource<ushort,uchar3>() {
    std::cout << "-1" << std::endl;
}

template <typename DepthType, typename ColorType>
RealSenseDepthSource<DepthType,ColorType>::~RealSenseDepthSource() {
    pipe.stop();
    delete _depthData;
}

template <typename DepthType, typename ColorType>
bool RealSenseDepthSource<DepthType,ColorType>::initialize(const bool isLive) {

    this->_scaleToMeters = 1;
    this->_frame = 0;
    this->_isLive = isLive;

    rs2::config cfg;
    if (this->_isLive)
    {
        cfg.enable_stream(RS2_STREAM_DEPTH); // Enable default depth
    }
    else
    {
        cfg.enable_device_from_file("/home/jonathan/Documents/test.bag");
    }
    auto profile = pipe.start(cfg);
    auto sensor = profile.get_device().first<rs2::depth_sensor>();

    this->_scaleToMeters = (float)1.0;
    
    // TODO: At the moment the SDK does not offer a closed enum for D400 visual presets
    // (because they keep changing)
    // As a work-around we try to find the High-Density preset by name
    // We do this to reduce the number of black pixels
    // The hardware can perform hole-filling much better and much more power efficient then our software

    //auto range = sensor.get_option_range(RS2_OPTION_VISUAL_PRESET);
    //for (auto i = range.min; i < range.max; i += range.step)
    //    if (std::string(sensor.get_option_value_description(RS2_OPTION_VISUAL_PRESET, i)) == "High Density")
    //        sensor.set_option(RS2_OPTION_VISUAL_PRESET, i);
    auto stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics(); // Calibration data
    // Initialize a shared pointer to a device with the current device on the pipeline
    rs2::device device = pipe.get_active_profile().get_device();

    // If the device is sreaming live and not from a file
    if (this->_isLive)
    {
        frames = pipe.wait_for_frames(); // wait for next set of frames from the camera
    }
    else
    //{
    //    frames = pipe.poll_for_frames();
    //}
    depth = frames.get_depth_frame(); // Find the depth data

    this->_depthWidth = stream.width();
    this->_depthHeight = stream.height();


    this->_focalLength = make_float2(intrinsics.fx,intrinsics.fy);
    this->_principalPoint = make_float2(intrinsics.ppx,intrinsics.ppy);
    
    this->_depth_scale = rs2::context().query_devices().front()
        .query_sensors().front().get_option(RS2_OPTION_DEPTH_UNITS);

    _depthData = new MirroredVector<DepthType>(this->_depthWidth*this->_depthHeight);
    
      
    auto pixels = (uint16_t*)depth.get_data();    
    auto yy = this->_depthHeight-1;
    for (auto y = 0; y < this->_depthHeight; y++)
    {
        auto xx = this->_depthWidth-1;
        for (auto x = 0; x < this->_depthWidth; x++)
        {
            _depthData->hostPtr()[this->_depthWidth*y + x] = pixels[this->_depthWidth*yy + xx] * this->_depth_scale;
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

    if (this->_isLive)
    {
        frames = pipe.wait_for_frames(); // wait for next set of frames from the camera
    }
    //else
    //{
     //   frames = pipe.poll_for_frames();
   // }
    depth = frames.get_depth_frame(); // Find the depth data
    
    auto pixels = (uint16_t*)depth.get_data();    
    auto yy = this->_depthHeight-1;
    for (auto y = 0; y < this->_depthHeight; y++)
    {
        auto xx = this->_depthWidth-1;
        for (auto x = 0; x < this->_depthWidth; x++)
        {
            _depthData->hostPtr()[this->_depthWidth*y + x] = pixels[this->_depthWidth*yy + xx] * this->_depth_scale;
            xx -= 1;
        }
        yy -= 1;
    }

    _depthData->syncHostToDevice();
}



}

#endif // REALSENSE_DEPTH_SOURCE_H
