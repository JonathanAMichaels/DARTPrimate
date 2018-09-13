#ifndef KINECTV2_DEPTH_SOURCE_H
#define KINECTV2_DEPTH_SOURCE_H

#include "depth_source.h"
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <vector_types.h>
#include "util/mirrored_memory.h"

namespace dart {

template <typename DepthType, typename ColorType>
class KinectV2DepthSource : public DepthSource<ushort,uchar3> {
public:
    KinectV2DepthSource();
    ~KinectV2DepthSource();

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
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;
    libfreenect2::Frame *depth;
    libfreenect2::Frame *rgb;
    libfreenect2::FrameMap frames;
    libfreenect2::SyncMultiFrameListener* listener;
    libfreenect2::Registration* registration;

    float _scaleToMeters;
    MirroredVector<DepthType> * _depthData;
};

// IMPLEMENTATION

template <typename DepthType, typename ColorType>
KinectV2DepthSource<DepthType,ColorType>::KinectV2DepthSource() : DepthSource<ushort,uchar3>() {

}

template <typename DepthType, typename ColorType>
KinectV2DepthSource<DepthType,ColorType>::~KinectV2DepthSource() {
    dev->stop();
    dev->close();
    delete _depthData;
}

template <typename DepthType, typename ColorType>
bool KinectV2DepthSource<DepthType,ColorType>::initialize(const bool isLive) {

    this->_frame = 0;
    this->_isLive = isLive;
    
    if(freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }
    auto serial = freenect2.getDefaultDeviceSerialNumber();
    pipeline = new libfreenect2::CudaPacketPipeline();
    dev = freenect2.openDevice(serial, pipeline);
  
    int types = 0;
    types |= libfreenect2::Frame::Color | libfreenect2::Frame::Depth;
    listener = new libfreenect2::SyncMultiFrameListener(types);

    dev->setColorFrameListener(listener);
    dev->setIrAndDepthFrameListener(listener);

    if (!dev->startStreams(true, true))
      return -1;
    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

    auto intrinsics = dev->getIrCameraParams();

    this->_scaleToMeters = (float)1.0/(float)1000.0;
  
    // If the device is sreaming live and not from a file
    if (this->_isLive)
    {
        if (!listener->waitForNewFrame(frames, 10*1000)) // 10 seconds
        {
            std::cout << "timeout!" << std::endl;
        }
        depth = frames[libfreenect2::Frame::Depth];
    }
    else
    {
      //  pipe->poll_for_frames(&frames);
    }
    this->_depthWidth = depth->width;
    this->_depthHeight = depth->height;

    this->_focalLength = make_float2(intrinsics.fx,intrinsics.fy);
    this->_principalPoint = make_float2(intrinsics.cx,intrinsics.cy);


    _depthData = new MirroredVector<DepthType>(this->_depthWidth*this->_depthHeight);
    
    registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
   
    rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);
    registration->apply(rgb, depth, &undistorted, &registered);

      
    auto pixels = (float*)undistorted.data;    
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
    listener->release(frames);

    return true;
}

template <typename DepthType, typename ColorType>
uint64_t KinectV2DepthSource<DepthType,ColorType>::getDepthTime() const {
    std::cout << "timestamp" << std::endl;
    return depth->timestamp;
}

template <typename DepthType, typename ColorType>
void KinectV2DepthSource<DepthType,ColorType>::setFrame(const uint frame) {

    if (_isLive)
        return;
    //advance();

}

template <typename DepthType, typename ColorType>
void KinectV2DepthSource<DepthType,ColorType>::advance() {
    this->_frame++;
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);
    // If the device is sreaming live and not from a file
    if (this->_isLive)
    {
        if (!listener->waitForNewFrame(frames, 10*1000)) // 10 seconds
        {
            std::cout << "timeout!" << std::endl;
        }
        depth = frames[libfreenect2::Frame::Depth];
        rgb = frames[libfreenect2::Frame::Color];     
        registration->apply(rgb, depth, &undistorted, &registered);
    }
    else
    {
      //  pipe->poll_for_frames(&frames);
    }

    auto pixels = (float*)undistorted.data;    
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
    listener->release(frames);
}



}

#endif // KINECTV2_DEPTH_SOURCE_H
