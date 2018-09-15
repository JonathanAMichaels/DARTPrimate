#ifndef DEPTH_SOURCE_H
#define DEPTH_SOURCE_H

#include <stdint.h>
#include <sys/types.h>
#include <vector_types.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>

namespace dart {

enum ColorLayout {
    LAYOUT_RGB, LAYOUT_BGR
};

class DepthSourceBase {

public:
    DepthSourceBase() :
        _hasColor(false),
        _hasTimestamps(false),
        _isLive(false),
        _depthWidth(0),
        _depthHeight(0),
        _colorWidth(0),
        _colorHeight(0),
        _frame(0) { }

    virtual ColorLayout getColorLayout() const { return LAYOUT_RGB; }

    inline bool hasColor() const { return _hasColor; }

    inline bool hasTimestamps() const { return _hasTimestamps; }

    inline bool isLive() const { return _isLive; }

    inline uint getDepthWidth() const { return _depthWidth; }

    inline uint getDepthHeight() const { return _depthHeight; }

    inline uint getColorWidth() const { return _colorWidth; }

    inline uint getColorHeight() const { return _colorHeight; }

    inline float2 getFocalLength() const { return _focalLength; }
    inline void setFocalLength(const float2 focalLength) { _focalLength = focalLength; }

    inline float2 getPrincipalPoint() const { return _principalPoint; }
    inline void setPrincipalPoint(const float2 principalPoint) { _principalPoint = principalPoint; }

    virtual uint64_t getDepthTime() const { return 0; }

    virtual uint64_t getColorTime() const { return 0; }

    virtual uint getFrame() const { return _frame; }

    virtual void setFrame(const uint frame) = 0;

    virtual void advance() = 0;

    virtual bool hasRadialDistortionParams() const = 0;

    virtual const float * getRadialDistortionParams() const { return 0; }

    virtual float getScaleToMeters() const { return 1.0f; }

protected:
    bool _hasColor;
    bool _hasTimestamps;
    bool _isLive;
    uint _depthWidth;
    uint _depthHeight;
    uint _colorWidth;
    uint _colorHeight;
    uint _frame;
    float2 _focalLength;
    float2 _principalPoint;

    bool _record = false;
};

template <typename DepthType, typename ColorType>
class DepthSource : public DepthSourceBase {
public:

    virtual const DepthType * getDepth() const = 0;
    virtual const DepthType * getDeviceDepth() const = 0;

    virtual const ColorType * getColor() const { return 0; }

    void setRecordStatus(const bool record);
    bool saveFrame(const std::string saveDir, const float* depth);
    float* loadFrame(const std::string loadDir);
};



// Implementation

template <typename DepthType, typename ColorType>
void DepthSource<DepthType, ColorType>::setRecordStatus(const bool record) {
    _record = record;
}


template <typename DepthType, typename ColorType>
bool DepthSource<DepthType, ColorType>::saveFrame(const std::string saveDir, const float* depth_frame) {

    cv::Mat depth_image { cv::Size { 512,
                                     424 },
                          CV_32FC1,
                          (void*)(depth_frame),
                          cv::Mat::AUTO_STEP};

    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(3);

    cv::Mat m1(depth_image.rows, depth_image.cols, CV_8UC4, depth_image.data);
    cv::imwrite(saveDir + "/" + std::to_string(_frame) + ".png", m1, compression_params);

    return true;
}


template <typename DepthType, typename ColorType>
float* DepthSource<DepthType, ColorType>::loadFrame(const std::string loadDir) {
    cv::Mat depth;
    depth = cv::imread(loadDir + "/" + std::to_string(_frame) + ".png", -1);
    if (depth.data == NULL)
    {
        return (float*)(depth.data);
    }

    cv::Mat m2(depth.rows, depth.cols, CV_32FC1, depth.data);
    auto depth_frame = (float*)(m2.data);

    return depth_frame;
}




}

#endif // DEPTH_SOURCE_H
