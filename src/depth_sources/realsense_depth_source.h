#ifndef REALSENSE_DEPTH_SOURCE_H
#define REALSENSE_DEPTH_SOURCE_H

#include "depth_source.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <vector_types.h>

namespace dart {

template <typename DepthType, typename ColorType>
class RealSenseDepthSource : public DepthSource<ushort,uchar3> {
public:
    RealSenseDepthSource();

    ~RealSenseDepthSource();

    bool initialize(const bool isLive = true);

    const ushort * getDepth() const;

    const ushort * getDeviceDepth() const;

    const uchar3 * getColor() const;

    ColorLayout getColorLayout() const { return LAYOUT_RGB; }

    uint64_t getDepthTime() const;

    uint64_t getColorTime() const;

    void setFrame(const uint frame);

    void advance();

    bool hasRadialDistortionParams() const { return false; }

    inline float getScaleToMeters() const { return 1/(1000.0f); }

private:
   

    int _frameIndexOffset;
    ushort * _deviceDepth;
};

}

#endif // REALSENSE_DEPTH_SOURCE_H
