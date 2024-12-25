#include "depthai/depthai.hpp"
#include <cmath>
#include <iostream>
#include <ostream>

class HostSpatialsCalc {
private:
  dai::CalibrationHandler calibData;
  int DELTA = 5;
  int THRESH_LOW = 200;    // 20cm
  int THRESH_HIGH = 30000; // 30m

  std::array<int, 4> checkInput(const std::array<int, 4> &roi,
                                const cv::Mat &frame) {
		std::cout<<"Func1"<<std::endl;
    return roi; // Already ROI
  }

  std::array<int, 4> checkInput(const std::array<int, 2> &point,
                                const cv::Mat &frame) {
    // Convert point to ROI
    int x = std::min(std::max(point[0], DELTA), frame.cols - DELTA);
    int y = std::min(std::max(point[1], DELTA), frame.rows - DELTA);
    return {x - DELTA, y - DELTA, x + DELTA, y + DELTA};
  }

  double calcAngle(const cv::Mat &frame, int offset, double HFOV) {
    return std::atan(std::tan(HFOV / 2.0) * offset / (frame.cols / 2.0));
  }

public:
  HostSpatialsCalc(dai::Device &device) {
    calibData = device.readCalibration();
  }

  void setLowerThreshold(int thresholdLow) { THRESH_LOW = thresholdLow; }

  void setUpperThreshold(int thresholdHigh) { THRESH_HIGH = thresholdHigh; }

  void setDeltaRoi(int delta) { DELTA = delta; }

  std::pair<std::map<std::string, float>, std::map<std::string, int>>
  calcSpatials(
      std::shared_ptr<dai::ImgFrame> depthData, const std::array<int, 2> &roi,
      std::function<float(const cv::Mat &)> averagingMethod =
          [](const cv::Mat &mat) {
            return static_cast<float>(cv::mean(mat)[0]);
          }) {
    // Get the depth frame as a cv::Mat
    cv::Mat depthFrame = depthData->getFrame();

    // Convert the ROI or point to a valid ROI2
    auto roiChecked = checkInput(roi, depthFrame);
    int xmin = roiChecked[0], ymin = roiChecked[1], xmax = roiChecked[2],
        ymax = roiChecked[3];

    // Get the ROI region
    cv::Mat depthROI =
       depthFrame(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));
    // cv::Mat depthROI =
        // depthFrame(cv::Rect(100, 300, 60, 60));

    // Mask based on thresholds
    cv::Mat inRange;
    cv::inRange(depthROI, THRESH_LOW, THRESH_HIGH, inRange);

    // Get the average depth within the valid ROI
    float averageDepth = averagingMethod(depthROI);

    // Calculate the centroid of the ROI
    std::map<std::string, int> centroid = {{"x", (xmin + xmax) / 2},
                                           {"y", (ymin + ymax) / 2}};

    // Required information for spatial calculation
    double HFOV =
        calibData.getFov(dai::CameraBoardSocket(depthData->getInstanceNum()));
    int midW = depthFrame.cols / 2;
    int midH = depthFrame.rows / 2;

    int bbXPos = centroid["x"] - midW;
    int bbYPos = centroid["y"] - midH;

    double angleX = calcAngle(depthFrame, bbXPos, HFOV);
    double angleY = calcAngle(depthFrame, bbYPos, HFOV);

    // Calculate spatial coordinates
    std::map<std::string, float> spatials = {
        {"z", averageDepth},
        {"x", static_cast<float>(averageDepth * std::tan(angleX))},
        {"y", static_cast<float>(-averageDepth * std::tan(angleY))}};

    return {spatials, centroid};
  }
};

