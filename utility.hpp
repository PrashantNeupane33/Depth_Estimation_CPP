#include <depthai/depthai.hpp>

// Clamp Function
std::array<int, 4> clampROI(int x, int y, int delta, int frameWidth,
                            int frameHeight) {
  int rectX1 = std::max(0, x - delta);
  int rectY1 = std::max(0, y - delta);
  int rectX2 = std::min(frameWidth - 1, x + delta);
  int rectY2 = std::min(frameHeight - 1, y + delta);
  return {rectX1, rectY1, rectX2, rectY2};
}

// Rescaling frame
cv::Mat rescale(const cv::Mat &frame, double scale) {
  int w = static_cast<int>(frame.cols * scale);
  int h = static_cast<int>(frame.rows * scale);
  cv::Mat resizedFrame;
  cv::resize(frame, resizedFrame, cv::Size(w, h), 0, 0, cv::INTER_AREA);
  return resizedFrame;
}

//Function to create a Rectangular Mask
std::tuple<cv::Mat, int, int, int, int>
recMask(int x, int y, int w, int h, int imgH, int imgW, int maskOffset) {
  cv::Mat mask = cv::Mat::zeros(imgH, imgW, CV_8U);

  int x1 = static_cast<int>(x - w / 2);
  int y1 = static_cast<int>(y - h / 2);
  int x2 = static_cast<int>(x + w / 2);
  int y2 = static_cast<int>(y + h / 2);

  cv::rectangle(mask, cv::Point(x1 + maskOffset, y1 + maskOffset),
                cv::Point(x2 + maskOffset, y2 + maskOffset), 255, -1);

  return {mask, x1, y1, x2, y2};
}
