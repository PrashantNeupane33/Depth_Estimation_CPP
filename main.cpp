#include <depthai/depthai.hpp>
#include <cmath>
#include <iostream>
#include "calc.hpp" 



int main() {
    // Create pipeline
    dai::Pipeline pipeline;

    // Define sources and outputs
    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();

    // Properties
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoLeft->setBoardSocket(dai::CameraBoardSocket::CAM_B);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_400_P);
    monoRight->setBoardSocket(dai::CameraBoardSocket::CAM_C);

    stereo->initialConfig.setConfidenceThreshold(255);
    stereo->setLeftRightCheck(true);
    stereo->setSubpixel(false);

    // Linking
    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);

    auto xoutDepth = pipeline.create<dai::node::XLinkOut>();
    xoutDepth->setStreamName("depth");
    stereo->depth.link(xoutDepth->input);

    auto xoutDisp = pipeline.create<dai::node::XLinkOut>();
    xoutDisp->setStreamName("disp");
    stereo->disparity.link(xoutDisp->input);

    // Connect to device and start pipeline
    dai::Device device(pipeline);
    auto depthQueue = device.getOutputQueue("depth");
    auto dispQueue = device.getOutputQueue("disp");

    HostSpatialsCalc hostSpatials(device);

    int y = 200;
    int x = 300;
    int step = 3;
    int delta = 10;
    hostSpatials.setDeltaRoi(delta);

    std::cout << "Use WASD keys to move ROI.\nUse 'r' and 'f' to change ROI size." << std::endl;

    while (true) {
				std::cout << "HERE1" << std::endl;
        auto depthData = depthQueue->get<dai::ImgFrame>();
				std::cout << "HERE2" << std::endl;
        auto spatials = hostSpatials.calcSpatials(depthData, {x, y});
				std::cout << "HERE3" << std::endl;

        // Get disparity frame for visualization
        auto dispData = dispQueue->get<dai::ImgFrame>();
			
        cv::Mat disp = dispData->getCvFrame();
        disp.convertTo(disp, CV_8UC1, 255.0 / stereo->initialConfig.getMaxDisparity());
        cv::applyColorMap(disp, disp, cv::COLORMAP_JET);

        // Clamp ROI dimensions to avoid exceeding frame bounds
        auto roi = clampROI(x, y, delta, disp.cols, disp.rows);

        // Extract ROI coordinates
        int rectX1 = roi[0];
        int rectY1 = roi[1];
        int rectX2 = roi[2];
        int rectY2 = roi[3];

        // Validate ROI size before creating a Mat object
        if (rectX2 - rectX1 > 0 && rectY2 - rectY1 > 0) {
            cv::rectangle(disp, cv::Point(rectX1, rectY1), cv::Point(rectX2, rectY2), cv::Scalar(255, 255, 255), 1);
        } else {
            std::cerr << "Invalid ROI dimensions: " << rectX1 << ", " << rectY1 << ", " << rectX2 << ", " << rectY2 << "\n";
            continue; // Skip to the next iteration
        }

        // Display spatial information
        // std::string xText = "X: " + (std::isnan(spatials.x) ? "--" : std::to_string(spatials.x / 1000) + "m");
        // std::string yText = "Y: " + (std::isnan(spatials.y) ? "--" : std::to_string(spatials.y / 1000) + "m");
        // std::string zText = "Z: " + (std::isnan(spatials.z) ? "--" : std::to_string(spatials.z / 1000) + "m");
        //
        // cv::putText(disp, xText, cv::Point(rectX1 + 10, rectY1 + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        // cv::putText(disp, yText, cv::Point(rectX1 + 10, rectY1 + 35), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        // cv::putText(disp, zText, cv::Point(rectX1 + 10, rectY1 + 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        //
        // // Show the frame
        cv::imshow("depth", disp);

        int key = cv::waitKey(1);
        if (key == 'q') {
            break;
        } else if (key == 'w') {
            y -= step;
        } else if (key == 'a') {
            x -= step;
        } else if (key == 's') {
            y += step;
        } else if (key == 'd') {
            x += step;
        } else if (key == 'r') { // Increase Delta
            if (delta < 50) {
                delta += 1;
                hostSpatials.setDeltaRoi(delta);
            }
        } else if (key == 'f') { // Decrease Delta
            if (delta > 3) {
                delta -= 1;
                hostSpatials.setDeltaRoi(delta);
            }
        }
    }

    return 0;
}

