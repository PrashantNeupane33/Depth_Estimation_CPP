#ifndef ONNX_INFERENCE_HPP
#define ONNX_INFERENCE_HPP

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// BoundingBox structure to hold the detection result
struct BoundingBox {
  float x_min, y_min, x_max, y_max, confidence;
  int class_id;
};

// Function to load a model, perform inference, and return bounding boxes
inline std::vector<BoundingBox> inferFrame(const cv::Mat &frame,
                                           const std::string &model_path) {
  std::vector<BoundingBox> boxes;

  // Ensure the frame is valid
  if (frame.empty()) {
    std::cerr << "Error: Input frame is empty!" << std::endl;
    return boxes;
  }

  // Initialize ONNX Runtime
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
  Ort::SessionOptions session_options;
  Ort::Session session(env, model_path.c_str(), session_options);

  // Get input details
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session.GetInputNameAllocated(0, allocator);
  auto input_shape =
      session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  int input_width = input_shape[3];
  int input_height = input_shape[2];

  // Preprocess the input frame
  cv::Mat resized, floatImage;
  cv::resize(frame, resized, cv::Size(input_width, input_height));
  resized.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

  // Convert HWC to CHW
  std::vector<cv::Mat> channels(3);
  cv::split(floatImage, channels);
  std::vector<float> input_tensor_values;
  for (const auto &channel : channels) {
    input_tensor_values.insert(input_tensor_values.end(),
                               channel.begin<float>(), channel.end<float>());
  }

  // Create input tensor
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(),
      input_shape.data(), input_shape.size());

  // Get output details
  auto output_name = session.GetOutputNameAllocated(0, allocator);

  // Run inference
  const char *input_names[] = {input_name.get()};
  const char *output_names[] = {output_name.get()};
  auto output_tensors =
      session.Run({}, input_names, &input_tensor, 1, output_names, 1);

  // Process output - Assuming output is in [batch_size, num_boxes, 6] format
  auto output_data = output_tensors[0].GetTensorData<float>();
  auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  int num_boxes = output_shape[1];

  for (int i = 0; i < num_boxes; ++i) {
    BoundingBox box;
    box.x_min = output_data[i * 6];
    box.y_min = output_data[i * 6 + 1];
    box.x_max = output_data[i * 6 + 2];
    box.y_max = output_data[i * 6 + 3];
    box.confidence = output_data[i * 6 + 4];
    box.class_id = static_cast<int>(output_data[i * 6 + 5]);

    // Filter boxes based on confidence threshold (e.g., 0.5)
    if (box.confidence > 0.5) {
      boxes.push_back(box);
    }
  }

  return boxes;
}

// Function to draw bounding boxes on the frame
inline void visualizeBoundingBoxes(cv::Mat &frame,
                                   const std::vector<BoundingBox> &boxes) {
  for (const auto &box : boxes) {
    // Draw rectangle
    cv::rectangle(
        frame,
        cv::Point(static_cast<int>(box.x_min), static_cast<int>(box.y_min)),
        cv::Point(static_cast<int>(box.x_max), static_cast<int>(box.y_max)),
        cv::Scalar(0, 255, 0), 2);

    // Annotate with class ID and confidence
    std::string label = "Class: " + std::to_string(box.class_id) +
                        " Confidence: " + std::to_string(box.confidence);
    int base_line = 0;
    cv::Size label_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
    cv::rectangle(
        frame,
        cv::Point(static_cast<int>(box.x_min),
                  static_cast<int>(box.y_min) - label_size.height - 5),
        cv::Point(static_cast<int>(box.x_min) + label_size.width,
                  static_cast<int>(box.y_min)),
        cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(
        frame, label,
        cv::Point(static_cast<int>(box.x_min), static_cast<int>(box.y_min) - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }
}

#endif // ONNX_INFERENCE_HPP

