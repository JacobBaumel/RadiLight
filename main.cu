#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include "nvapriltags/include/nvAprilTags.h"
using namespace cv;

// copy from https://github.com/NVIDIA-AI-IOT/ros2-nvapriltags/blob/main/src/AprilTagNode.cpp

struct AprilTagsImpl {
    // Handle used to interface with the stereo library.
    nvAprilTagsHandle april_tags_handle = nullptr;
    // Camera intrinsics
    nvAprilTagsCameraIntrinsics_t cam_intrinsics;

    // Output vector of detected Tags
    std::vector<nvAprilTagsID_t> tags;

    // CUDA stream
    cudaStream_t main_stream = {};

    // CUDA buffers to store the input image.
    nvAprilTagsImageInput_t input_image;

    // CUDA memory buffer container for RGBA images.
    uchar4 *input_image_buffer = nullptr;

    // Size of image buffer
    size_t input_image_buffer_size = 0;

    int max_tags;

    void initialize(const uint32_t width,
                    const uint32_t height, const size_t image_buffer_size,
                    const size_t pitch_bytes,
                    const float fx, const float fy, const float cx, const float cy,
                    float tag_edge_size_, int max_tags_) {
        assert(!april_tags_handle), "Already initialized.";

        // Get camera intrinsics
        cam_intrinsics = {fx, fy, cx, cy};

        // Create AprilTags detector instance and get handle
        const int error = nvCreateAprilTagsDetector(
                &april_tags_handle, width, height, nvAprilTagsFamily::NVAT_TAG36H11,
                &cam_intrinsics, tag_edge_size_);
        if (error != 0) {
            throw std::runtime_error(
                    "Failed to create NV April Tags detector (error code " +
                    std::to_string(error) + ")");
        }

        // Create stream for detection
        cudaStreamCreate(&main_stream);

        // Allocate the output vector to contain detected AprilTags.
        tags.resize(max_tags_);
        max_tags = max_tags_;
        // Setup input image CUDA buffer.
        const cudaError_t cuda_error =
                cudaMalloc(&input_image_buffer, image_buffer_size);
        if (cuda_error != cudaSuccess) {
            throw std::runtime_error("Could not allocate CUDA memory (error code " +
                                     std::to_string(cuda_error) + ")");
        }

        // Setup input image.
        input_image_buffer_size = image_buffer_size;
        input_image.width = width;
        input_image.height = height;
        input_image.dev_ptr = reinterpret_cast<uchar4 *>(input_image_buffer);
        input_image.pitch = pitch_bytes;
    }

    ~AprilTagsImpl() {
        if (april_tags_handle != nullptr) {
            cudaStreamDestroy(main_stream);
            nvAprilTagsDestroy(april_tags_handle);
            cudaFree(input_image_buffer);
        }
    }
};

int main() {
    printf("cuda main");
    VideoCapture capture;
    int width = 1920;
    int height = 1080;
    float fx = 802.9265702293416;
    float fy = 803.6309422064642;
    float cx = 966.4221440154661;
    float cy = 477.6409613889424;


    capture.open(0);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    Mat frame;
    Mat img_rgba8;
    capture>>frame;
    cv::cvtColor(frame, img_rgba8, cv::COLOR_BGR2RGBA);
    auto *impl_ = new AprilTagsImpl();
    impl_->initialize(img_rgba8.cols, img_rgba8.rows,
                      img_rgba8.total() * img_rgba8.elemSize(),  img_rgba8.step,
                      fx,fy,cx,cy,
                      0.5,
                      6);

    while (capture.isOpened())
    {

        capture>>frame;
        cv::cvtColor(frame, img_rgba8, cv::COLOR_BGR2RGBA);
        auto start = std::chrono::system_clock::now();

        const cudaError_t cuda_error =
                cudaMemcpy(impl_->input_image_buffer, (uchar4 *)img_rgba8.ptr<unsigned char>(0),
                           impl_->input_image_buffer_size, cudaMemcpyHostToDevice);

        if (cuda_error != cudaSuccess) {
            throw std::runtime_error(
                    "Could not memcpy to device CUDA memory (error code " +
                    std::to_string(cuda_error) + ")");
        }

        uint32_t num_detections;
        const int error = nvAprilTagsDetect(
                impl_->april_tags_handle, &(impl_->input_image), impl_->tags.data(),
                &num_detections, impl_->max_tags, impl_->main_stream);
        if (error != 0) {
            throw std::runtime_error("Failed to run AprilTags detector (error code " +
                                     std::to_string(error) + ")");
        }

        for (int i = 0; i < num_detections; i++) {
            const nvAprilTagsID_t &detection = impl_->tags[i];

            // corners
            for (auto corner : detection.corners) {
               float x = corner.x;
               float y = corner.y;
            }

        }

        auto end = std::chrono::system_clock::now();
        int fps = int(1000 / ( std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() + 1));
        cv::putText(frame, "FPS: "+ std::to_string(fps), cv::Point(100,100),
                    cv::FONT_HERSHEY_PLAIN, 5, cv::Scalar(0xFF, 0xFF, 0), 2);
        std::cout<<"num_detections: "<<num_detections<<std::endl;

        cv::namedWindow("frame", 0);
        cv::resizeWindow("frame", 1280,800);
        cv::imshow("frame", frame);
        if (cv::waitKey(10)==27)
            break;
    }
    delete(impl_);
    return 0;
}