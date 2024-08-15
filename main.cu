#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "nvapriltags/include/nvAprilTags.h"
#include <iostream>
#include <string.h>
#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#define PORT 8080

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
                    float tag_edge_size_, int max_tags_) {
        assert(!april_tags_handle), "Already initialized.";

        // Get camera intrinsics
        cam_intrinsics = {802.9265702293416, 803.6309422064642, 966.4221440154661, 477.6409613889424};

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
                cudaMallocManaged(&input_image_buffer, image_buffer_size);
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


float *converterQuaternion(const float *matrix ){
    float *quaternion = new float[4];
    float fourWSquaredMinus1 = matrix[0] + matrix[4] + matrix[8];
            float fourXSquaredMinus1 = matrix[0] - matrix[4] - matrix[8];
            float fourYSquaredMinus1 = matrix[4] - matrix[0] - matrix[8];
            float fourZSquaredMinus1 = matrix[8] - matrix[0] - matrix[4];

            int biggestIndex = 0;
            float fourBiggestSquaredMinus1 = fourWSquaredMinus1;

            if(fourXSquaredMinus1 > fourBiggestSquaredMinus1) {
                fourBiggestSquaredMinus1 = fourXSquaredMinus1;
                biggestIndex = 1;
            }
            if (fourYSquaredMinus1 > fourBiggestSquaredMinus1) {
                fourBiggestSquaredMinus1 = fourYSquaredMinus1;
                biggestIndex = 2;
            }
            if (fourZSquaredMinus1 > fourBiggestSquaredMinus1) {
                fourBiggestSquaredMinus1 = fourZSquaredMinus1;
                biggestIndex = 3;
            }
            // Per form square root and division
            float biggestVal = sqrt (fourBiggestSquaredMinus1 + 1.0f ) * 0.5f;
            float mult = 0.25f / biggestVal;

            // Apply table to compute quaternion values
            switch (biggestIndex) {
                case 0:
                    quaternion[0] = biggestVal;
                    quaternion[1] = (matrix[7] - matrix[5]) * mult;
                    quaternion[2] = (matrix[2] - matrix[6]) * mult;
                    quaternion[3] = (matrix[3] - matrix[1]) * mult;
                    break;
                case 1:
                    quaternion[1] = biggestVal;
                    quaternion[0] = (matrix[7] - matrix[5]) * mult;
                    quaternion[2] = (matrix[3] + matrix[1]) * mult;
                    quaternion[3] = (matrix[2] + matrix[6]) * mult;
                    break;
                case 2:
                    quaternion[2] = biggestVal;
                    quaternion[0] = (matrix[2] - matrix[6]) * mult;
                    quaternion[1] = (matrix[3] + matrix[1]) * mult;
                    quaternion[3] = (matrix[7] + matrix[5]) * mult;
                    break;
                case 3:
                    quaternion[3] = biggestVal;
                    quaternion[0] = (matrix[3] - matrix[1]) * mult;
                    quaternion[1] = (matrix[2] + matrix[6]) * mult;
                    quaternion[2] = (matrix[7] + matrix[5]) * mult;
                    break;
            }
            return quaternion;
}

void sender(float sendData[7], size_t size) {
    std::string hostname{"192.168.86.238"};
    uint16_t port = 9000;

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);

    sockaddr_in destination;
    destination.sin_family = AF_INET;
    destination.sin_port = htons(port);
    destination.sin_addr.s_addr = inet_addr(hostname.c_str());

    int n_bytes = ::sendto(sock, sendData, size, 0, reinterpret_cast<sockaddr*>(&destination), sizeof(destination));
    std::cout << n_bytes << " bytes sent" << std::endl;
    ::close(sock);
}



int main() {
    printf("cuda main");
    cv::VideoCapture capture;
    cv::Mat frame;
    cv::Mat img_rgba8;
    cv::cuda::GpuMat img_rgba8gpu;
    float *quaternion;
    float sendData[7];
    capture.open(0);
    //capture.set(cv::CAP_PROP_FPS, 30);
    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    
    capture >> frame;
    img_rgba8gpu.upload(frame);
    cv::cuda::cvtColor(img_rgba8gpu, img_rgba8gpu, cv::COLOR_BGR2RGBA);
    img_rgba8gpu.download(img_rgba8);
    auto *impl_ = new AprilTagsImpl();
    impl_->initialize(img_rgba8.cols, img_rgba8.rows,
                      img_rgba8.total() * img_rgba8.elemSize(),  img_rgba8.step, .1651f, 1);

    while (capture.isOpened()){
        capture >> frame;
        img_rgba8gpu.upload(frame);
        cv::cuda::cvtColor(img_rgba8gpu, img_rgba8gpu, cv::COLOR_BGR2RGBA);
        img_rgba8gpu.download(img_rgba8);

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
            quaternion = converterQuaternion(detection.orientation);
            sendData[0] = detection.translation[0];
            sendData[1] = detection.translation[1];
            sendData[2] = detection.translation[2];
            sendData[3] = quaternion[0];
            sendData[4] = quaternion[1];
            sendData[5] = quaternion[2];
            sendData[6] = quaternion[3];
            sender(sendData, sizeof(sendData));
            for (auto corner : detection.corners) {
                float x = corner.x;
                float y = corner.y;
                cv::circle(frame, cv::Point(x, y), 4, cv::Scalar(255, 0, 0), -1);
            }
        }
        
        cv::imshow("frame", frame); 
        std::cout << num_detections << std::endl;
        if (cv::waitKey(10)==27)
            break;
    }
    delete[] quaternion;
    delete(impl_);
    return 0;
}