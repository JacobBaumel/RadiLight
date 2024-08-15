#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include "nvapriltags/include/nvAprilTags.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <iostream>

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
/*
int sender(int argc, char *argv[], float sendData[7]){
    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    char buffer[256];
    if (argc < 3) {
       fprintf(stderr,"usage %s hostname port\n", argv[0]);
       exit(0);
    }
    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, 
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
        error("ERROR connecting");

    /*if(!sendData)                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    n = write(sockfd,sendData,sizeof(sendData));
    if (n < 0) 
         error("ERROR writing to socket");
    close(sockfd);
}


void error(const char *msg){
    perror(msg);
    exit(0);
}
*/


int main() {
    printf("cuda main");
    int framesRead = 0;
    cv::VideoCapture capture;
    cv::Mat frame;
    cv::Mat img_rgba8;
    float *quaternion;

    capture.open(0);
    //capture.set(cv::CAP_PROP_FPS, 30);
    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    
    capture >> frame;
    cv::cvtColor(frame, img_rgba8, cv::COLOR_BGR2RGBA);
    auto *impl_ = new AprilTagsImpl();
    impl_->initialize(img_rgba8.cols, img_rgba8.rows,
                      img_rgba8.total() * img_rgba8.elemSize(),  img_rgba8.step, .1651f, 2);

    while (capture.isOpened()){
        capture >> frame;
        framesRead++;
        cv::cvtColor(frame, img_rgba8, cv::COLOR_BGR2RGBA);
        

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
            for (auto corner : detection.corners) {
                float x = corner.x;
                float y = corner.y;
                quaternion = converterQuaternion(detection.orientation);
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