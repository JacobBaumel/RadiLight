#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "nvapriltags/include/nvAprilTags.h"
#include <iostream>
#include <string.h>
#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <net/if.h>


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

std::array<float, 4> averageRotationMatrixToQuaternion(const std::vector<cv::Mat> &rotationMatrices) {
    cv::Mat avgRotationMatrix = cv::Mat::zeros(3, 3, CV_64F);

    for (const auto& R : rotationMatrices) {
        avgRotationMatrix += R;
    }

    avgRotationMatrix /= rotationMatrices.size();

    cv::Mat U, S, VT;
    cv::SVD::compute(avgRotationMatrix, S, U, VT);
    avgRotationMatrix = U * VT;

    cv::Mat quaternion;
    avgRotationMatrix.convertTo(avgRotationMatrix, CV_32F);
    cv::Rodrigues(avgRotationMatrix, quaternion);

    float norm = cv::norm(quaternion);
    quaternion /= norm;

    std::array<float, 4> quaternionArray;
    quatArray[0] = quaternion.at<float>(0); // x
    quatArray[1] = quaternion.at<float>(1); // y
    quatArray[2] = quaternion.at<float>(2); // z
    quatArray[3] = quaternion.at<float>(3); // w

    return quaternionArray;
}

void roborioSender(std::array<float, 7>& sendData, int sock) {
    sockaddr_in destination;
    destination.sin_family = AF_INET;
    destination.sin_port = htons(9000);
    destination.sin_addr.s_addr = inet_addr("192.168.86.42");

    size_t size = sendData.size() * sizeof(float);

    int n_bytes = ::sendto(sock, sendData.data(), size, 0, reinterpret_cast<sockaddr*>(&destination), sizeof(destination));
    std::cout << n_bytes << " bytes sent" << std::endl;
}

void webserverSender(std::array<float, 7>& sendData, int sock) {
    sockaddr_in destination;
    destination.sin_family = AF_INET;
    destination.sin_port = htons(9001);
    destination.sin_addr.s_addr = inet_addr("127.0.0.1");

    size_t size = sendData.size() * sizeof(float);

    int n_bytes = ::sendto(sock, sendData.data(), size, 0, reinterpret_cast<sockaddr*>(&destination), sizeof(destination));
    std::cout << n_bytes << " bytes sent" << std::endl;
}

std::string webserverRecevier(int sock) {
    int len = sizeof(sockaddr_in);
    char buff[1024];
    sockaddr_in destination;
    destination.sin_family = AF_INET;
    destination.sin_port = htons(9001);
    destination.sin_addr.s_addr = inet_addr("127.0.0.1");

    int readStatus = recvfrom(sock, buff, sizeof(buff), 0, (struct sockaddr*)&destination, (socklen_t*)&len);
    buff[readStatus] = '\0';
    return std::string(buff);
}
    
void setStaticIP(char ip_address[15]){
    ifreq ifr;
    sockaddr_in *addr;    
    //make socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    //define ipv4
    ifr.ifr_addr.sa_family = AF_INET;

    //define network interface
    memcpy(ifr.ifr_name, "eth0", IFNAMSIZ-1);

    //define address
    addr=(struct sockaddr_in *)&ifr.ifr_addr;

    //convert ip
    inet_pton(AF_INET,ip_address,&addr->sin_addr);

    //ipset
    ioctl(sock, SIOCSIFADDR, &ifr);
    
    std::memset(&ifr, 0, sizeof(ifr));
    std::strncpy(ifr.ifr_name, "eth0", IFNAMSIZ - 1);

    //disable eth0
    ioctl(sock, SIOCGIFFLAGS, &ifr);
    ifr.ifr_flags &= ~IFF_UP;
    ioctl(sock, SIOCSIFFLAGS, &ifr);

    //wait
    sleep(1);

    //enable eth0
    ifr.ifr_flags |= IFF_UP;
    ioctl(sock, SIOCSIFFLAGS, &ifr); 

    //socket close
    close(sock);

    std::cout << ("IP Address updated sucessfully.\n");
}

std::array<float, 7> multiTagFinder(uint32_t numDetections, const std::vector<std::vector<cv::Point2d>> &imagePtsSet){
    static std::array<float, 7> lastValidPose = {0}; 
    std::array<float, 7> finalPose = {0};

    if (numDetections == 0 || imagePtsSet.empty()) {
        std::cout << "No detections" << "\n";
        return lastValidPose;
    }
    
    std::vector<cv::Point3d> apriltagPts = {
        cv::Point3d(-0.08255, 0.08255, 0), 
        cv::Point3d(0.08255, 0.08255, 0), 
        cv::Point3d(0.08255, -0.08255, 0), 
        cv::Point3d(-0.08255, -0.08255, 0)
    }; 

    float K[] = {802.9265702293416, 0, 966.4221440154661, 0, 803.6309422064642, 477.6409613889424, 0, 0, 1};
    cv::Mat kMat(3, 3, CV_32F, K);

    float distCoeffs[5] = {0}; 
    cv::Mat distMat(1, 5, CV_32F, distCoeffs);

    cv::Mat avgTranslation = cv::Mat::zeros(3, 1, CV_64F);

    std::vector<cv::Mat> rotationMatrices;

    for (uint32_t i = 0; i < numDetections; i++) {
        std::vector<cv::Point2d> imagePts = imagePtsSet[i];
        imagePts = {imagePts[1], imagePts[2], imagePts[3], imagePts[0]};
        cv::Mat rvec, tvec, R;

        bool error = cv::solvePnP(apriltagPts, imagePts, kMat, distMat, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
        if (!error){
            std::cout << "Error with SolvePNP" << "\n";
            return lastValidPose;
        }
        cv::Rodrigues(rvec, R);
        rotationMatrices.push_back(R);
        avgTranslation += tvec;
    }
    avgTranslation /= numDetections;

    std::array<float, 4> avgQuaternion = averageRotationMatrixToQuaternion(rotationMatrices);

    finalPose[0] = static_cast<float>(avgTranslation.at<double>(0));
    finalPose[1] = static_cast<float>(avgTranslation.at<double>(1));
    finalPose[2] = static_cast<float>(avgTranslation.at<double>(2));
    finalPose[3] = avgQuaternion[0];
    finalPose[4] = avgQuaternion[1];
    finalPose[5] = avgQuaternion[2];
    finalPose[6] = avgQuaternion[3];
    
    lastValidPose = finalPose;
    return finalPose;
}


int main() {
    //setStaticIP("192.168.86.5");
    printf("cuda main");
    std::array<float, 7> finalPose;
    std::vector<cv::Point2d> apriltagPoints;
    std::vector<std::vector<cv::Point2d>> imagePts;
    std::string buff;
    int roborioSock = ::socket(AF_INET, SOCK_DGRAM, 0);
    int webserverSock = ::socket(AF_INET, SOCK_DGRAM, 0);
    cv::VideoCapture capture = cv::VideoCapture(0, cv::CAP_GSTREAMER);
    cv::Mat frame;
    cv::Mat img_rgba8;
    cv::cuda::GpuMat img_rgba8gpu;  
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
                      img_rgba8.total() * img_rgba8.elemSize(),  img_rgba8.step, .1651f, 5);

    while (capture.isOpened()){
        //auto start = std::chrono::system_clock::now();
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

        uint32_t numDetections;
        const int error = nvAprilTagsDetect(
                impl_->april_tags_handle, &(impl_->input_image), impl_->tags.data(),
                &numDetections, impl_->max_tags, impl_->main_stream);
        if (error != 0) {
            throw std::runtime_error("Failed to run AprilTags detector (error code " +
                                     std::to_string(error) + ")");
        }

        imagePts.clear(); 
        apriltagPoints.clear();
        for (int i = 0; i < numDetections; i++) {
            const nvAprilTagsID_t &detection = impl_->tags[i];
            for (auto corner : detection.corners) {
                apriltagPoints.push_back(cv::Point2d(corner.x, corner.y));
                cv::circle(frame, cv::Point(corner.x, corner.y), 4, cv::Scalar(255, 0, 0), -1);
            }
            imagePts.push_back(apriltagPoints);
        }

        finalPose = multiTagFinder(numDetections, imagePts);
        for (int i = 0; i < finalPose.size(); i++){
            std::cout << finalPose[i] << "\n";
        }    
        webserverSender(finalPose, webserverSock);
        roborioSender(finalPose, roborioSock);
        buff = webserverRecevier(webserverSock);
        std::cout << buff;
        cv::imshow("frame", frame); 
        if (cv::waitKey(10)==27)
                break;
    }

    ::close(roborioSock);
    ::close(webserverSock);
    delete impl_;
    return 0;
}
