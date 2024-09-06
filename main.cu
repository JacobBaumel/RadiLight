#include "cuda.h"
#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "nvapriltags/include/nvAprilTags.h"
#include <Eigen/Dense>
#include <iostream>
#include <string.h>
#include <json.hpp>
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

Eigen::Quaterniond computeAverageQuaternion(const std::vector<Eigen::Quaterniond> quaternions) {
    Eigen::MatrixXd quat_matrix(4, quaternions.size());
    for (size_t i = 0; i < quaternions.size(); ++i) {
        quat_matrix.col(i) << quaternions[i].w(), quaternions[i].x(), quaternions[i].y(), quaternions[i].z();
    }

    Eigen::Vector4d avg_quat_vector = quat_matrix.rowwise().mean();
    Eigen::Quaterniond avg_quat(avg_quat_vector[0], avg_quat_vector[1], avg_quat_vector[2], avg_quat_vector[3]);

    avg_quat.normalize();
    return avg_quat;
}

std::array<float, 4> converterQuaternion(const cv::Mat matrix){
    std::array<float, 4> quaternion;
    float fourWSquaredMinus1 = matrix.at<float>(0, 0) + matrix.at<float>(1, 1) + matrix.at<float>(2, 2);
    float fourXSquaredMinus1 = matrix.at<float>(0, 0) - matrix.at<float>(1, 1) - matrix.at<float>(2, 2);
    float fourYSquaredMinus1 = matrix.at<float>(1, 1) - matrix.at<float>(0, 0) - matrix.at<float>(2, 2);
    float fourZSquaredMinus1 = matrix.at<float>(2, 2) - matrix.at<float>(0, 0) - matrix.at<float>(1, 1);
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
            
            float biggestVal = sqrt (fourBiggestSquaredMinus1 + 1.0f ) * 0.5f;
            float mult = 0.25f / biggestVal;
            
    switch (biggestIndex) {
        case 0:
            quaternion[0] = biggestVal;
            quaternion[1] = (matrix.at<float>(2, 1) - matrix.at<float>(1, 2)) * mult;
            quaternion[2] = (matrix.at<float>(0, 2) - matrix.at<float>(2, 0)) * mult;
            quaternion[3] = (matrix.at<float>(1, 0) - matrix.at<float>(0, 1)) * mult;
            break;
        case 1:
            quaternion[1] = biggestVal;
            quaternion[0] = (matrix.at<float>(2, 1) - matrix.at<float>(1, 2)) * mult;
            quaternion[2] = (matrix.at<float>(1, 0) + matrix.at<float>(0, 1)) * mult;
            quaternion[3] = (matrix.at<float>(0, 2) + matrix.at<float>(2, 0)) * mult;
            break;
        case 2:
            quaternion[2] = biggestVal;
            quaternion[0] = (matrix.at<float>(0, 2) - matrix.at<float>(2, 0)) * mult;
            quaternion[1] = (matrix.at<float>(1, 0) + matrix.at<float>(0, 1)) * mult;
            quaternion[3] = (matrix.at<float>(2, 1) + matrix.at<float>(1, 2)) * mult;
            break;
        case 3:
            quaternion[3] = biggestVal;
            quaternion[0] = (matrix.at<float>(1, 0) - matrix.at<float>(0, 1)) * mult;
            quaternion[1] = (matrix.at<float>(0, 2) + matrix.at<float>(2, 0)) * mult;
            quaternion[2] = (matrix.at<float>(2, 1) + matrix.at<float>(1, 2)) * mult;
            break;
    }
    return quaternion;
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

std::array<float, 7> computeAveragePose(const std::vector<Eigen::Affine3d> poses) {
    Eigen::Vector3d mean_translation(0, 0, 0);
    Eigen::Matrix3d mean_rotation = Eigen::Matrix3d::Zero();
    std::vector<Eigen::Quaterniond> quaternions;

    for (const auto &pose : poses) {
        mean_translation += pose.translation();
        quaternions.push_back(Eigen::Quaterniond(pose.linear()));
    }

    mean_translation /= poses.size();

    Eigen::Quaterniond average_quaternion = computeAverageQuaternion(quaternions);

    Eigen::Matrix3d mean_rotation_matrix = average_quaternion.toRotationMatrix();

    return {mean_translation.x(), mean_translation.y(), mean_translation.z(), mean_rotation_matrix.w(), mean_rotation_matrix.x(), mean_rotation_matrix.y(), mean_rotation_matrix.z()};
}

Eigen::Affine3d estimateFieldToRobotAprilTag(std::array<float, 7> apriltagPose, int id){
    Eigen::Affine3d cameraToTarget = Eigen::Affine3d::Identity();
    cameraToTarget.translation() = Eigen::Vector3d(apriltagPose[0], apriltagPose[1], apriltagPose[2]);
    cameraToTarget.linear() = Eigen::Quaterniond(apriltagPose[3], apriltagPose[4], apriltagPose[5], apriltagPose[6]).toRotationMatrix();
    
    Eigen::Affine3d cameraToRobot = Eigen::Affine3d::Identity();
    cameraToRobot.translation() = Eigen::Vector3d(-.1064,.0873,0);
    cameraToRobot.linear() = Eigen::Quaterniond(0, 0.130526, 0, 0.9914449).toRotationMatrix();

    Eigen::Affine3d fieldRelativeTagPose = Eigen::Affine3d::Identity();
    std::ifstream file("2024-crescendo.json");
    nlohmann::json j;
    file >> j;
    for (const auto &tag : j["tags"]) {
        if (tag["ID"] == id) {
                fieldRelativeTagPose.translation() = Eigen::Vector3d(tag["pose"]["translation"]["x"].get<float>(), 
                tag["pose"]["translation"]["y"].get<float>(), 
                tag["pose"]["translation"]["z"].get<float>());
                fieldRelativeTagPose.linear() = Eigen::Quaterniond( tag["pose"]["quaternion"]["W"].get<float>(), 
                tag["pose"]["quaternion"]["X"].get<float>(), 
                tag["pose"]["quaternion"]["Y"].get<float>(), 
                tag["pose"]["quaternion"]["Z"].get<float>()).toRotationMatrix();
        }
    }

    Eigen::Affine3d eigenCameraToTargetInverse = cameraToTarget.inverse();
    
    Eigen::Affine3d result = fieldRelativeTagPose * eigenCameraToTargetInverse * cameraToRobot;

    return result;
}

std::array<float, 7> poseEstimate(const std::vector<cv::Point2d> &imagePts){
    static std::array<float, 7> lastValidPose = {0}; 
    std::array<float, 7> finalPose = {0};

    if (imagePts.empty()) {
        std::cout << "No detections" << "\n";
        return lastValidPose;
    }

    std::vector<cv::Point3d> apriltagPts = {
        cv::Point3d(-0.08255, 0.08255, 0), 
        cv::Point3d(0.08255, 0.08255, 0), 
        cv::Point3d(0.08255, -0.08255, 0), 
        cv::Point3d(-0.08255, -0.08255, 0)
    }; 

    float K[] = {802.9265702293416f, 0, 966.4221440154661f, 
                 0, 803.6309422064642f, 477.6409613889424f, 
                 0, 0, 1};
    cv::Mat kMat(3, 3, CV_32F, K);

    float distCoeffs[5] = {0}; 
    cv::Mat distMat(1, 5, CV_32F, distCoeffs);

    std::vector<cv::Point2d> reorderedImagePts = {imagePts[1], imagePts[2], imagePts[3], imagePts[0]};

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(apriltagPts, reorderedImagePts, kMat, distMat, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
    if (!success){
        std::cout << "Error with SolvePNP" << "\n";
        return lastValidPose;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    
    std::array<float, 4> quaternion = converterQuaternion(R);

    finalPose[0] = static_cast<float>(tvec.at<double>(0));
    finalPose[1] = static_cast<float>(tvec.at<double>(1));
    finalPose[2] = static_cast<float>(tvec.at<double>(2));
    finalPose[3] = quaternion[0];
    finalPose[4] = quaternion[1];
    finalPose[5] = quaternion[2];
    finalPose[6] = quaternion[3];
    
    lastValidPose = finalPose;
    return finalPose;
}

int main() {
    //setStaticIP("192.168.86.5");
    printf("cuda main");
    std::vector<cv::Point2d> imagePts;
    std::array<float, 7> multitagfieldPose;
    std::vector<Eigen::Affine3d> fieldPoses;
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
        for (int i = 0; i < numDetections; i++) {
            const nvAprilTagsID_t &detection = impl_->tags[i];
            for (auto corner : detection.corners) {
                imagePts.push_back(cv::Point2d(corner.x, corner.y));
                cv::circle(frame, cv::Point(corner.x, corner.y), 4, cv::Scalar(255, 0, 0), -1);
            }
            fieldPoses.push_back(estimateFieldToRobotAprilTag(poseEstimate(imagePts), detection.id));
        }
        multitagfieldPose = computeAveragePose(fieldPoses);
        
        for (int i = 0; i < multitagfieldPose.size(); i++){
            std::cout << multitagfieldPose[i] << "\n";
        }    


        webserverSender(multitagfieldPose, webserverSock);
        roborioSender(multitagfieldPose, roborioSock);
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