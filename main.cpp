#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <vector>
#include "nvapriltags/nvapriltags/nvApril"
#include <opencv2/cudafeatures2d.hpp>


int main(){
    cv::VideoCapture cap(0);

    cv::Mat frame, tvecs, rvecs;
    cv::cuda::GpuMat imageGpu;
    
    std::vector<cv::Point2f> points2dVec;
    std::vector<cv::KeyPoint> keypoints2dVec;

    cv::Ptr<cv::cuda::FastFeatureDetector> gpuFastDetector = cv::cuda::FastFeatureDetector::create(70, true, 2);
    
    cv::Ptr<cv::cuda::Filter> filter2 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 0);

    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    
    cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, kernel, cv::Point(-1, -1), 3);
    cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, kernel, cv::Point(-1, -1), 3);

    double points_3d[] = {-165.1, 165.1, 0, 165.1, 165.1, 0, 165.1, -165.1, 0, -165.1, -165.1, 0};
    cv::Mat1d points_3dMat(4, 3, points_3d);
    
    float K[] = {802.9265702293416, 0, 966.4221440154661, 0, 803.6309422064642, 477.6409613889424, 0, 0, 1};
    cv::Mat1f kMat(3, 3, K);

    int distCoeffs[5] = {}; 
    cv::Mat1i distMat(1, 5, distCoeffs);
    
    
    cap.read(frame);
    imageGpu.upload(frame);
    
    cv::cuda::cvtColor(imageGpu, imageGpu, cv::COLOR_BGR2GRAY);    
    dilate -> apply(imageGpu, imageGpu);
    erode -> apply(imageGpu, imageGpu);
    
    gpuFastDetector -> detect(imageGpu, keypoints2dVec);
    imageGpu.download(frame);
    /*
    for(int i = 0; i < keypoints2dVec.size(); i++){
        points2dVec.push_back(keypoints2dVec[i].pt);
    }
    
    std::vector<cv::Point2f> taggedCorners = tagCorners(points2dVec);
    
    for(int i = 0; i < points2dVec.size(); i++){
        cv::circle(frame, cv::Point2f(taggedCorners[i].x, taggedCorners[i].y), 2, cv::Scalar(0,255,255));
    }
    */
    cv::drawKeypoints(frame, keypoints2dVec, frame, cv::Scalar(255,0,0));


    cv::imshow("frame", frame);
    cv::waitKey(0);
    //cv::cuda::solvePnPRansac(points_3dMat, points_2d, kMat, distMat, rvecs, tvecs);   
    
}



