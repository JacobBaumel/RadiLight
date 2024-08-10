#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaarithm.hpp"
#include <vector>
#include <opencv2/cudafeatures2d.hpp>

std::vector<cv::Point2f> checkIfPointNearLine(const cv::Point2f& pt1, const cv::Point2f& pt2, std::vector<cv::Point2f> corners) {
    std::vector<cv::Point2f> newCorners;
    for (const auto& pt : corners) {
        double d = std::abs((pt2.x - pt1.x) * (pt1.y - pt.y) - (pt1.x - pt.x) * (pt2.y - pt1.y)) / cv::norm(pt2 - pt1);
        if (d >= 10) {
            newCorners.push_back(pt);
        }
    }
    return newCorners;
}

//tt Function to tag corners
std::vector<cv::Point2f> tagCorners(std::vector<cv::Point2f> corners) {
    if (corners.size() < 4) {
        throw std::invalid_argument("Not enough corners provided");
    }

    auto x_min = std::min_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    auto y_min = std::min_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    auto x_max = std::max_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    auto y_max = std::max_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    
    cv::Point2f pt1 = *x_min;
    cv::Point2f pt2 = *y_max;
    cv::Point2f pt3 = *x_max;
    cv::Point2f pt4 = *y_min;

    corners.erase(std::remove(corners.begin(), corners.end(), pt1), corners.end());
    corners.erase(std::remove(corners.begin(), corners.end(), pt2), corners.end());
    corners.erase(std::remove(corners.begin(), corners.end(), pt3), corners.end());
    corners.erase(std::remove(corners.begin(), corners.end(), pt4), corners.end());

    corners = checkIfPointNearLine(pt1, pt2, corners);
    corners = checkIfPointNearLine(pt2, pt3, corners);
    corners = checkIfPointNearLine(pt3, pt4, corners);  
    corners = checkIfPointNearLine(pt4, pt1, corners);

    x_min = std::min_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    y_min = std::min_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    x_max = std::max_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    y_max = std::max_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });  
    
    cv::Point2f leftmost_corner = *x_min;
    cv::Point2f bottommost_corner = *y_max;
    cv::Point2f rightmost_corner = *x_max;
    cv::Point2f topmost_corner = *y_min;
  
    std::vector<cv::Point2f> tag_corners = { leftmost_corner, bottommost_corner, rightmost_corner, topmost_corner };
    return tag_corners;
}

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
    cv::imwrite("yuh.jpg", frame);
    cv::waitKey(0);
    //cv::cuda::solvePnPRansac(points_3dMat, points_2d, kMat, distMat, rvecs, tvecs);   
    
}



