
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorDistType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorDistType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
        if (descSource.type() != CV_32F)
        { 
            descSource.convertTo(descSource, CV_32F);
        }

        if (descRef.type() != CV_32F)
        {
            descRef.convertTo(descRef, CV_32F);
        }

        // Implement FLANN matching (used L2_Norm by default to create a kd-tree)
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << matcherType << " matcher (with " << selectorType << " selector) with n=" << matches.size() << "(out of: " << kPtsRef.size() << ")" << " matched keypoints; " << "matched in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        int k = 2;
        vector<vector<cv::DMatch>> knn_matches;

        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto item = knn_matches.begin(); item != knn_matches.end(); ++item)
        {
            if ((*item)[0].distance < minDescDistRatio * (*item)[1].distance)
            {
                // this means that best match has much lower distance than second-best match 
                // and most probably is a good match (not a False Positive)
                matches.push_back((*item)[0]);
            }
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << matcherType << " matcher (with " << selectorType << " selector) with n=" << matches.size() << " (out of: " << kPtsRef.size() << ")" << " matched keypoints; " << "matched in " << 1000 * t / 1.0 << " ms" << endl;
    }


}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::SIFT::create();
    }
    else
    {
        throw invalid_argument("Invalid descriptorType: " + descriptorType + "; should be on of: [BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT]");
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extracted in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " detected keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 4;           //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int apertureSize = 5;        //  aperture parameter for the Sobel operator (usually odd number larger than blockSize)
    int k = 0.04;                //  controls the sensitivity of the corner detector (in corner respose R; suggested: 0.04 - 0.06); smaller -> more sensitive -> more corners detected --> more false positives
    int minResponse = 15;        //  minimum value for a corner in the 8bit scaled response matrix
    double maxOverlap = 0.0;     //  max. permissible overlap between two features in %, used during non-maxima suppression
    double t = (double)cv::getTickCount();

    cv::Mat cornerness, cornernessNorm; 
    cornerness = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, cornerness, blockSize, apertureSize, k);
    cv::normalize(cornerness, cornernessNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // add corners to result vector
    for (int y = 0; y < cornernessNorm.rows; y++)
    {
        for (int x = 0; x < cornernessNorm.cols; x++)
        {
            int response = (int)cornernessNorm.at<float>(y, x);
            if ( response > minResponse)  // only store points above threshold
            {
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(x, y);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                bool bOverlap = false;
                for (auto item = keypoints.begin(); item != keypoints.end(); ++item)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *item);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*item).response)
                        {
                            // if overlapping and new response is stronger --> use the new keypoint
                            *item = newKeyPoint;
                            break;
                        }
                    }
                }
                if (!bOverlap)
                {
                    // only add keypoints if it's not overlapping
                    keypoints.push_back(newKeyPoint);
                }
            }

        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris-Corner detection with n=" << keypoints.size() << " detected keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris-Corner Detector Results";
        cv::namedWindow(windowName, 7);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;    // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;      // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        // This uses the 16 surrounding pixels to detect whether a pixel is a corner, requiring a contiguous set of 9 out of 16 pixels to be either darker or lighter by the threshold.
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detector = cv::BRISK::create();
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector = cv::SIFT::create();
    }
    else
    {
        throw invalid_argument("Invalid detectorType: " + detectorType + "; should be on of: [FAST, BRISK, ORB, AKAZE, SIFT]");
    }

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " with n=" << keypoints.size() << " detected keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Keypoint Detector Results";
        cv::namedWindow(windowName, 7);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
