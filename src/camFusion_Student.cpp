
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.3f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 6);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    boundingBox.kptMatches.clear();     // was populated when matching bounding-boxes across two subsequent frames
    vector<cv::DMatch> matchesInROI;
    vector<double> distanceVec;

    float shrinkFactor = 0.10;
    cv::Rect smallerBox;
    smallerBox.x = boundingBox.roi.x + shrinkFactor * boundingBox.roi.width / 2.0;
    smallerBox.y = boundingBox.roi.y + shrinkFactor * boundingBox.roi.height / 2.0;
    smallerBox.width = boundingBox.roi.width * (1 - shrinkFactor);
    smallerBox.height = boundingBox.roi.height * (1 - shrinkFactor);

    for (const auto& match : kptMatches)
    {
        cv::KeyPoint* prevPt = &kptsPrev[match.queryIdx]; 
        cv::KeyPoint* currPt = &kptsCurr[match.trainIdx];
        if (smallerBox.contains(prevPt->pt) && smallerBox.contains(currPt->pt))
        {
            // if the matches are correct; then the points should be in similar positions in the two subsequent frames
            // hence, points from prev frame that is matched should also be inside the current bounding-box (this could itself filter some outliers)
            distanceVec.push_back(sqrt((prevPt->pt.x - currPt->pt.x)*(prevPt->pt.x - currPt->pt.x) + (prevPt->pt.y - currPt->pt.y)*(prevPt->pt.y - currPt->pt.y)));
            matchesInROI.push_back(match);
        }
    }

    double meanDistance = accumulate(distanceVec.begin(), distanceVec.end(), 0.0) / distanceVec.size();
    double variance = 0.0;
    for (double val : distanceVec)
    {
        variance += (val - meanDistance)*(val - meanDistance); 
    }
    variance /= (distanceVec.size() - 1);       // sample variance
    double stddevDistance = sqrt(variance);

    // key matched points that are within 2*stddev distance from the mean
    for (const auto& match : matchesInROI)
    {
        cv::KeyPoint* prevPt = &kptsPrev[match.queryIdx]; 
        cv::KeyPoint* currPt = &kptsCurr[match.trainIdx];
        double distance = sqrt((prevPt->pt.x - currPt->pt.x)*(prevPt->pt.x - currPt->pt.x) + (prevPt->pt.y - currPt->pt.y)*(prevPt->pt.y - currPt->pt.y));
        if (abs(distance - meanDistance) < 2.0*stddevDistance)
        {
            boundingBox.kptMatches.push_back(match);
        }
    }

    cout << "Removed " << int(matchesInROI.size()) - int(boundingBox.kptMatches.size()) << " kpt outliers from bounding-box " 
         << boundingBox.boxID << endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, double &meanDistChange,double &stddevDistChange,int &distRatioVecSize, cv::Mat *visImg)
{
    // Calculate Distance change between kpts in two subsequent frames
    double minDist = 100.0; // min. required distance (to not choose points very close to each other)
    vector<double> distChangeVec;
    vector<double> distanceCurrVec;
    vector<double> distancePrevVec;
    for (auto it1 = kptMatches.begin(); it1 != std::prev(kptMatches.end()); ++it1)
    {
        cv::KeyPoint* prevPt1 = &kptsPrev.at(it1->queryIdx);
        cv::KeyPoint* currPt1 = &kptsCurr.at(it1->trainIdx);
        // start from the one element ahead of outer iterator to prevent repeated distance calculation
        for (auto it2 = std::next(it1); it2 != kptMatches.end(); ++it2)
        {
            cv::KeyPoint* prevPt2 = &kptsPrev.at(it2->queryIdx);
            cv::KeyPoint* currPt2 = &kptsCurr.at(it2->trainIdx);
            double distPrev = sqrt((prevPt2->pt.x - prevPt1->pt.x)*(prevPt2->pt.x - prevPt1->pt.x) + (prevPt2->pt.y - prevPt1->pt.y)*(prevPt2->pt.y - prevPt1->pt.y));
            double distCurr = sqrt((currPt2->pt.x - currPt1->pt.x)*(currPt2->pt.x - currPt1->pt.x) + (currPt2->pt.y - currPt1->pt.y)*(currPt2->pt.y - currPt1->pt.y));
            if(distPrev > numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                distChangeVec.push_back(distCurr - distPrev);
                distanceCurrVec.push_back(distCurr);
                distancePrevVec.push_back(distPrev);
            }
        }
    }

    meanDistChange = accumulate(distChangeVec.begin(), distChangeVec.end(), 0.0) / distChangeVec.size();
    double variance = 0.0;
    for (double val : distChangeVec)
    {
        variance += (val - meanDistChange)*(val - meanDistChange); 
    }
    variance /= (distChangeVec.size() - 1);       // sample variance
    stddevDistChange = sqrt(variance);
    cout << "meanDistanceChange: " << meanDistChange << " | stddevDistChange: " << stddevDistChange << endl;

    vector<double> distRatioVec;
    for (int i = 0; i < distChangeVec.size(); ++i)
    {
        if (abs(distChangeVec[i] - meanDistChange) < 2.0*stddevDistChange)
        {
            distRatioVec.push_back(distanceCurrVec[i] / distancePrevVec[i]);
        }
    }
    distRatioVecSize = (int)distRatioVec.size();
    cout << "Number of elements in distRatioVec: " << distRatioVecSize << endl;

    // only continue if list of distance ratios is not empty
    if (distRatioVec.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from the median of distance ratios
    sort(distRatioVec.begin(), distRatioVec.end());
    int vecSize = distRatioVec.size();
    double medianDistRatio;
    if (vecSize % 2 == 0)
    {
        // even number of points; need to calculate the mean of two elements in the middle of the sorted array
        medianDistRatio = (distRatioVec[vecSize/2 - 1] + distRatioVec[vecSize/2]) / 2;
    }
    else
    {
        // odd number of points; just take the middle element of the sorted array
        medianDistRatio = distRatioVec[vecSize/2];
    }

    double dt = 1.0 / (frameRate + 1e-8);
    double denom = 1 - medianDistRatio;
    if (abs(denom) < 1e-6)
    {
        TTC = NAN;
        cerr << "Warning: Calculated medianDistRatio is close to 1 (i.e.: no scale change): " << medianDistRatio << endl;
    }
    else
    {
        TTC = -dt / denom;
        cout << "Camera-TTC= " << TTC << endl;
    }
    
}

pcl::PointCloud<pcl::PointXYZ>::Ptr convertToPointCloud(const std::vector<LidarPoint>& lidarPoints)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // reserve memory the cloud to match the number of lidar points
    cloud->points.reserve(lidarPoints.size());

    for (const auto& point : lidarPoints)
    {
        // Add each LidarPoint as a pcl::PointXYZ
        cloud->points.push_back(pcl::PointXYZ(point.x, point.y, point.z));
    }

    // Set the size of the cloud
    cloud->width = cloud->points.size();
    cloud->height = 1; // 1 for unorganized point cloud (i.e.: list of points)

    return cloud;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // To be statistically robust we need to cluster the lidar points 
    // and then find the min distance accross points in all clusters.
    cout << "Number of Lidar Points in Current Frame: " << lidarPointsCurr.size() << " | in Previous Frame: " << lidarPointsPrev.size() << endl;
    // First, convert LidarPoint into PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr prevCloud = convertToPointCloud(lidarPointsPrev);
    pcl::PointCloud<pcl::PointXYZ>::Ptr currCloud = convertToPointCloud(lidarPointsCurr);

    // Create a KD-Tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

    // Create the clustering object (similar to DBSCAN algorithm)
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.15); // 20cm
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(2000);
    ec.setSearchMethod(tree);

    // Run clustering on the current frame
    std::vector<pcl::PointIndices> clusterIndicesCurr;
    ec.setInputCloud(currCloud->makeShared());
    ec.extract(clusterIndicesCurr);

    // Run clustering on the previous frame
    std::vector<pcl::PointIndices> clusterIndicesPrev;
    ec.setInputCloud(prevCloud->makeShared());
    ec.extract(clusterIndicesPrev);

    if (clusterIndicesCurr.empty() || clusterIndicesPrev.empty())
    {
        // This means we have insufficient data to correctly calculate the TTC
        // returning large positive number
        TTC = NAN;
        cerr << "Warning: Insufficient Data to Calculate TTC from LIDAR Points" << endl;
        return;
    }

    // Find the point with minimum distance (iterating over clusters found) for both current and previous frame
    double minDistanceXCurr = numeric_limits<double>::max();
    int numPointsClusteredCurr = 0;
    for (size_t i = 0; i < clusterIndicesCurr.size(); ++i)
    {
        int N = clusterIndicesCurr[i].indices.size(); // number of points in cluster
        numPointsClusteredCurr += N;

        int P;
        if (N >= 100) {
            P = N/100; // 1st percentile
        } else if (N >= 10) {
            P = N/10; // 10th percentile
        } else {
            P = N/2; // median
        }

        vector<double> xValues(N);
        for (int j = 0; j < N; ++j)
        {
            xValues[j] = (*currCloud)[clusterIndicesCurr[i].indices[j]].x;
        }

        sort(xValues.begin(), xValues.end());
        if (xValues[P] < minDistanceXCurr)
        {
            // calculating minimum X distance based on percentiles in each cluster (for robustness).
            minDistanceXCurr = xValues[P];
        }
    }
    cout << "Number of Discarded-Lidar-Points in the Current-Frame: " << lidarPointsCurr.size() - numPointsClusteredCurr << endl;
    cout << "minDistanceXCurr: " << minDistanceXCurr << endl;

    double minDistanceXPrev = numeric_limits<double>::max();
    int numPointsClusteredPrev = 0;
    for (size_t i = 0; i < clusterIndicesPrev.size(); ++i)
    {
        int N = clusterIndicesPrev[i].indices.size(); // number of points in cluster
        numPointsClusteredPrev += N;

        int P;
        if (N >= 100) {
            P = N/100; // 1st percentile
        } else if (N >= 10) {
            P = N/10; // 10th percentile
        } else {
            P = N/2; // median
        }

        vector<double> xValues(N);
        for (int j = 0; j < N; ++j)
        {
            xValues[j] = (*prevCloud)[clusterIndicesPrev[i].indices[j]].x;
        }

        sort(xValues.begin(), xValues.end());
        if (xValues[P] < minDistanceXPrev)
        {
            // calculating minimum X distance based on percentiles in each cluster (for robustness).
            minDistanceXPrev = xValues[P];
        }
    }
    cout << "Number of Discarded-Lidar-Points in the Previous-Frame: " << lidarPointsPrev.size() - numPointsClusteredPrev << endl;
    cout << "minDistanceXPrev: " << minDistanceXPrev << endl;

    // Computing Time-To-Collision using Constant Velocity Model (CVM) both for x (forward driving direction) 
    // added small value to avoid division by zero
    double dt = 1.0 / (frameRate + 1e-8);
    double relVelX = (minDistanceXCurr - minDistanceXPrev) / (dt + 1e-8);    
    if (abs(relVelX) < 0.0001)
    {
        TTC = NAN;
        cerr << "Warning: Calculated Relative-Velocity is close to zero: " << relVelX << endl;
    }
    else
    {
        TTC = -minDistanceXCurr / relVelX;
    }
    

    cout << "Found " << clusterIndicesCurr.size() << " lidar-cluster(s) in current frame | Found " 
         << clusterIndicesPrev.size() << " lidar-cluster(s) in previous frame\n" << "Lidar-TTC= " << TTC << " seconds" << endl;

    return;
}


void helperMatchCurrBox(BoundingBox& currBox, const DataFrame& prevFrame, map<pair<int, int>, int>& countPairs , map<int, int>& bestMatches, map<int, pair<int, float>>& alreadyMatchedBB, map<int, BoundingBox*>& boxMap)
{
    // criterion: confidence_ratio = num common matched kpts between bounding-boxes / the total matched kpts in the currBounding-Box.
    float confidenceRatioThreshold = 0.5;
    int currBoxID = currBox.boxID;
    int totalCurrMatchKpts = currBox.kptMatches.size();
    boxMap[currBoxID] = &currBox;

    int maxOverlapMatches = 0;
    pair<int, int> bestMatchPair;
    for (auto& prevBox : prevFrame.boundingBoxes)
    {
        pair<int, int> pairKey = make_pair(prevBox.boxID, currBox.boxID);
        int numOverlapKpts = countPairs[pairKey];
    
        if (alreadyMatchedBB.find(prevBox.boxID) != alreadyMatchedBB.end())
        {
            // if prevBox is already matched; check the confidence ratios
            float confidenceRatio = (float)numOverlapKpts / (float)totalCurrMatchKpts;
            if (confidenceRatio > alreadyMatchedBB[prevBox.boxID].second)
            {
                // if it's larger then consider updating the maxOverlapMatches and bestMatchPair variables
                // but don't update the alreadyMatchedBB (it will be updated later); because maxOverlapMatches and bestMatchPair still not finalized
                if (numOverlapKpts > maxOverlapMatches)
                {
                    maxOverlapMatches = numOverlapKpts;
                    bestMatchPair = pairKey;
                }
            }
        }
        else
        {
            if (numOverlapKpts > maxOverlapMatches)
            {
                maxOverlapMatches = numOverlapKpts;
                bestMatchPair = pairKey;
            }
        }
    }

    float bestConfidenceRatio = (float)maxOverlapMatches / (float)totalCurrMatchKpts;
    if (bestConfidenceRatio > confidenceRatioThreshold)
    {
        if (alreadyMatchedBB.find(bestMatchPair.first) != alreadyMatchedBB.end())
        {
            // means if key (i.e. prevBoxID) exists --> is already used to matched BB
            // so it must have larger confidence ratio; otherwise, it wouldn't get added in the first place
            assert (bestConfidenceRatio > alreadyMatchedBB[bestMatchPair.first].second);

            // need to find new pair for this boxID from current frame 
            // (because it will get replaced by the new pair)
            int replacedCurrBoxID = bestMatches[bestMatchPair.first]; 

            // now replace the new pair (i.e.: the better/more-confident pair)
            alreadyMatchedBB[bestMatchPair.first] = make_pair(bestMatchPair.second, bestConfidenceRatio);
            bestMatches[bestMatchPair.first] = bestMatchPair.second;

            // and find the new match for the one that is being replaced
            helperMatchCurrBox(*boxMap[replacedCurrBoxID], prevFrame, countPairs , bestMatches, alreadyMatchedBB, boxMap);

        }
        else
        {
            bestMatches[bestMatchPair.first] = bestMatchPair.second;
            alreadyMatchedBB[bestMatchPair.first] = make_pair(bestMatchPair.second, bestConfidenceRatio);
        }
    }
    else
    {
        cerr << "Warning: bounding-box couldn't match; the confidence-ratio = " << bestConfidenceRatio
             << " which is below the threshold= " << confidenceRatioThreshold << endl;
    }

}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, bool bVis)
{
    // Note: matched kpt could be at the intersection of multiple bounding-boxes and we add this point to all of them 
    // (don't ignore them, since number of matched keypoints might not be large and we can still match the bounding-boxes based on the highest number of matched keypoints). 

    map<pair<int, int>, int> countBoxOverlap;
    for (const auto& matchKpt: matches)
    {
        for (auto& currBox : currFrame.boundingBoxes)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            float shrinkFactor = 0.10;
            cv::Rect smallerBox;
            smallerBox.x = currBox.roi.x + shrinkFactor * currBox.roi.width / 2.0;
            smallerBox.y = currBox.roi.y + shrinkFactor * currBox.roi.height / 2.0;
            smallerBox.width = currBox.roi.width * (1 - shrinkFactor);
            smallerBox.height = currBox.roi.height * (1 - shrinkFactor);

            // in the matchKpt, trainIdx is the index in the currFrame
            if (smallerBox.contains(currFrame.keypoints[matchKpt.trainIdx].pt))
            {
                currBox.kptMatches.push_back(matchKpt);

                // Now, loop over the Bounding-Boxes from the prevFrame to find the match
                for (auto& prevBox: prevFrame.boundingBoxes)
                {
                    cv::Rect smallerBox;
                    smallerBox.x = prevBox.roi.x + shrinkFactor * prevBox.roi.width / 2.0;
                    smallerBox.y = prevBox.roi.y + shrinkFactor * prevBox.roi.height / 2.0;
                    smallerBox.width = prevBox.roi.width * (1 - shrinkFactor);
                    smallerBox.height = prevBox.roi.height * (1 - shrinkFactor);

                    // in the matchKpt, queryIdx is the index in the prevFrame
                    if (smallerBox.contains(prevFrame.keypoints[matchKpt.queryIdx].pt))
                    {
                        countBoxOverlap[make_pair(prevBox.boxID, currBox.boxID)]++;
                    }
                }
            }
        }
    }

    // We have stored the num kpt matches between all pairs of bounding-boxes; now we can find the ones that best match each other    
    // set of bounding-boxes from prevFrame that are already matched with one of currFrame bounding-boxes   
    map<int, pair<int, float>> alreadyMatchedBB;    // key: prevBoxID, pair: <currBoxID, confidenceRatio>
    map<int, BoundingBox*> currProcessedBoxMap;

    for (auto& currBox : currFrame.boundingBoxes)
    {
        helperMatchCurrBox(currBox, prevFrame, countBoxOverlap, bbBestMatches, alreadyMatchedBB, currProcessedBoxMap);
    }

    cout << "number of bounding-boxes in prev-frame: " << prevFrame.boundingBoxes.size() << " | in curr-frame: " << currFrame.boundingBoxes.size()
         << " | number of matched-bounding-boxes: " << bbBestMatches.size() << endl;

    if (bVis)
    {
        cv::Mat currImg = (currFrame.cameraImg).clone();
        cv::Mat prevImg = (prevFrame.cameraImg).clone();
        cv::RNG rng(0xFFFFFFFF); // Seed for random number

        for (auto& bbmatch : bbBestMatches)
        {
            int prevBoxId = bbmatch.first;
            int currBoxId = bbmatch.second;
            cv::Rect prevBox, currBox;
            
            for (auto& box : prevFrame.boundingBoxes)
            {
                if (box.boxID == prevBoxId)
                {
                    prevBox = box.roi;
                    break;
                }
            }

            for (auto& box : currFrame.boundingBoxes)
            {
                if (box.boxID == currBoxId)
                {
                    currBox = box.roi;
                    break;
                }
            }

            int icolor = (unsigned) rng;
            auto color = cv::Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
            // the operation icolor&255 effectively takes the last 8 bits of icolor (AND '&' is a bitwise operation)
            // (the least significant 8 bits, or the rightmost 8 bits in binary representation), 
            // discards the rest, and this is how we get the blue channel value for our color.
            // icolor>>8: shifts the random number 8 bits to the right, effectively discarding the least significant 8 bits

            // Draw bounding box
            cv::rectangle(prevImg, prevBox, color, 2);
            cv::rectangle(currImg, currBox, color, 2);
        }

        cv::Mat matchImg;
        int visConfig = 2;  // [1: bounding boxes, 
                            //  2: bounding-boxes + match-kpts, 
                            //  3: bounding-boxes + all-kpts + match-lines]
        
        if (visConfig == 1)
        {
            // Concatenate images horizontally
            cv::hconcat(prevImg, currImg, matchImg);
        }
        else if (visConfig == 2)
        {
            // extract keypoints for drawing
            std::vector<cv::KeyPoint> prevKeypoints, currKeypoints;
            for (auto& currBox : currFrame.boundingBoxes)
            {
                for (auto& match : currBox.kptMatches)
                {
                    prevKeypoints.push_back(prevFrame.keypoints[match.queryIdx]);
                    currKeypoints.push_back(currFrame.keypoints[match.trainIdx]);
                }
            }
            cv::drawKeypoints(prevImg, prevKeypoints, prevImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::drawKeypoints(currImg, currKeypoints, currImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            
            // Concatenate images horizontally
            cv::hconcat(prevImg, currImg, matchImg);

        }
        else if (visConfig == 3)
        {
            // to plot keypoints matches as well
            std::vector<cv::DMatch> currBBkptMatches;
            for (auto& currBox : currFrame.boundingBoxes)
            {
                currBBkptMatches.insert(currBBkptMatches.end(), currBox.kptMatches.begin(), currBox.kptMatches.end());
            }
            cv::drawMatches(prevImg, prevFrame.keypoints,
                            currImg, currFrame.keypoints,
                            currBBkptMatches, matchImg,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }

        // Display the Image
        cv::namedWindow("Matched Bounding Boxes", cv::WINDOW_NORMAL);
        cv::imshow("Matched Bounding Boxes", matchImg);
        cv::waitKey(0); // Wait for a keystroke in the window
    }
}
