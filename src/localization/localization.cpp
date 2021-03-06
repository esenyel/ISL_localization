#include "localization.h"
#include "imageprocess.h"
#include "bubbleprocess.h"
#include <ros/ros.h>
#include <QStringList>
#include <QTextStream>
#include "qstring.h"
#include <math.h>
#include <QDebug>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <QFile>
#include <QRgb>
#include <opencv2/opencv.hpp>


cv::Mat localization::selectReadFilter(int filter_number, std::string filters_dir_path){

    cv::Mat filter_matrix;
    std::stringstream ss;
    ss << filter_number;
    std::string str_filter_no = ss.str();

    std::string filter_path = filters_dir_path + "/filtre" + str_filter_no + ".txt";

    QString filter_paths = QString::fromStdString(filter_path);

    //Read the filters and store the kernels in an array
    filter_matrix = ImageProcess::mstReadFilter(filter_paths,29,false,false,false);
    ROS_INFO("Filters are read successfully!");
    return filter_matrix;
}
cv::Mat localization::calculateInvariants(cv::Mat bgr_image, pcl::PointCloud<pcl::PointXYZRGB> normalCloud, cv::Mat filters[5],  int noHarmonics, bool normalize_invariants){
    cv::Mat resg;
    cv::Mat invariants(1,noHarmonics*noHarmonics,CV_32FC1);
    int satLower = 30;
    int satUpper = 230;
    int valLower = 30;
    int valUpper = 230;
    int focalLengthPixels = 525;
    std::vector<bubblePointXYZ> bubble;
    cv::Mat normalizedDummyVector;
    double maxRangeMeters = 6;

    for(unsigned int i=0; i<5; i++){
        //Convert bgr image to gray image
        cv::cvtColor(bgr_image,resg,CV_BGR2GRAY);
        //Apply the filter
        cv::Mat sonuc = ImageProcess::mstApplyFilter(resg, filters[i]);

        //Calculate bubbles and then the invariant vector for this filter
        vector<bubblePoint> imgBubble = bubbleProcess::convertGrayImage2Bub(sonuc,focalLengthPixels,255);
        vector<bubblePoint> resred;
        resred = bubbleProcess::reduceBubble(imgBubble);
        DFCoefficients dfcoeff =  bubbleProcess::calculateDFCoefficients(resred,noHarmonics,noHarmonics);
        //Normalize the invariant and append to the overall invariant vector
        if(i==0){
            if(normalize_invariants == true){
                cv::normalize(bubbleProcess::mstCalculateInvariants(resred, dfcoeff,noHarmonics, noHarmonics), invariants);
            }
            else{
                invariants = bubbleProcess::mstCalculateInvariants(resred, dfcoeff,noHarmonics, noHarmonics);
            }
        }
        else{
            if(normalize_invariants == true){
                cv::normalize(bubbleProcess::mstCalculateInvariants(resred,dfcoeff,noHarmonics,noHarmonics), normalizedDummyVector);
                cv::hconcat(invariants, normalizedDummyVector, invariants);
            }
            else{
                cv::hconcat(invariants, bubbleProcess::mstCalculateInvariants(resred,dfcoeff,noHarmonics,noHarmonics), invariants);
            }
        }
    }

    //For the hue image, calculate bubbles, the invariant vector, normalize it and append to the overall vector
    cv::Mat hueChannel= ImageProcess::generateChannelImage(bgr_image,0,satLower,satUpper,valLower,valUpper);
    vector<bubblePoint> hueBubble = bubbleProcess::convertGrayImage2Bub(hueChannel,focalLengthPixels,180);
    vector<bubblePoint> reducedHueBubble = bubbleProcess::reduceBubble(hueBubble);
    DFCoefficients dfcoeffRGB = bubbleProcess::calculateDFCoefficients(reducedHueBubble,noHarmonics,noHarmonics);
    if(normalize_invariants == true){
        cv::normalize(bubbleProcess::mstCalculateInvariants(reducedHueBubble,dfcoeffRGB,noHarmonics,noHarmonics), normalizedDummyVector);
        cv::hconcat(invariants, normalizedDummyVector, invariants);
    }
    else{
        cv::hconcat(invariants, bubbleProcess::mstCalculateInvariants(reducedHueBubble,dfcoeffRGB,noHarmonics,noHarmonics), invariants);
    }

    //rotation around X and Y to get same coordinates with RGB frame and database images
    int rotX=-90;
    int rotY=90;
    int rotZ=0;

    Eigen::Matrix4f yy;

        double xRad = rotX*M_PI/180;

        double yRad = rotY*M_PI/180;

        double zRad = rotZ*M_PI/180;

        yy<<cos(yRad), 0, sin(yRad), 0,
            0, 1, 0, 0,
            -sin(yRad), 0, cos(yRad), 0,
            0, 0, 0, 1;

        Eigen::Matrix4f xx;

        xx<<1, 0, 0, 0,
            0, cos(xRad), -sin(xRad), 0,
            0, sin(xRad), cos(xRad), 0,
            0, 0, 0, 1;


        Eigen::Matrix4f zz;

        zz<<cos(zRad),-sin(zRad), 0, 0,
            sin(zRad), cos(zRad), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
    //rotation
    pcl::transformPointCloud(normalCloud,normalCloud,xx*yy*zz);

        //Create the bubble points out of the depth information
        for(unsigned int i = 0; i < normalCloud.points.size(); i++)
        {
            bubblePointXYZ pt;
            pt.x = normalCloud.points.at(i).x;
            pt.y = normalCloud.points.at(i).y;
            pt.z = normalCloud.points.at(i).z;
            bubble.push_back(pt);
        }

    //For the depth image, calculate bubbles, the invariant vector, normalize it and append to the overall vector
    vector<bubblePoint> sphBubble = bubbleProcess::convertBubXYZ2BubSpherical(bubble,maxRangeMeters);
    vector<bubblePoint> sphRedBubble = bubbleProcess::reduceBubble(sphBubble);
    DFCoefficients dfcoeff = bubbleProcess::calculateDFCoefficients(sphRedBubble,noHarmonics,noHarmonics);
    if(normalize_invariants == true){
        cv::normalize(bubbleProcess::mstCalculateInvariants(sphRedBubble, dfcoeff,noHarmonics, noHarmonics), normalizedDummyVector);
        cv::hconcat(invariants, normalizedDummyVector, invariants);
    }
    else{
        cv::hconcat(invariants, bubbleProcess::mstCalculateInvariants(sphRedBubble, dfcoeff,noHarmonics, noHarmonics), invariants);
    }
    return invariants;

}

// location estimation function of incoming images
cv::Mat localization::onlineLocationEstimation(cv::Mat invariant_matrix, cv::Mat omni_invariants, cv::Mat locations, int base_point_number, int orientation_number){

    // initialization of matrices used for location estimation
    cv::Mat location_estimation(1, 2, CV_64F);;
    cv::vconcat(omni_invariants, omni_invariants,omni_invariants);
    cv::Mat I[base_point_number];
    cv::Mat summation(base_point_number,orientation_number, CV_64F);
    cv::Mat gamma(base_point_number,1,CV_64F) ;
    cv::Mat k(1,base_point_number,CV_64F);
    double K_c=0;
    double dummy_array[orientation_number];

    // splitting the invariant matrix (which is invariants from all points visited before) according to the point number
    for(int i=0; i<base_point_number; i++ ){
        cv::Mat dummy_invariants = invariant_matrix.rowRange(orientation_number*i , (orientation_number*(i+1)));
        dummy_invariants.copyTo(I[i]);

    }

    // calculating the summation of the difference betweeen current invariant and database invariants
    for (int m=0; m<base_point_number; m++){
        for (int k=0; k<orientation_number; k++){
            summation.at<double>(m,k)=0;
            for (int i=0; i<orientation_number; i++){

                summation.at<double>(m,k)=summation.at<double>(m,k)+norm(omni_invariants.row(k+i)-I[m].row(i));
                dummy_array[k]= summation.at<double>(m,k);

            }
            std::cout << dummy_array[k] << std::endl;
        }

        // minimizing this summation with respect to k (in order to find the angle with maximum correlation)
        double minVal = *std::min_element(dummy_array,dummy_array+8);

        // calculating gamma values for each database points
        gamma.at<double>(m,0) = minVal;

        // calculating k values for each database points
        k.at<double>(0,m)=exp(-pow(minVal,2));

    }

    // summation of elements of k vector
    for (int m=0; m<base_point_number; m++){
        K_c=K_c+k.at<double>(0,m);
    }

    // location estimation calculated by location matrix, k vector and K_c value
    location_estimation = (1/K_c)*k*locations;

    return location_estimation;

}

