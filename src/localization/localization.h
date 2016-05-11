#ifndef LOCALIZATION_H
#define LOCALIZATION_H
//#include "globals.h"
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <QFile>
#include <QRgb>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <vector>

#include <pcl-1.5/pcl/common/common_headers.h>
#include <pcl-1.5/pcl/filters/voxel_grid.h>
#include <pcl-1.5/pcl/common/transforms.h>
#include <pcl-1.5/pcl/impl/point_types.hpp>
//#include <pcl-1.5/pcl/point_types_conversion.h>
#include <pcl-1.5/pcl/io/pcd_io.h>

class localization
{
public:
    localization();

    static cv::Mat selectReadFilter(int filter_number, std::string filters_dir_path);
    static cv::Mat calculateInvariants(cv::Mat bgr_image, pcl::PointCloud<pcl::PointXYZRGB> normalCloud, cv::Mat filters[5], int noHarmonics, bool normalize_invariants);
    static cv::Mat onlineLocationEstimation(cv::Mat invariant_matrix, cv::Mat omni_invariants, cv::Mat locations,int base_point_number, int orientation_number);

};

#endif
