#ifndef PCPROCESSING_H
#define PCPROCESSING_H
#include <pcl-1.5/pcl/io/pcd_io.h>
#include <pcl-1.5/pcl/point_types.h>
#include <pcl-1.5/pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d_omp.h>

#include <QString>
#include <QStringList>
#include <opencv2/opencv.hpp>

class PCprocessing
{
public:

    PCprocessing();

    static void setDataSetPath(QString dir);
    static QString getDataSetPath();
    static void setDataSetItems(QStringList items);
    static int getNumofItems();

     void setViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer> viwer);

   // bool loadItem(int itemNumber, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

   // void applyVoxelGridFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr input);

     bool loadItem(int itemNumber, QString fileName, sensor_msgs::PointCloud2::Ptr cloud);

     bool loadItem(QString filePath, sensor_msgs::PointCloud2::Ptr cloud);

     void applyVoxelGridFilter(sensor_msgs::PointCloud2::Ptr input);

     void calculateNormals(sensor_msgs::PointCloud2::Ptr input);

     bool saveNormalAngleHistogram(pcl::PointCloud<pcl::Normal>::Ptr normals, int itemNumber);

     bool saveNormalPanTiltHistograms(pcl::PointCloud<pcl::Normal>::Ptr normals);

     void rotatePointCloud(sensor_msgs::PointCloud2::Ptr input, int rotX, int rotY, int rotZ);

     void scalePointCloud(sensor_msgs::PointCloud2::Ptr input, double scale);

     bool savePointCloud(int itemNumber, QString fileName);

     sensor_msgs::PointCloud2::Ptr getCurrentCloud();

     pcl::PointCloud<pcl::Normal>::Ptr getCurrentCloudNormals();

     void showPointCloud(sensor_msgs::PointCloud2::Ptr cloud);

     void showPointCloud(pcl::PointCloud<pcl::PointXYZRGB> cloud);

     void initializeViewer(bool iscolor);

 private:

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

   //   pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb;

     sensor_msgs::PointCloud2::Ptr currentCloud;

     pcl::PointCloud<pcl::Normal>::Ptr currentCloudNormals;

     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
 //   pcl::PointCloud<pcl::PointXYZ> currentCloud;




};

#endif // PCPROCESSING_H
