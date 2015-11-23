#include "pcprocessing.h"
#include "bubbleprocess.h"
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <pcl-1.5/pcl/filters/voxel_grid.h>
#include <pcl-1.5/pcl/common/transforms.h>
#include <pcl-1.5/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.5/pcl/impl/point_types.hpp>
#include <pcl-1.5/pcl/point_types_conversion.h>
#include <Eigen/Geometry>
#include <Eigen/Eigen>
#include <Eigen/Core>

//sensor_msgs::PointCloud2::Ptr currentCloud = sensor_msgs::PointCloud2::Ptr(new sensor_msgs::PointCloud2 ());

double round(double r) {
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}



static QString dataSetPath;
QStringList dataSetItems;
//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

PCprocessing::PCprocessing()
{
    currentCloud = sensor_msgs::PointCloud2::Ptr(new sensor_msgs::PointCloud2 ());

    currentCloudNormals = boost::shared_ptr< pcl::PointCloud<pcl::Normal> >(new pcl::PointCloud<pcl::Normal>());

    // cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud = boost::shared_ptr< pcl::PointCloud<pcl::PointXYZRGB> >(new pcl::PointCloud<pcl::PointXYZRGB>());

    // this->initializeViewer();
}

void PCprocessing::setDataSetPath(QString dir){


    dataSetPath = dir;

}
void PCprocessing::initializeViewer(bool iscolor)
{

    // If we don't have a kinect cloud inside the viewer, add it
    if(!viewer->updatePointCloud(cloud,"kinect cloud"))
    {

        viewer->removeAllPointClouds();
        //  qDebug()<<viewer;

        cloud->height = 1;
        cloud->width =1;
        cloud->is_dense = false;
        cloud->points.resize (cloud->width * cloud->height);
        pcl::PointXYZRGB pt ;
        pt.x = 0;
        pt.y = 0;
        pt.z = 0.1;

        pt.r = 0;
        pt.g = 1;
        pt.b = 0;

        cloud->points.push_back(pt);

        qDebug()<<"I am here....";

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
{

        //viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "kinect cloud");

        viewer->setBackgroundColor(0, 0, 0);

        qDebug()<<"I am here 2....";

        viewer->addPointCloud<pcl::PointXYZRGB>(cloud,rgb,"kinect cloud");

        //  qDebug()<<"I am here 3....";


        viewer->setPointCloudRenderingProperties
                (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3,"kinect cloud");

        //   qDebug()<<"I am here 4....";

        viewer->addCoordinateSystem(1.0);

        //  qDebug()<<"I am here 5....";


        viewer->initCameraParameters();

        //      qDebug()<<"I am here 6....";

        // viewer->spinOnce();
    }
    else viewer->addCoordinateSystem(1.0);
}
QString PCprocessing::getDataSetPath(){

    return dataSetPath;

}
void PCprocessing::setDataSetItems(QStringList items){

    if(dataSetItems.size() > 0){

        dataSetItems.clear();
    }


    dataSetItems<<items;

    qDebug()<<"datasetitems: "<<dataSetItems;





}
int PCprocessing::getNumofItems()
{

    return dataSetItems.size();


}
void PCprocessing::setViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer> viwer){
    // if(viewer.use_count()>0)
    // viewer.reset();


    viewer = viwer;

    //  initializeViewer();

}
bool PCprocessing::loadItem(QString filePath, sensor_msgs::PointCloud2::Ptr cloud)
{
    if( pcl::io::loadPCDFile(filePath.toStdString(),*cloud) < 0) return false;

    qDebug()<<"Loaded item path: "<<filePath;

    std::vector<sensor_msgs::PointField> fields = cloud->fields;

    if(fields.at(3).name == "rgba")
    {

        pcl::PointCloud<pcl::PointXYZRGBA> tempCloud;

        pcl::fromROSMsg(*cloud,tempCloud);


        pcl::PointXYZRGBA pt = tempCloud.points.at(0);

        if(pt.r == pt.r )
        {

            if(pt.r != 0 || pt.b != 0 || pt.g != 0)
            {

                qDebug()<<"r is"<<pt.r;

                viewer->removeCoordinateSystem();

                this->initializeViewer(true);

                viewer->updatePointCloud(tempCloud.makeShared(),"kinect cloud");

            }
            else
            {

                viewer->removeAllPointClouds();

                viewer->addPointCloud<pcl::PointXYZRGBA>(tempCloud.makeShared());

            }
        }
        else
        {

            viewer->removeAllPointClouds();

            // viewer->ad

            viewer->addPointCloud<pcl::PointXYZRGBA>(tempCloud.makeShared());


        }
    }
    //    pcl::toROSMsg(tempCloud,*cloud);

    //  viewer->removeAllPointClouds();

    // viewer->removeCoordinateSystem();

    //  viewer->addPointCloud<pcl::PointXYZRGB>(tempCloud.makeShared());

    //  viewer->addCoordinateSystem (1.0);

    viewer->resetCamera();


    return true;


}
bool PCprocessing::loadItem(int itemNumber, QString fileName, sensor_msgs::PointCloud2::Ptr cloud)
{
    // qDebug()<<"datasetitems:: "<<dataSetItems;

    if(dataSetPath==NULL) return false;

    QString itemPath = dataSetPath;

    itemPath.append(fileName);

    if(itemNumber >=0)
    {

        QString ss;

        ss.setNum(itemNumber);

        itemPath.append(ss);
    }

    itemPath.append(".pcd");

    // itemPath.append(dataSetItems.at(itemNumber));

    if( pcl::io::loadPCDFile(itemPath.toStdString(),*cloud) < 0) return false;

    qDebug()<<"Loaded item path: "<<itemPath;

    std::vector<sensor_msgs::PointField> fields = cloud->fields;

    if(fields.at(3).name == "rgba")
    {

        pcl::PointCloud<pcl::PointXYZRGBA> tempCloud;

        pcl::fromROSMsg(*cloud,tempCloud);


        pcl::PointXYZRGBA pt = tempCloud.points.at(0);

        if(pt.r == pt.r )
        {

            if(pt.r != 0 || pt.b != 0 || pt.g != 0)
            {


                qDebug()<<"r is"<<pt.r;

                viewer->removeCoordinateSystem();

                this->initializeViewer(true);

                if(!viewer->updatePointCloud(tempCloud.makeShared(),"kinect cloud"))
                {
                    qDebug()<<"We don't have a kinect cloud";

                }

            }
            else
            {

                viewer->removeAllPointClouds();

                viewer->addPointCloud<pcl::PointXYZRGBA>(tempCloud.makeShared());

            }
        }
        else
        {

            viewer->removeAllPointClouds();

            // viewer->ad

            viewer->addPointCloud<pcl::PointXYZRGBA>(tempCloud.makeShared());


        }
    }
    else if(fields.at(3).name == "rgb")
    {

        pcl::PointCloud<pcl::PointXYZRGB> tempCloud;

        pcl::fromROSMsg(*cloud,tempCloud);


        pcl::PointXYZRGB pt = tempCloud.points.at(0);

        if(pt.r == pt.r )
        {

            if(pt.r != 0 || pt.b != 0 || pt.g != 0)
            {


                qDebug()<<"r is"<<pt.r;

                viewer->removeCoordinateSystem();

                this->initializeViewer(true);

                if(!viewer->updatePointCloud(tempCloud.makeShared(),"kinect cloud"))
                {
                    qDebug()<<"We don't have a kinect cloud";

                }

            }
            else
            {

                viewer->removeAllPointClouds();

                viewer->addPointCloud<pcl::PointXYZRGB>(tempCloud.makeShared());

            }
        }
        else
        {

            viewer->removeAllPointClouds();

            // viewer->ad

            viewer->addPointCloud<pcl::PointXYZRGB>(tempCloud.makeShared());


        }


    }
    //    pcl::toROSMsg(tempCloud,*cloud);

    //  viewer->removeAllPointClouds();

    // viewer->removeCoordinateSystem();

    //  viewer->addPointCloud<pcl::PointXYZRGB>(tempCloud.makeShared());

    //  viewer->addCoordinateSystem (1.0);

    viewer->resetCamera();


    return true;

}
void PCprocessing::showPointCloud(sensor_msgs::PointCloud2::Ptr cloud)
{

    pcl::PointCloud<pcl::PointXYZRGB> tempCloud;

    pcl::fromROSMsg(*cloud,tempCloud);

    viewer->removeAllPointClouds();

    viewer->addPointCloud<pcl::PointXYZRGB>(tempCloud.makeShared());


    viewer->resetCamera();




}
void PCprocessing::showPointCloud(pcl::PointCloud<pcl::PointXYZRGB> cloud)
{

    //  viewer->removeAllPointClouds();

    viewer->updatePointCloud(cloud.makeShared(),"kinect cloud");

    // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud.makeShared());

    //  viewer->addPointCloud<pcl::PointXYZRGB>(cloud.makeShared(),rgb,"sample cloud");

    //  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");

    //  viewer->addCoordinateSystem (1.0);


    // viewer->resetCamera();


}

void PCprocessing::rotatePointCloud(sensor_msgs::PointCloud2::Ptr input, int rotX, int rotY, int rotZ)
{
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


    std::vector<sensor_msgs::PointField> fields = input->fields;

    if(fields.at(3).name == "rgba")
    {
        pcl::PointCloud<pcl::PointXYZRGBA> tempCloud;

        pcl::fromROSMsg(*input,tempCloud);

        pcl::transformPointCloud(tempCloud,tempCloud,xx*yy*zz);

        pcl::toROSMsg(tempCloud,*currentCloud);

        pcl::PointXYZRGBA pt = tempCloud.points.at(0);

        if(pt.r == pt.r )
        {

            if(pt.r != 0 || pt.b != 0 || pt.g != 0)
            {

                qDebug()<<"r is"<<pt.r;

                viewer->removeCoordinateSystem();

                this->initializeViewer(true);

                viewer->updatePointCloud(tempCloud.makeShared(),"kinect cloud");

            }
            else
            {

                viewer->removeAllPointClouds();

                viewer->addPointCloud<pcl::PointXYZRGBA>(tempCloud.makeShared());

            }
        }


        // viewer->removeAllPointClouds();

        // viewer->addPointCloud<pcl::PointXYZRGBA>(tempCloud.makeShared());
    }
    else
    {

        pcl::PointCloud<pcl::PointXYZRGB> tempCloud;

        pcl::fromROSMsg(*input,tempCloud);

        pcl::transformPointCloud(tempCloud,tempCloud,xx*yy*zz);

        pcl::toROSMsg(tempCloud,*currentCloud);

        pcl::PointXYZRGB pt = tempCloud.points.at(0);

        if(pt.r == pt.r )
        {

            if(pt.r != 0 || pt.b != 0 || pt.g != 0)
            {

                qDebug()<<"r is"<<pt.r;

                viewer->removeCoordinateSystem();

                this->initializeViewer(true);

                viewer->updatePointCloud(tempCloud.makeShared(),"kinect cloud");

            }
            else
            {

                viewer->removeAllPointClouds();

                viewer->addPointCloud<pcl::PointXYZRGB>(tempCloud.makeShared());

            }
        }



    }

    viewer->resetCamera();

}
void PCprocessing::scalePointCloud(sensor_msgs::PointCloud2::Ptr input, double scale)
{
     std::vector<sensor_msgs::PointField> fields = input->fields;

    if(fields.at(3).name == "rgba")
    {

        pcl::PointCloud<pcl::PointXYZRGBA> tempCloud;

        pcl::fromROSMsg(*input,tempCloud);


        for(long k = 0; k < tempCloud.width; k++){

            pcl::PointXYZRGBA pt =  tempCloud.points[k];

            pt.x = pt.x/scale;
            pt.y = pt.y/scale;
            pt.z = pt.z/scale;

            tempCloud.points[k] = pt;


        }

        pcl::toROSMsg(tempCloud,*input);

        pcl::PointXYZRGBA pt = tempCloud.points.at(0);

        if(pt.r == pt.r )
        {

            if(pt.r != 0 || pt.b != 0 || pt.g != 0)
            {

                qDebug()<<"r is"<<pt.r;

                viewer->removeCoordinateSystem();

                this->initializeViewer(true);

                viewer->updatePointCloud(tempCloud.makeShared(),"kinect cloud");

            }
            else
            {

                viewer->removeAllPointClouds();

                viewer->addPointCloud<pcl::PointXYZRGBA>(tempCloud.makeShared());

            }
        }

    }
    else
    {
        pcl::PointCloud<pcl::PointXYZRGB> tempCloud;

        pcl::fromROSMsg(*input,tempCloud);


        for(long k = 0; k < tempCloud.width; k++){

            pcl::PointXYZRGB pt =  tempCloud.points[k];

            pt.x = pt.x/scale;
            pt.y = pt.y/scale;
            pt.z = pt.z/scale;

            tempCloud.points[k] = pt;


        }

        pcl::toROSMsg(tempCloud,*input);

        pcl::PointXYZRGB pt = tempCloud.points.at(0);

        if(pt.r == pt.r )
        {

            if(pt.r != 0 || pt.b != 0 || pt.g != 0)
            {

                qDebug()<<"r is"<<pt.r;

                viewer->removeCoordinateSystem();

                this->initializeViewer(true);

                viewer->updatePointCloud(tempCloud.makeShared(),"kinect cloud");

            }
            else
            {

                viewer->removeAllPointClouds();

                viewer->addPointCloud<pcl::PointXYZRGB>(tempCloud.makeShared());

            }
        }

    }

    //  viewer->removeAllPointClouds();
    //   viewer->addPointCloud<pcl::PointXYZRGB>(tempCloud.makeShared());


    viewer->resetCamera();



}

sensor_msgs::PointCloud2::Ptr PCprocessing::getCurrentCloud()
{

    return this->currentCloud;



}
pcl::PointCloud<pcl::Normal>::Ptr PCprocessing::getCurrentCloudNormals(){


    return this->currentCloudNormals;

}

void PCprocessing::applyVoxelGridFilter(sensor_msgs::PointCloud2::Ptr input)
{

    qDebug()<<"PointCloud before filtering: "<< input->width * input->height<< " data points (" << QString::fromStdString( pcl::getFieldsList (*input) )<< ").";;

    sensor_msgs::PointCloud2::Ptr filtered_cloud(new sensor_msgs::PointCloud2 ());

    // Create the filtering object
    pcl::VoxelGrid<sensor_msgs::PointCloud2> sor;
    sor.setInputCloud(input);
    sor.setLeafSize (0.01f, 0.01f, 0.01f);
    sor.filter (*filtered_cloud);

    qDebug()<< "PointCloud after filtering: " << filtered_cloud->width * filtered_cloud->height;//<< " data points (" << QString::fromStdString( pcl::getFieldsList (*filtered_cloud) )<< ").";

    currentCloud.reset();

    currentCloud = filtered_cloud;

    filtered_cloud.reset();

    //qDebug()<<"Current cloud use count: "<<currentCloud.use_count();

    pcl::PointCloud<pcl::PointXYZRGBA> tempCloud;

    pcl::fromROSMsg(*currentCloud,tempCloud);

    pcl::PointXYZRGBA pt = tempCloud.points.at(0);

    if(pt.r == pt.r )
    {

        if(pt.r != 0 || pt.b != 0 || pt.g != 0)
        {

            qDebug()<<"r is"<<pt.r;

            viewer->removeCoordinateSystem();

            this->initializeViewer(true);

            viewer->updatePointCloud(tempCloud.makeShared(),"kinect cloud");

        }
        else
        {

            viewer->removeAllPointClouds();

            viewer->addPointCloud<pcl::PointXYZRGBA>(tempCloud.makeShared());

        }
    }

    /* viewer->removeAllPointClouds();
    viewer->addPointCloud<pcl::PointXYZRGBA>(tempCloud.makeShared());*/


    viewer->resetCamera();


}
void PCprocessing::calculateNormals(sensor_msgs::PointCloud2::Ptr input)
{


    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZRGBA>());

    pcl::fromROSMsg(*input,*tempCloud);

    ne.setInputCloud (tempCloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBA> ());
    ne.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    if(currentCloudNormals->size() > 0) currentCloudNormals.reset(new pcl::PointCloud<pcl::Normal>());



    //currentCloudNormals = cloud_normals;

    // Use all neighbors in a sphere of radius 10cm
    ne.setRadiusSearch (0.1);

    // Compute the features
    ne.compute (*currentCloudNormals);

    // pcl::Normal pt = cloud_normals->points.at(0);

    //  qDebug()<<"normal x :"<<pt.normal_x;

    /* for(unsigned long i = 0; i < currentCloudNormals->points.size(); i++){

        pcl::Normal pt = currentCloudNormals->points.at(i);

    // Just to see the normals towards me
        if(fabs(pt.normal_x) < 0.9 )
        {
            pt.normal_x = 0;
            pt.normal_y = 0;
            pt.normal_z = 0;

            currentCloudNormals->points.at(i) = pt;


        }

      //  i = i-1;


    }*/

    // currentCloudNormals->height = currentCloudNormals->size();
    // currentCloudNormals->width = 1;

    viewer->addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal> (tempCloud, currentCloudNormals, 50, 0.05, "normals");

    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "normals");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "normals");

    viewer->resetCamera();


}
bool PCprocessing::saveNormalAngleHistogram(pcl::PointCloud<pcl::Normal>::Ptr normals,int itemNumber)
{
    if(dataSetPath==NULL) return false;

    // if(dataSetItems.size() == 0) return false;

    QString itemPath = dataSetPath;

    //  itemPath.append("n");

    itemPath.append("normalAngleHistBubble_");

    QString ss;

    ss.setNum(itemNumber);

    itemPath.append(ss);

    itemPath.append(".m");

    // itemPath.append(dataSetItems.at(itemNumber));

    qDebug()<<"Save normal histogram bubble: "<<itemPath;


    QFile file(itemPath);

    QFile file2("normals.txt");

    if(!file.open(QFile::WriteOnly|QFile::Text) || !file2.open(QFile::WriteOnly|QFile::Text)){


        return false;
    }

    if(normals->size() <= 0) return false;

    QVector< QVector<int> > angles(360, QVector<int>(360));

    std::vector<bubblePoint> result;

    QTextStream stream(&file);

    QTextStream stream2(&file2);

    for(unsigned long i = 0; i < normals->size(); i+=5)
    {

        pcl::Normal pt = normals->points.at(i);

        double pan = atan2(pt.normal_y,pt.normal_x)*180/3.14159;

        pan = round(pan);

        double xy = sqrt(pt.normal_x*pt.normal_x + pt.normal_y*pt.normal_y);

        double tilt = atan2(pt.normal_z,xy)*180/3.14159;

        tilt = round(tilt);

        if(pan < 0)pan += 360;
        else if(pan > 359) pan -=360;

        if(tilt < 0) tilt += 360;
        else if(tilt > 359) tilt -=360;

        //  qDebug()<<"pan: "<<pan<<"tilt: "<<tilt;

        // This is for checking NAN situationss
        if(pan == pan && tilt == tilt)
            angles[(int)pan][(int)tilt] += 1;

        stream2<<pt.normal_x<<" "<<pt.normal_y<<" "<<pt.normal_z<<" "<<pan<<" "<<" "<<tilt<<"\n";



    }

    for(int i = 0; i < 360; i++){
        for(int j = 0; j < 360; j++){

            double val = (double)angles[i][j]/50;

            if(val > 1) val = 0.99;
            //  if(angles[i][j]!=0)

            stream<<i<<" "<<j<<" "<<val<<"\n";

        }

    }


    file.close();

    file2.close();

    return true;
}
bool PCprocessing::saveNormalPanTiltHistograms(pcl::PointCloud<pcl::Normal>::Ptr normals)
{

    return false;

}
bool PCprocessing::savePointCloud(int itemNumber, QString fileName){


    if(dataSetPath==NULL) return false;

    // if(dataSetItems.size() == 0) return false;

    QString itemPath = dataSetPath;

    //itemPath.append("n");

    itemPath.append(fileName);

    QString ss;

    ss.setNum(itemNumber);

    itemPath.append(ss);

    itemPath.append(".pcd");

    // itemPath.append(dataSetItems.at(itemNumber));

    qDebug()<<itemPath;

    std::vector<sensor_msgs::PointField> fields = currentCloud->fields;

   if(fields.at(3).name == "rgba")
   {

    pcl::PointCloud<pcl::PointXYZRGBA> tempCloud;

    pcl::fromROSMsg(*currentCloud,tempCloud);

    pcl::io::savePCDFileBinary(itemPath.toStdString(),tempCloud);

    return true;
   }
   else
   {
       pcl::PointCloud<pcl::PointXYZRGB> tempCloud;

       pcl::fromROSMsg(*currentCloud,tempCloud);

       pcl::io::savePCDFileBinary(itemPath.toStdString(),tempCloud);

   }

    return false;
}
