#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <stdio.h>
#include "bubbleprocess.h"
#include "imageprocess.h"
#include "databasemanager.h"
#include "localization.h"
#include "qstring.h"
#include <fstream>
#include <iostream>
#include <time.h>
#include <sys/time.h>


#include <pcl-1.5/pcl/common/common_headers.h>
#include <pcl-1.5/pcl/filters/voxel_grid.h>
#include <pcl-1.5/pcl/common/transforms.h>
//#include <pcl-1.5/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.5/pcl/impl/point_types.hpp>
#include <pcl-1.5/pcl/point_types_conversion.h>
#include <pcl-1.5/pcl/io/pcd_io.h>

//Mustafa include for max_reliable_kinect_range
#include <pcl/filters/passthrough.h>
//#include <pcl/point_types.h>

//Includes for database
#include <QObject>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlError>
#include <QFile>
#include <QDir>
#include <QtSql/QSqlQuery>
#include <QVariant>
#include <QDebug>
#include <QVector>

//Include for initial pose publishing
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <complex>
#include <geometry_msgs/Twist.h>

//Gloabal Variables


ros::Publisher velocityCommandPublisher;
bool is_callback_called=false;
int cloud_name=0;

typedef struct localizationStruct
{
    cv::Mat invariants;
    cv::Mat filters[5];
    bool normalize_invariants;
    bool exclude_depth;
    double bubble_update_period;
    double angular_velocity;

} localizationStruct;


void localizationCallback(const sensor_msgs::PointCloud2ConstPtr& cloud, localizationStruct* callbackStruct){

    // reading the point cloud data from the kinect sensor
    sensor_msgs::PointCloud2 _cloud = *cloud;

    //changing the global variable when the callback function is called
    is_callback_called=true;

    // create variables for invariant calculation
    int satLower = 30;
    int satUpper = 230;
    int valLower = 30;
    int valUpper = 230;
    int focalLengthPixels = 525;
    int noHarmonics=10;
    double maxRangeMeters = 6;

    bool normalize_invariants=callbackStruct->normalize_invariants;
    bool exclude_depth= callbackStruct->exclude_depth;
    double bubble_update_period = callbackStruct->bubble_update_period;
    cv::Mat filters[5]= callbackStruct->filters;
    double angular_velocity=callbackStruct->angular_velocity;

    cv::Mat invariants(1,noHarmonics*noHarmonics,CV_32FC1);

    //Create variables for processing time calculation
    struct timespec t1, t2;
    double elapsed_time;
    clock_gettime(CLOCK_MONOTONIC,  &t1);

    //Create vector for bubbles
    std::vector<bubblePointXYZ> bubble;
    //Create vector for cloud fields
    std::vector<sensor_msgs::PointField> fields = _cloud.fields;

    //Create empty invariant vector for current data

    //Create temporary vector for normalized single feature vectors
    cv::Mat normalizedDummyVector;

    if(fields.at(3).name == "rgba")
    {
        ROS_INFO("Processing Pointcloud - name=rgba");
        pcl::PointCloud<pcl::PointXYZRGBA> normalCloud;
        pcl::fromROSMsg(_cloud,normalCloud);
    }
    else
    {
        // saving the point cloud data
        pcl::PointCloud<pcl::PointXYZRGB> normalCloud;
        pcl::fromROSMsg(_cloud,normalCloud);

        QString fileName = "/home/turtlebot2/Desktop/cloud_";

        fileName.append(QString::number(cloud_name));

        fileName.append(".pcd");
        pcl::io::savePCDFileBinary(fileName.toStdString(),normalCloud);

        cloud_name++;

        //Create BGR image matrix
        cv::Mat bgr_image;

        //Extract BGR data from pointcloud into BGR image matrix
        if (normalCloud.isOrganized()) {
            bgr_image = cv::Mat(normalCloud.height, normalCloud.width, CV_8UC3);

            if (!normalCloud.empty()){
                unsigned int index;
                for (int h=0; h<bgr_image.rows; h++) {
                    index = h*bgr_image.cols;
                    for (int w=0; w<bgr_image.cols; w++) {
                        /*pcl::PointXYZRGB point = normalCloudRGB.at(w, h);
                        Eigen::Vector3i rgb = point.getRGBVector3i();*/
                        bgr_image.at<cv::Vec3b>(h,w)[2] = normalCloud.points.at(index).r;
                        bgr_image.at<cv::Vec3b>(h,w)[1] = normalCloud.points.at(index).g;
                        bgr_image.at<cv::Vec3b>(h,w)[0] = normalCloud.points.at(index).b;
                        index++;
                    }
                }
            }
        }

        std::cout << "image is constructed" << std::endl;

        // for all filters, calculate image invariants, normalize them if normalization is true and append them
        cv::Mat resg;

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

        std::cout << "filters are applied" << std::endl;

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


        std::cout << "hue invariants are constructed" << std::endl;

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
    }

    std::cout << "depth invariants are constructed" << std::endl;

    //save the invariants in to the callback struct to use in the main loop
    invariants.copyTo(callbackStruct->invariants);

    //rotate the robot 45 degree to obtain new Kinect data
    geometry_msgs::Twist velocityCommandMsg;
    velocityCommandMsg.linear.x = 0;
    velocityCommandMsg.angular.z = -angular_velocity/2;
    velocityCommandPublisher.publish(velocityCommandMsg);

    //Send velocity command message for 45 degrees rotation to left
    struct timespec tMotion1, tMotion2;
    clock_gettime(CLOCK_MONOTONIC,  &tMotion1);
    clock_gettime(CLOCK_MONOTONIC,  &tMotion2);
    float elapsedMotionTime = (tMotion2.tv_sec - tMotion1.tv_sec) + (double) (tMotion2.tv_nsec - tMotion1.tv_nsec) * 1e-9;
    while(elapsedMotionTime <= 2*(bubble_update_period-9)/12.4){
        velocityCommandPublisher.publish(velocityCommandMsg);
        clock_gettime(CLOCK_MONOTONIC,  &tMotion2);
        elapsedMotionTime = (tMotion2.tv_sec - tMotion1.tv_sec) + (double) (tMotion2.tv_nsec - tMotion1.tv_nsec) * 1e-9;
    }

    struct timespec t_stop1, t_stop2;
    clock_gettime(CLOCK_MONOTONIC,  &t_stop1);
    clock_gettime(CLOCK_MONOTONIC,  &t_stop2);
    float elapsedStopTime=(t_stop2.tv_sec - t_stop1.tv_sec) + (double) (t_stop2.tv_nsec - t_stop1.tv_nsec) * 1e-9;
    while(elapsedStopTime <= 1){
        velocityCommandMsg.angular.z = 0;
        velocityCommandPublisher.publish(velocityCommandMsg);
        clock_gettime(CLOCK_MONOTONIC,  &t_stop2);
        elapsedStopTime=(t_stop2.tv_sec - t_stop1.tv_sec) + (double) (t_stop2.tv_nsec - t_stop1.tv_nsec) * 1e-9;
    }


    //Calculate the processing time
    clock_gettime(CLOCK_MONOTONIC,  &t2);
    elapsed_time = (t2.tv_sec - t1.tv_sec) + (double) (t2.tv_nsec - t1.tv_nsec) * 1e-9;
    qDebug() << "Processed in " << elapsed_time << " seconds.";
}

int main( int argc, char* argv[] )
{
    struct timespec t_initial, t_final, t_loc_estimation1, t_loc_estimation2;
    clock_gettime(CLOCK_MONOTONIC, &t_initial);

    //Initialize the ROS node for bubble space
    ros::init( argc, argv, "bubble_space_node" );
    ros::NodeHandle n;
    ros::NodeHandle nh("~");

    bool normalize_invariants;
    bool exclude_depth;
    bool omni_directional, online_localization;
    int rotation_count=0;
    cv::Mat invariantMatrix;
    cv::Mat omni_invariants;
    cv::Mat invariants;
    cv::Mat bubble_location_matrix;

    double angular_velocity;
    double bubble_update_period;
    cv::Mat location_estimation;


    DatabaseManager current_place_db_manager;
    DatabaseManager learned_place_db_manager;
    DatabaseManager bubble_database_manager;

    localizationStruct callback_struct;


    //Read the parameters
    std::string bubble_database_path, filters_dir_path, positions_dir_path, learned_place_dir_path, current_place_dir_path;
    int base_point_number, orientation_number, filter_no, noHarmonics;

    nh.param("normalize_invariants", normalize_invariants, true);
    nh.param("bubble_update_period", bubble_update_period, 25.0);
    nh.param("angular_velocity", angular_velocity, 2/(25.0-5)*9);
    nh.param("exclude_depth", exclude_depth, false);
    nh.param("harmonics_number", noHarmonics, 10);
    nh.getParam("bubble_database_path", bubble_database_path);
    nh.getParam("filters_dir_path", filters_dir_path);
    nh.getParam("base_point_number", base_point_number);
    nh.getParam("orientation_number", orientation_number);
    nh.getParam("filter_no", filter_no);
    nh.getParam("positions_dir_path", positions_dir_path);
    nh.getParam("omni_directional",omni_directional);
    nh.getParam("online_localization",online_localization);

    if(!online_localization){

        nh.getParam("learned_place_dir_path",learned_place_dir_path);
        nh.getParam("current_place_dir_path",current_place_dir_path);

        std::string learned_place_db_path = learned_place_dir_path.append("/detected_places.db");
        std::string current_place_db_path = current_place_dir_path.append("/detected_places.db");

        learned_place_db_manager.openDB(learned_place_db_path.c_str());
        current_place_db_manager.openDB(current_place_db_path.c_str());

    }

    if (online_localization){
        //Check, whether the database file path is good
        std::ifstream database_file(bubble_database_path.c_str());
        if (!database_file.good()){
            ROS_FATAL("Database file cannot be found!");
            return 0;
        }


        //Check minimum bubble update period and calculate the angular velocity according to bubble update period
        if(bubble_update_period < 5){
            ROS_WARN("Minimum allowed update period is 5 seconds! Setting the update period to 2 seconds now.");
            bubble_update_period = 5;
            angular_velocity = 0;
        }
        else{
            angular_velocity = 2/(bubble_update_period-5)*9;        //2 radians
        }
        //cv::Mat location_matrix(base_point_number, 2, CV_32F);
        //Open database
        bubble_database_manager.openDB(bubble_database_path.c_str());
        //Read the database and create the invariant matrix
        //invariantMatrix = DatabaseManager::createInvariantMatrix(normalize_invariants, noHarmonics);
        invariantMatrix=bubble_database_manager.createInvariantMatrix(normalize_invariants, noHarmonics, orientation_number, base_point_number, filter_no);
        ROS_INFO("Invariant database matrix is created successfully!");

        //read the locations of the learned base points from the database
        bubble_location_matrix=bubble_database_manager.createLocationMatrix(base_point_number);

        //Calculate pan and tilt angles for bubble space
        bubbleProcess::calculateImagePanAngles(525,640,480);
        bubbleProcess::calculateImageTiltAngles(525,640,480);

        // definition of filters used
        int filter_numbers[5] = {14,16,20,21,43};
        //read the filters
        for(int i=0; i<filter_no; i++){
            callback_struct.filters[i]=localization::selectReadFilter(filter_numbers[i],filters_dir_path);
        }
        callback_struct.bubble_update_period=bubble_update_period;
        callback_struct.exclude_depth= exclude_depth;
        callback_struct.normalize_invariants=normalize_invariants;
        callback_struct.angular_velocity=angular_velocity;

        //Subscriber for /camera/depth_registered/points topic
        ros::Subscriber sub_pclReg = nh.subscribe<sensor_msgs::PointCloud2> ("/camera/depth_registered/points", 1, boost::bind(&localizationCallback, _1, &callback_struct ) );

        //Publisher for cmd_vel topic
        velocityCommandPublisher = n.advertise<geometry_msgs::Twist>("cmd_vel",1);
    }

    //Set the bubble update period
    ros::Rate r(10);

    // turn the robot, get the Kinect data and calculate invariants 8 times
    if (online_localization && !omni_directional){
        while (ros::ok() && rotation_count < (orientation_number+1))
        {
            //Step 1: Analyze Kinect data until finding a localization in maximally three iterations
            //Get Kinect data and apply bubble space algorithm
            ros::spinOnce();
            std::cout << is_callback_called << std::endl;
            if(is_callback_called){
                invariants = callback_struct.invariants;

                // connecting the invariants as the robot turns and gets new Kinect data
                if (rotation_count == 1){
                    omni_invariants = invariants;
                }
                else if (rotation_count != 0){
                    cv::vconcat(omni_invariants, invariants, omni_invariants);
                }

                rotation_count++;
                std::cout << rotation_count << std::endl;
                is_callback_called=false;
            }

            r.sleep();

        }

        // estimate the location of the robot based on the database invariant matrix and the current invariants
        location_estimation=localization::onlineLocationEstimation(invariantMatrix,omni_invariants, bubble_location_matrix, base_point_number, orientation_number);

        std::cout << "Estimated location is:" << location_estimation << std::endl;
        clock_gettime(CLOCK_MONOTONIC, &t_final);
        double elapsed_time = (t_final.tv_sec - t_initial.tv_sec) + (double) (t_final.tv_nsec - t_initial.tv_nsec) * 1e-9;
        qDebug() << "Processed in " << elapsed_time << " seconds.";
    }

    // if the offline localization (by images from datasets) and omnidirectional images are used:
    if (!online_localization && omni_directional){
        while(ros::ok())
        {
            ros::spinOnce();

        }
    }



    ROS_INFO("ROS EXIT");
    return 0;
}
