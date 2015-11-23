#ifndef BUBBLEPROCESS_H
#define BUBBLEPROCESS_H
//#include "globals.h"
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <QFile>
#include <QRgb>
#include <opencv2/opencv.hpp>

class point{
public:
    long r;
    double theta; // laser beam angle in radians
    double phi; // tilt angle (pantilt System)
    double rho; //pan angle (pantilt System)
    double x;
    double y;
    double z;

    QRgb color;
    double red;
    double green;
    double blue;

    int px; // x pixel value of the point
    int py; // y pixel value of the point
};

struct bubblePoint{

    int panAng;
    int tiltAng;
    double val;


};
struct bubbleStatistics
{
    // mean of the bubble Surface
    double mean;

    // variance of the bubble Surface
    double variance;

    // maximum value of the bubbble point
    double maxDist;

};
struct bubblePointXYZ{

    double x;
    double y;
    double z;


};

// A structure for holding the odometry data
struct positionData{

    double x;
    double y;
    double z;

    double headingD;


};
struct DFCoefficients
{
    std::vector< std::vector<float> > a;
    std::vector< std::vector<float> > b;
    std::vector< std::vector<float> > c;
    std::vector< std::vector<float> > d;

};

using std::vector;

class bubbleProcess
{
public:
    
    bubbleProcess();

    static void calculateDFCoefficients(std::vector <bubblePoint> bubble, QString path, QString fileName, int itemNo,int harmonic1, int harmonic2);

    static DFCoefficients calculateDFCoefficients(std::vector <bubblePoint> bubble, int harmonic1, int harmonic2);

    static std::vector< std::vector< float > > calculateInvariants(std::vector <bubblePoint> bubble, QString path, QString fileName, int itemNo, int harmonic1, int harmonic2);

    static std::vector< std::vector< float > > calculateInvariants(std::vector <bubblePoint> bubble, DFCoefficients coeff, int harmonic1, int harmonic2);

    static cv::Mat mstCalculateInvariants(std::vector <bubblePoint> bubble, DFCoefficients coeff, int harmonic1, int harmonic2);

	// Reduces the number of points in a bubble by combining points falling in the same patch
	static vector<bubblePoint> reduceBubble(vector<bubblePoint>bubble); 
	
	// Converts bubble in XYZ coordinates to a bubble in spherical coordinates
    static vector<bubblePoint> convertBubXYZ2BubSpherical(vector<point> bubbleXYZ, double maxRange);

    // Converts bubble in XYZ coordinates to a bubble in spherical coordinates
    static vector<bubblePoint> convertBubXYZ2BubSpherical(vector<bubblePointXYZ> bubbleXYZ, double maxRange);

	// Converts bubble in XYZ coordinates to a bubble in spherical coordinates including heading information
    static vector<bubblePoint> convertBubXYZ2BubSpherical(vector<point> bubbleXYZ, int heading, double maxRange);

    static vector<bubblePointXYZ> convertBubSph2BubXYZ(vector <bubblePoint> bubble, double maxRange);

    static vector<bubblePoint> convertGrayImage2Bub(cv::Mat grayImage, int focalLengthPixels, int maxval);

    static vector<vector <int> > calculateImagePanAngles(int focalLengthPixels,int imageWidth,int imageHeight);

    static vector<vector <int> > calculateImageTiltAngles(int focalLengthPixels,int imageWidth,int imageHeight);

private:
	
    vector< vector<bubblePointXYZ> > bubblesXYZ;

    vector< vector<bubblePoint> > bubbles;


};

#endif // BUBBLEPROCESS_H
