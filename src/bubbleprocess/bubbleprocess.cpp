#include "bubbleprocess.h"
#include <QStringList>
#include <QTextStream>
#include <math.h>
#include <QDebug>

using std::vector;

QString bubblesRootDirectory;
QString threeDFilesRootDirectory;
QString writeBubblesRootDirectory;


QStringList bubblesFolderList;
QStringList bubblesfileList;
QStringList threeDfileList;

vector< vector<bubblePoint> > staticBubbles;

vector <positionData> robotPoses;

static vector<vector<int> > imagePanAngles;

static vector<vector<int> > imageTiltAngles;


bubbleProcess::bubbleProcess()
{
}

vector<vector<int> > bubbleProcess::calculateImagePanAngles(int focalLengthPixels, int imageWidth, int imageHeight)
{
    vector<vector<int> > result(imageHeight, std::vector<int>(imageWidth));


    for(int i = 0; i < imageHeight; i++){

        for(int j = 0; j < imageWidth; j++ )
        {

            int deltax = imageWidth/2 - j;


            float pan = atan2((double)deltax,(double)focalLengthPixels);


            int panInt = (pan*180)/M_PI;


            if(panInt < 0)panInt += 360;
            else if(panInt > 359) panInt -=360;


            result[i][j] = panInt;

        }

    }

    if(imagePanAngles.size() > 0){

        for(int i = 0; i < imagePanAngles.size(); i++){
            imagePanAngles[i].clear();
        }

        imagePanAngles.clear();

    }

    imagePanAngles.resize(imageHeight,std::vector<int> (imageWidth));

    imagePanAngles = result;

    return result;

}
vector<vector<int> > bubbleProcess::calculateImageTiltAngles(int focalLengthPixels, int imageWidth, int imageHeight)
{
    vector<vector<int> > result(imageHeight, std::vector<int>(imageWidth));


    for(int i = 0; i < imageHeight; i++){

        for(int j = 0; j < imageWidth; j++ )
        {

            int deltay = imageHeight/2 - i;

            float tilt = atan2((double)deltay,(double)focalLengthPixels);


            int tiltInt = (tilt*180)/M_PI;

            if(tiltInt < 0)tiltInt += 360;
            else if(tiltInt > 359) tiltInt -=360;

            result[i][j] = tiltInt;

        }

    }

    if(imageTiltAngles.size() > 0){

        for(int i = 0; i < imageTiltAngles.size(); i++){
            imageTiltAngles[i].clear();
        }

        imageTiltAngles.clear();

    }

    imageTiltAngles.resize(imageHeight,std::vector<int>(imageWidth));

    imageTiltAngles = result;



    return result;

}
vector<bubblePoint> bubbleProcess::convertGrayImage2Bub(cv::Mat grayImage, int focalLengthPixels, int maxval)
{

    vector<bubblePoint> result;

    int centerx = grayImage.cols/2;

    int centery = grayImage.rows/2;

    for(int i = 0; i < grayImage.rows; i++)
    {
        for(int j = 0; j < grayImage.cols; j++)
        {

            float val = (float)grayImage.at<uchar>(i,j)/(float)maxval;

            if(val > 0){

                bubblePoint pt;

                pt.panAng = imagePanAngles[i][j];

                pt.tiltAng = imageTiltAngles[i][j];

                pt.val = val;

                result.push_back(pt);
            }
        }

    }

    return result;

}

vector<bubblePoint> bubbleProcess::convertBubXYZ2BubSpherical(std::vector<bubblePointXYZ> bubbleXYZ, double maxRange){

    // create the result vector
    vector<bubblePoint> result;

    // for all the points in XYZ bubble
    for(int i = 0; i < bubbleXYZ.size(); i++){

        bubblePoint bp;

        // Calculate panAngle
        bp.panAng = (int)round(((atan2(bubbleXYZ[i].y, bubbleXYZ[i].x))*(double)180/3.14159));

        if(bp.panAng < 0)bp.panAng += 360;
        else if(bp.panAng > 359) bp.panAng -=360;
        // Calculate the magnitude of XY projection of the laser ray
        double rXY = sqrt(bubbleXYZ[i].y*bubbleXYZ[i].y + bubbleXYZ[i].x*bubbleXYZ[i].x);

        // Calculate the tilt angle
        bp.tiltAng = (int)round(((atan2(bubbleXYZ[i].z, rXY))*(double)180/3.14159));

        if(bp.tiltAng < 0)bp.tiltAng += 360;
        else if(bp.tiltAng > 359) bp.tiltAng -=360;

        // Calculate the r value and normalize it
        bp.val = (float)sqrt(bubbleXYZ[i].y*bubbleXYZ[i].y + bubbleXYZ[i].x*bubbleXYZ[i].x + bubbleXYZ[i].z*bubbleXYZ[i].z)/maxRange;

        result.push_back(bp);

    }

    // return resulting bubble
    return result;
}


// Converts a bubble in XYZ space to a bubble in Spherical coordinates
vector<bubblePoint> bubbleProcess::convertBubXYZ2BubSpherical(std::vector<point> bubbleXYZ, double maxRange){
	
	// create the result vector
	vector<bubblePoint> result;

	// for all the points in XYZ bubble
	for(int i = 0; i < bubbleXYZ.size(); i++){
		
		bubblePoint bp;
		
		// Calculate panAngle
		bp.panAng = (int)round(((atan2(bubbleXYZ[i].y, bubbleXYZ[i].x))*(double)180/3.14159));

        if(bp.panAng < 0)bp.panAng += 360;
        else if(bp.panAng > 359) bp.panAng -=360;
		// Calculate the magnitude of XY projection of the laser ray
		double rXY = sqrt(bubbleXYZ[i].y*bubbleXYZ[i].y + bubbleXYZ[i].x*bubbleXYZ[i].x);

		// Calculate the tilt angle
		bp.tiltAng = (int)round(((atan2(bubbleXYZ[i].z, rXY))*(double)180/3.14159));

        if(bp.tiltAng < 0)bp.tiltAng += 360;
        else if(bp.tiltAng > 359) bp.tiltAng -=360;

		// Calculate the r value and normalize it 
        bp.val = (float)sqrt(bubbleXYZ[i].y*bubbleXYZ[i].y + bubbleXYZ[i].x*bubbleXYZ[i].x + bubbleXYZ[i].z*bubbleXYZ[i].z)/maxRange;

        if(bp.val > 0 && bp.val <1.0)
		result.push_back(bp);
	
	}
	
	// return resulting bubble
	return result;

}
vector<bubblePoint> bubbleProcess::convertBubXYZ2BubSpherical(std::vector<point> bubbleXYZ, int heading, double maxRange){
	
	vector<bubblePoint> result;

	for(int i = 0; i < bubbleXYZ.size(); i++){
		
		bubblePoint bp;
		
		bp.panAng = (int)round(((atan2(bubbleXYZ[i].y, bubbleXYZ[i].x))*(double)180/3.14159)) - heading;

		double rXY = sqrt(bubbleXYZ[i].y*bubbleXYZ[i].y + bubbleXYZ[i].x*bubbleXYZ[i].x);

		bp.tiltAng = (int)round(((atan2(bubbleXYZ[i].z, rXY))*(double)180/3.14159));

        bp.val = (float)sqrt(bubbleXYZ[i].y*bubbleXYZ[i].y + bubbleXYZ[i].x*bubbleXYZ[i].x + bubbleXYZ[i].z*bubbleXYZ[i].z)/maxRange;

		result.push_back(bp);
	
	}

	return result;

}

vector<bubblePoint> bubbleProcess::reduceBubble(std::vector<bubblePoint> bubble){
	
	vector<bubblePoint> result;

    double vals[360][360];

    double counts[360][360];

    for(int i = 0; i < 360; i++){

        for(int j = 0; j< 360; j++){

            vals[i][j] = 0;
            counts[i][j] = 0;

        }
    }

	for(long i = 0; i < bubble.size(); i++){

		bubblePoint pt;

		int simCount = 1;

		if(bubble[i].val < 1){	
            pt = bubble[i];

            vals[pt.panAng][pt.tiltAng] += pt.val;
            counts[pt.panAng][pt.tiltAng] += 1;

        }

	}

    for(int i = 0; i < 360; i++){

        for(int j = 0; j< 360; j++){

            if(vals[i][j] != 0)  {
                bubblePoint pt;

                pt.panAng = i;

                 pt.tiltAng = j;

                if(counts[i][j] > 1){

                 pt.val = vals[i][j]/counts[i][j];
                }

                result.push_back(pt);
            }
        }
    }
	
	return result;

}


vector <bubblePointXYZ> bubbleProcess::convertBubSph2BubXYZ(vector<bubblePoint> bubble, double maxRange)
{

    vector <bubblePointXYZ> result;

    for(unsigned int i = 0; i < bubble.size(); i++){

        bubblePointXYZ pt;

        int pan =   bubble[i].panAng;

        int tilt =  bubble[i].tiltAng;

        float val = bubble[i].val;

        pt.z = val*maxRange*sin((float)tilt*3.14159/180);

        float xy =  val*maxRange*cos((float)tilt*3.14159/180);

        pt.x = xy*cos((float)pan*3.14159/180);

        pt.y = xy*sin((float)pan*3.14159/180);

        result.push_back(pt);

    }

    return result;

}

