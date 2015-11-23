#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H
#include <opencv2/opencv.hpp>
#include <QString>

using namespace cv;

class ImageProcess
{
public:

    ImageProcess();

    static Mat generateChannelImage(Mat rgbimage, int channelNo, int satLower, int satUpper, int valLower, int valUpper);

    static Mat generateHueImage(int satLower, int satUpper, int valLower, int valUpper);

    static Mat mstApplyFilter(Mat singleChannelImage, Mat currentFilter);

    static Mat mstReadFilter(QString fileName, int filterSize, bool transpose, bool save, bool show);

};

#endif // IMAGEPROCESS_H
