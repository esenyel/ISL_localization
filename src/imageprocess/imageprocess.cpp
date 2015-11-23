#include "imageprocess.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QString>
#include <QStringList>

Mat orgImg;

vector<Mat> channels;

Mat filter;

Mat filterOrg;

static Mat filters[5];

static QString datasetPath;

ImageProcess::ImageProcess()
{

}

Mat ImageProcess::mstReadFilter(QString fileName, int filterSize, bool transpose, bool save, bool show)
{
        filterOrg = Mat(filterSize,filterSize,CV_32FC1);
        /*QString dirr = fileName;
        qDebug()<<"Dir is :"<<dirr;
        QFile file(dirr);*/
        QFile file(fileName);
        file.open(QFile::ReadOnly);
        QTextStream stream(&file);
        //if(!file.isOpen()) return ;
        QString line = stream.readLine();
        double count = 0;
        double count2 = 0;
        while(line != NULL)
        {
            filterOrg.at<float>(count,count2) = line.toFloat();
            count++;
            if(count == filterSize){
                count2++;
                count = 0;
            }
            line = stream.readLine();
        }
        file.close();

        if(transpose)
            cv::transpose(filterOrg,filterOrg);

        cv::convertScaleAbs(filterOrg,filter,128,128);
        cv::Mat resizedFilter;
        cv::resize(filter,resizedFilter,resizedFilter.size(),5,5);

        if(show)
        {
            namedWindow("filter");
            imshow("filter",resizedFilter);
            waitKey();
            destroyWindow("filter");
        }
        if(save)
        {
            imwrite("filter.jpg",resizedFilter);
            qDebug()<<"Filter image saved";
        }
    return filterOrg;
}



Mat ImageProcess::mstApplyFilter(Mat singleChannelImage, Mat currentFilter)
{
    Mat result = Mat::zeros(singleChannelImage.rows,singleChannelImage.cols,CV_8UC1);
    cv::GaussianBlur(singleChannelImage,singleChannelImage,cv::Size(5,5),5,5);
    cv::filter2D(singleChannelImage,result,result.depth(),currentFilter);
    return result;
}


Mat ImageProcess::generateChannelImage(Mat rgbimage, int channelNo, int satLower, int satUpper, int valLower, int valUpper)
{

    Mat hsvimage;
    cv::cvtColor(rgbimage,hsvimage,CV_BGR2HSV);

    Mat result;

    result = Mat::zeros(rgbimage.rows,rgbimage.cols,CV_8UC1);

    cv::split(hsvimage,channels);
    for(int i = 0; i < rgbimage.rows; i++)
    {

        for(int j = 0; j < rgbimage.cols; j++)
        {

            uchar satval = channels[1].at<uchar>(i,j);

            uchar valval = channels[2].at<uchar>(i,j);

            if(valval > valLower && valval < valUpper)
            {
                if(satval > satLower && satval < satUpper)
                {
                    //   if(hueval < 15 ) hueval = 180;

                    result.at<uchar>(i,j) = channels[channelNo].at<uchar>(i,j);
                }
            }
        }
    }
    return result;
}

