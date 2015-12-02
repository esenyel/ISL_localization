#include "databasemanager.h"
#include <QtSql/QSqlQuery>
#include <QVariant>
#include <QDebug>
#include <QVector>
#include <QMatrix>

#include <stdio.h>
#include <fstream>
#include <complex>

//static QSqlDatabase bubbledb;
//static QSqlDatabase invariantdb;

static QSqlDatabase db;
static QVector<int> placeLabels;

DatabaseManager::DatabaseManager(QObject *parent) :
    QObject(parent)
{
}



bool DatabaseManager::openDB(QString filePath)
{
    if(!db.isOpen())
    {
        // Find QSLite driver
        db = QSqlDatabase::addDatabase("QSQLITE");

#ifdef Q_OS_LINUX
        // NOTE: We have to store database file into user home folder in Linux
        //QString path(QDir::home().path());

        //path.append(QDir::separator()).append("Development").append(QDir::separator()).append("ISL").append(QDir::separator()).append("Datasets").append(QDir::separator()).append("ImageClef2012").append(QDir::separator()).append("bubble.bubbledb");
        QString path = QDir::toNativeSeparators(filePath);

        db.setDatabaseName(path);
        //Development/ISL/Datasets/ImageClef2012
#else
        // NOTE: File exists in the application private folder, in Symbian Qt implementation
        //  bubbledb.setDatabaseName("my.bubbledb.sqlite");
#endif

        // Open databasee
        return db.open();
    }

    return true;
}


QSqlError DatabaseManager::lastError()
{
    // If opening database has failed user can ask
    // error description by QSqlError::text()
    return db.lastError();

}
bool DatabaseManager::isOpen()
{
    return db.isOpen();
}
void DatabaseManager::closeDB()
{
    if(db.isOpen()) db.close();
}

bool DatabaseManager::deleteDB()
{
    // Close database
    db.close();

#ifdef Q_OS_LINUX
    // NOTE: We have to store database file into user home folder in Linux
    QString path(QDir::home().path());
    path.append(QDir::separator()).append("my.bubbledb.sqlite");
    path = QDir::toNativeSeparators(path);
    return QFile::remove(path);
#else

    // Remove created database binary file
    return QFile::remove("my.bubbledb.sqlite");
#endif

}


cv::Mat DatabaseManager::createInvariantMatrix(bool normalise, int harmonics, int orientations, int points, int filter_no){


   int length = (filter_no+2)*harmonics*harmonics;

   int* pointID = new int[ points ];
   for (int i=0; i< points; i++){
        pointID[i]=i;
   }

   std::cout << pointID << std::endl;

   int pointIndex = 0;
   unsigned int i, j, k;
   cv::Mat invariantMatrix(orientations*points, length, CV_32F);

   QSqlQuery query_init(QString("select * from invariant order by number,placelabel"));
    //Each row of the invariant matrix corresponds to one training point
    for(i=0; i<points; i++){
        for(j=0; j<orientations; j++){
            QSqlQuery query(QString("select * from invariant where number = %1 and placelabel = %2").arg(pointID[i]).arg(j));

            query.next();
            for(k=0; k<length; k++){
                invariantMatrix.at<float>(pointIndex,k) = query.value(3).toFloat();
                if(!query.next() && k<length-1)
                    qDebug() << "Too few data";
            }
            pointIndex++;
        }
    }

    delete pointID;

    if(normalise==true){
        cv::Mat dummyVec;
        for(i=0; i<invariantMatrix.rows; i++){
            for(j=0; j<length/(harmonics*harmonics); j++){
                dummyVec = invariantMatrix.rowRange(i, i+1).colRange(j*harmonics*harmonics, (j+1)*harmonics*harmonics);
                cv::normalize(dummyVec, dummyVec);

            }
        }
    }

    return invariantMatrix;
}


cv::Mat DatabaseManager::createLocationMatrix( int points){

    cv::Mat location_matrix(points, 2, CV_64F);
    for(int i=0; i<points; i++){

        QSqlQuery query(QString("select * from invariant where number = %1").arg(i));
        query.next();
        location_matrix.at<double>(i,0)=query.value(4).toFloat();
        location_matrix.at<double>(i,1)=query.value(5).toFloat();

    }

    return location_matrix.clone();
}

