#ifndef DATABASEMANAGER_H
#define DATABASEMANAGER_H

#include "bubbleprocess.h"
#include <QObject>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlError>
#include <QFile>
#include <QDir>
#include <QMatrix>

//#define DB_PATH "/home/hakan/Development/ISL/Datasets/ImageClef2012/training3v2.db"
//#define INVARIANTS_DB_PATH "/home/hakan/Development/ISL/Datasets/ImageClef2012/invariants.db"

#define LASER_TYPE 55
#define HUE_TYPE 56
#define SAT_TYPE 57
#define VAL_TYPE 58

class DatabaseManager : public QObject
{
    Q_OBJECT
public:
    explicit DatabaseManager(QObject *parent = 0);
   // ~DatabaseManager();

   static bool openDB(QString filePath);

   static void closeDB();

   static bool deleteDB();

   static bool isOpen();

   static cv::Mat createInvariantMatrix(bool normalise, int harmonics, int orientations, int points, int filter_no);

   static cv::Mat createLocationMatrix(int points);

   QSqlError lastError();

private:
 //  QSqlDatabase db;

    
signals:
    
public slots:
    
};

#endif // DATABASEMANAGER_H
