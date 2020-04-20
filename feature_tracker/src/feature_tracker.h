#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

/**
* @class FeatureTracker
* @Description 视觉前端预处理：对每个相机进行角点LK光流跟踪
*/
class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;//图像掩码
    cv::Mat fisheye_mask;//鱼眼相机mask，用来去除边缘噪点

    // prev_img是上一次发布的帧的图像数据
    // cur_img是光流跟踪的前一帧的图像数据
    // forw_img是光流跟踪的后一帧的图像数据，真正意义上的当前帧
    cv::Mat prev_img, cur_img, forw_img;

    vector<cv::Point2f> n_pts;//每一帧中新提取的特征点

    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;

    vector<cv::Point2f> prev_un_pts, cur_un_pts;//归一化相机坐标系下的坐标
    
    vector<cv::Point2f> pts_velocity;//当前帧相对前一帧特征点沿x,y方向的像素移动速度

    vector<int> ids;//能够被跟踪到的特征点的id

    vector<int> track_cnt;//当前帧forw_img中每个特征点被追踪的时间次数

    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;

    camodocal::CameraPtr m_camera;//相机模型

    double cur_time; //对应于forw_img的时间
    double prev_time; //对应于cur_img的时间

    static int n_id;//特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
};

/*值得学习的：
 * 1、从vector中剔除标记为0的点：  reduceVector
     void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
   {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
   }
 *2、rejectWithF中用liftProjective得到un_point有啥用？  是去畸变后的点？
 *   listProjective对于pinhole的相机模型来说，具体的实现在PinholeCamera.cc 450行
 *   该函数中实现了对传入的特征点“迭代去畸变”的过程。最终得到的un_point是normalize单位化后去了畸变的点,再除以z坐标的值就可以得到归一化平面上的点
 *3、利用opencv中的goodFeaturesToTrack可以直接提取图像中的角点