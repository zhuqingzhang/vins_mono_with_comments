#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;//条件变量
double current_time = -1;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;

int sum_of_wait = 0;

//互斥量
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;

//IMU项[P,Q,B,Ba,Bg,a,g]
//全局变量一开始的默认值都为0
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;  //q_W_B  将一个点由机体坐标系转到世界坐标系下。也表示机体在世界坐标系下的姿态
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

//从IMU测量值imu_msg和上一个PVQ递推得到下一个tmp_Q，tmp_P，tmp_V，中值积分
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();

    //init_imu=1表示第一个IMU数据
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }

    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};  //由IMU直接获得的机体(body frame)的线加速度

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};
    //TODO:tmp_Q一开始默认为(0,0,0,0)  ????
    // 其实不影响。tmp的值只是用于在pubLatestOdometry发布的
    // 最开始即使为0也没关系，并不会对状态估计有什么影响。仅仅是发布的数据一开始为0。
    // 在process中通过estimator估计的位姿等信息会在update函数中赋值给tmp变量，
    //之后在imu_callback中调用predict函数时的位姿等数据就是正确的了，从而publish的数据也是正确的
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;   //得到的是上一时刻机体在世界坐标系下的加速度(不包括重力加速度)

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg; //mid-point得到的角速度
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;
    //将当前时刻测得的值赋值给上一时刻相应的变量
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]
//对imu_buf中剩余的imu_msg进行PVQ递推
//在执行update函数的时候，m_state已经被锁住了。所以不会执行imu_callback中的predict步骤
void update()
{
    TicToc t_predict;
    latest_time = current_time;  //调用update是在process函数中将之前的一批数据处理完后进行的。此时current_time为上一批处理的imu数据的最后一帧所对应的时间
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;
    //在process函数中处理上一批数据的时候，imu_buf中又有了新的数据。而此时的tmp数据为上一批处理的数据中最新的一帧的数据。
    // 以该数据为为基础，通过imu_buf来进行imu预测。
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

/**
 * @brief   对imu和图像数据进行对齐并组合
 * @Description     img:    i -------- j  -  -------- k
 *                  imu:    - jjjjjjjj - j/k kkkkkkkk -  
 *                  直到把缓存中的图像特征数据或者IMU数据取完，才能够跳出此函数，并返回数据           
 * @return  vector<std::pair<vector<ImuConstPtr>, PointCloudConstPtr>> (IMUs, img_msg)s
*/
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        //对齐标准：IMU最后一个数据的时间要大于第一个图像特征数据的时间
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        //对齐标准：IMU第一个数据的时间要小于第一个图像特征数据的时间
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue; 
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;

        //图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            //emplace_back相比push_back能更好地避免内存的拷贝与移动
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        //这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());
        
        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

//imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布[P,Q,V,header]
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //判断时间间隔是否为正
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        //ROS_WARN("imu message ins disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();


    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();

    con.notify_one();//唤醒作用于process线程中的con.wait,获取观测值数据的函数


    last_imu_t = imu_msg->header.stamp.toSec();

    {
        //构造互斥锁m_state，析构时解锁
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);//递推得到IMU的PQV
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        //发布最新的由IMU直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

//feature回调函数，将feature_msg放入feature_buf     
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        //TODO：已经在feature_tracker_node中跳过了没有optical flow speed的feature了，这块应该可以不用跳过了。
        init_feature = 1;
        return;
    }

    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();

    con.notify_one();
}

//restart回调函数，收到restart时清空feature_buf和imu_buf，估计器重置，时间重置
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");

        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();

        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();

        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

//relocalization回调函数，将points_msg放入relo_buf        
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

/**
 * @brief   VIO的主线程
 * @Description 等待并获取measurements：(IMUs, img_msg)s，计算dt
 *              estimator.processIMU()进行IMU预积分         
 *              estimator.setReloFrame()设置重定位帧
 *              estimator.processImage()处理图像帧：初始化，紧耦合的非线性优化     
 * @return      void
*/
void process()
{
    while (true)
    {
        //measurements存放了imu和每一帧feature的信息。
        //vec<pair(vec<imu>,feature)> measurements
        // measurements[i].first为 measurements[i-1].second和measurements[i].second这两帧之间的所有imu数据
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        
        std::unique_lock<std::mutex> lk(m_buf);

        //程序运行到这里会阻塞，等待上面的imu_callback和feature_callback中con.notify_one()来唤醒
        //imu_buf和feature_buf有值了后会通过con.notify_one()来唤醒此处的wait，使其进行getMeasurements的函数
        //通过getMeasurements函数的返回值来判断是否继续阻塞。只要return返回的值为false，就继续阻塞。
        //con.wait被唤醒的时候，wait函数自己调用lk.lock(),即在调用getMeasurements的时候，互斥锁m_buf会锁住，此时无法接收数据
        //当返回false后，程序继续阻塞，此时wait会自己调用lk.unlock()，程序可以继续通过imu/feature_callback来获取数据
        //当返回true时，程序朝下进行
        //[&]的意思是以引用的方式将所有父作用域的变量传递到后边的lambda函数中
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();

        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            //对应这段的img data
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;

                //发送IMU数据进行预积分
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t; //IMU初始帧

                    double dt = t - current_time; //imu当前帧时间t减去上一帧时间current_time
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    //imu预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    //插值  (x-last)/dt_1=(cur-x)/dt_2  -->  x=w1*last+w2*cur
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            
            //取出最后一个重定位帧
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }

            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;//归一化平面的x
                    u_v_id.y() = relo_msg->points[i].y;//归一化平面的y
                    u_v_id.z() = relo_msg->points[i].z;//id
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];

                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;

            //建立每个特征点的(camera_id,[x,y,z,u,v,vx,vy])s的map，索引为feature_id， camera_id为实体相机的id，并不是指不同时刻的相机状态对应的id，如果只有一个单目相机，则camera_id只可能为0,如果有两个相机，则camera_id可能为0,1。
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;  //+0.5后取int是为了四舍五入  channel[0]为存储的vector(id)
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i]; //channel[1]为存储的vector(u_of_point)
                double p_v = img_msg->channels[2].values[i]; //channel[2]为存储的vector(v_of_point)
                double velocity_x = img_msg->channels[3].values[i]; //channel[3]为存储的vector(pts_velocity.x)
                double velocity_y = img_msg->channels[4].values[i];  //channel[4]为存储的vector(pts_velocity.y)
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            
            //处理图像特征
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            //给RVIZ发送topic
            pubOdometry(estimator, header);//"odometry" 里程计信息PQV
            pubKeyPoses(estimator, header);//"key_poses" 滑窗中的每一帧（10帧）的位置信息，没有姿态信息
            pubCameraPose(estimator, header);//"camera_pose" 相机位姿
            pubPointCloud(estimator, header);//点云信息（世界坐标系下的）
            pubTF(estimator, header);//"extrinsic" 相机到IMU的外参
            pubKeyframe(estimator);//"keyframe_point"、"keyframe_pose" 关键帧位姿和点云

            if (relo_msg != NULL)
                pubRelocalization(estimator);//"relo_relative_pose" 重定位位姿
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();

        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();//更新IMU参数[P,Q,V,ba,bg,a,g]
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    //ROS初始化，设置句柄n
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    
    //读取参数，设置估计器参数
    //TODO：只进行了一次参数读取，对于NUM_OF_CAM大于1的情形不适用！
    readParameters(n);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    //用于RVIZ显示的Topic
    registerPub(n);

    //订阅IMU、feature、restart、match_points的topic,执行各自回调函数
    //IMU直接从传感器获取， feature和restart从前端feature_tracker_node中发布， match_pionts从pose_graph_node中发布
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    //创建VIO主线程
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
