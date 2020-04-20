#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

//清楚特征管理器中的特征
void FeatureManager::clearState()
{
    feature.clear();
}

//窗口中被跟踪的特征数量
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();
       //满足这两个条件的特征点才算是被跟踪上。
       //第一个条件确保有两帧以上观测到该点
       //第二个条件确保第一次观测到该点的帧要比次次新帧老。
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)  //观察到该特征点的第一帧图像必须在滑窗的前8帧才可。
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief   把特征点放入feature的list容器中，计算每一个点跟踪次数和它在次新帧和次次新帧间的视差，返回是否是关键帧
 * @param[in]   frame_count 为 传入的image为滑窗中的第几帧
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   td IMU和cam同步时间差
 * @return  bool true：次新帧是关键帧;false：非关键帧
*/
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    //把image map中的所有特征点放入feature list容器中
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        //迭代器寻找feature list中是否有这feature_id
        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        //滑窗中没有该特征点，则说明该特征点是在image中被第一次观测到，将该特征点存入feature中
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        //如果之前就存在该特征点了，就把image帧添加到feature_per_id中的feature_per_frame，同时跟踪上的点的数量加一
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    //如果image是滑窗中的前两帧或者image中跟踪上之前帧的点数过少（小于20个），就将该帧设为关键帧
    if (frame_count < 2 || last_track_num < 20)
        return true;

    //计算每个特征在次新帧和次次新帧中的视差
    for (auto &it_per_id : feature)
    {

        //对于某一个特征点pt，第一次观测到特征点的(帧id)小于(当前帧id-2);观测到该pt的最后一帧id要大于等于(当前帧id-1)
        //这两个判别条件确保了it_per_id.feature_per_frame中有次新帧和次次新帧
        //第一个条件确保第一次观测到该点的帧的id要小于等于次次新帧
        //第二个条件确保有次新帧观测到该点
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            //计算it_per_id在次新帧和次次新帧之间的归一化坐标的前两维的距离，记为视差
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    //如果paralax_num为0，说明上边的if条件不满足，也就是说当前帧(最新帧)观测到的所有点都不在次新帧或次次新帧中观测到，
    //则当前帧为关键帧。应该不会出现这种情况。否则last_track_num一定小于20,在之前就return true了
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH); //单位是pixel
        return parallax_sum / parallax_num >= MIN_PARALLAX;  //单位是米  视差大于10个像素，约2厘米就设置关键帧
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

//得到frame_count_l与frame_count_r两帧之间的对应特征点
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

//设置特征点的逆深度估计值
//传入的VectorXd存放的是特征点的深度值
void FeatureManager::setDepth(const VectorXd &x )
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;//失败估计
        }
        else
            it_per_id.solve_flag = 1;//成功估计
    }
}

//剔除feature中估计失败的点（solve_flag == 2）
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

//TODO:和setDepth几乎一样 有啥用？
//在Estimator::visualInitialAlign()中，先通过getDepthVector获得dep_vec,然后将其全部赋值为-1,然后传入该函数中，
//使所有的特征点的深度都为-1。，完成clearDepth
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

//从逆深度转成深度
VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

//对特征点进行三角化求深度（SVD分解）
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        //R0 t0为第i帧相机坐标系到参考系的变换矩阵
        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            //R t为第j帧相机坐标系到第i帧(start_frame)相机坐标系的变换矩阵，P为i到j的变换矩阵
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;   //R_startframe_j
            Eigen::Matrix<double, 3, 4> P;  //P_j_startframe
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            //P = [P1 P2 P3]^T 
            //AX=0      A = [A(2*i) A(2*i+1) A(2*i+2) A(2*i+3) ...]^T
            //A(2*i)   = x(i) * P3 - z(i) * P1
            //A(2*i+1) = y(i) * P3 - z(i) * P2
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j) //这句应该放在前边
                continue;
        }
        //对A的SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;  //注意，这块的是深度值，不是逆深度值,得到的深度值是相对于start_frame的
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

//移除外点
void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

//边缘化最老帧时，处理特征点保存的帧号，将起始帧是最老帧的特征点的深度值进行转移
//marg_R、marg_P为被边缘化的位姿，new_R、new_P为下一帧的位姿
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        //特征点起始帧不是最老帧则将帧号减一
        if (it->start_frame != 0)
            it->start_frame--;
        else//特征点起始帧是最老帧
        {

            //获得起始帧对当前点的观测
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            //将起始帧删除
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            //删除起始帧后，如果仅剩一帧能观测到特征点，则认为该点质量不好，将该点删除
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                //pts_i为特征点在起始帧(最老帧)坐标系下的三维坐标
                //w_pts_i为特征点在世界坐标系下的三维坐标
                //将其转换到在下一帧坐标系下的坐标pts_j
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;  //深度值
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

//边缘化最老帧时，直接将特征点所保存的帧号向前滑动
//和FeatureManager::removeBackShiftDepth的实现相似 只不过后者还计算了特征点在新的start_frame下的深度值
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        //如果特征点起始帧号start_frame不为零则减一
        if (it->start_frame != 0)
            it->start_frame--;
        //如果start_frame为0则直接移除feature_per_frame的第0帧FeaturePerFrame
        //如果feature_per_frame为空则直接删除特征点
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

//边缘化次新帧时，对特征点在次新帧的信息进行移除处理
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        //起始帧为最新帧的滑动成次新帧
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            //start_frame距离滑窗最后一帧（次新帧）的距离。
            // 最新帧frame_count还没有加到滑窗中，因为滑窗已经满了，需要剔除次新帧才能将最新帧加入到滑窗中
            int j = WINDOW_SIZE - 1 - it->start_frame;
            //如果次新帧之前已经跟踪结束(即次新帧并没有跟踪上该点)则什么都不做
            if (it->endFrame() < frame_count - 1)
                continue;
            //如果在次新帧仍被跟踪，则删除feature_per_frame中次新帧对应的FeaturePerFrame
            //如果feature_per_frame为空则直接删除特征点
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

//计算某个特征点it_per_id在次新帧和次次新帧的视差
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    //feature_per_frame[0]-->start_frame  feature_per_frame[frame_count-start_frame]-->the_last_frame
    //feature_per_frame[frame_count-start_frame-1]-->second last frame
    //feature_per_frame[frame_count-start_frame-2]-->third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;



    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;
    //du_comp 和 du一样的...
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}