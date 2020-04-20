#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();  //R_2_1，g在标准系下的旋转矩阵
    double yaw = Utility::R2ypr(R0).x();//g在相对于标准系下的偏航角
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;  //使g相对于标准系的偏航角为0.
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
