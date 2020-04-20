#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    //x为7维，delta为6维
    Eigen::Map<const Eigen::Vector3d> _p(x);  //Eigen::Map 将指针映射成一个Eigen数据类型，从而直接进行矩阵操作。与指针共享地址
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3)); //delta对应的dq是三维的

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}
//The jacobian of Plus(x, delta) w.r.t delta at delta = 0
//按理说，我们要求得的是量测误差相对于扰动的雅可比：J_error_delta
//但是由于Ceres的机制，我们只能现在CostFunction中的ComputeJacobian中计算量测误差相对于变量的雅可比：J_error_var
//然后再在LocalParameter中的ComputeJacobian中变量相对于扰动的雅可比：J_var_delta
//然后两者相乘得到J_error_delta=J_error_var*J_var_delta。这样做有点多此一举，我们通常能直接求得J_error_delta
//但是Ceres的机制非要将两个雅可比矩阵相乘，
//基于Ceres的这种机制，如果我们分别计算这两个雅可比，再相乘，就非常的费时间费算力，所以有如下的trick：
//       Note:很多情况下var是过参数化的，而delta的参数量是刚刚好的。
//      比如对于旋转，四元数q就相当于var，其维度是4，而我们更新变量时，用的delta的维度是3。
//0、假设var的维度为4,delta的维度为3，error的维度为2, 则J_error_delta(2*3)=J_error_var(2*4)*J_var_delta(4*3)
//1、在CostFunction中的ComputeJacobian我们将J_error_var的前3列填充为事先手动求好J_error_delta的值，最后一列填充为0.
//2、在LocalParameter中的ComputeJacobian我们将J_var_delta的前三行设为单位矩阵，最后一行设为0.
//3、这样以来，J_error_var*J_var_delta仍然能得到J_error_delta, 但省去了复杂的矩阵计算，仅仅是简单的矩阵计算，省时省力。
//  J_error_var(2*4) <--- [J_error_delta(2*3) | 0(2*1)]
//  J_var_delta(4,3) <--- [ I (3*3) ]
//                        [ 0 (1*3) ]
//  从而有 J_error_delta=J_error_var*J_var_delta
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
