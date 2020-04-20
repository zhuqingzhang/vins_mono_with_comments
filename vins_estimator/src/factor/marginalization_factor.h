#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;//优化变量数据
    std::vector<int> drop_set;//存放的元素为待边缘化的优化变量id,该id对应paramter_blocks中的id

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;//残差 IMU:15X1 视觉2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;//所有观测项
    //m 要边缘化的变量对应的localsize， n为保留下来的变量的localsize
    int m, n;
    //map的大小对应于优化变量的个数,map中的成员first对应于优化变量的地址，second对应于优化变量的globalsize
    std::unordered_map<long, int> parameter_block_size;
    int sum_block_size;  //保留的变量的globalsize的累加值
    //<待边缘化的优化变量内存地址,每个变量的localsize的依次累加值(用于矩阵中的位置索引)>
    std::unordered_map<long, int> parameter_block_idx;
    std::unordered_map<long, double *> parameter_block_data;//map的大小对应于优化变量的个数,map中的成员first对应于优化变量的地址，second对应于优化变量的相关数据

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size index
    std::vector<double *> keep_block_data;
    //A=J^T*J --> A特征值分解为 VSV^T  --> VSV^T=J^T*J --> Vsqrt(S)sqrt(S)V^T =(sqrt(S)V^T)^T*(sqrt(S)V^T)=J^T*J
    //--> linearized_jacobians 对应于 sqrt(S)V^T
    //b=J^T*r -->r=(J^T)^-1*b -->
    //  linearized_residuals对应于 sqrt(S)^-1*V^T*b
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
