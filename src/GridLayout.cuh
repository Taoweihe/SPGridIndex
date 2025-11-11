#pragma once
#include <muda/muda.h>
#include <muda/ext/eigen/eigen_dense_cxx20.h>

template<size_t block_nx = 4 , size_t block_ny = 4, size_t block_nz = 4>
class GridLayout{

    public:

    static constexpr size_t dim = 3;
    static constexpr size_t nx = block_nx;
    static constexpr size_t ny = block_ny;
    static constexpr size_t nz = block_nz;
    static constexpr size_t log2_voxel_num = static_cast<size_t>(std::log2(block_nx * block_ny * block_nz));

    

};