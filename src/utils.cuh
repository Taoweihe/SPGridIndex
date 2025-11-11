#pragma once
#include <muda/muda.h>
#include <muda/ext/eigen/eigen_dense_cxx20.h>

namespace muda{
namespace SParseGrid{

// Helper function: 分离位（用于Morton码编码）
template<typename T>
MUDA_GENERIC T split_by_3(T a) {
    T x = a & 0x1fffff; // 只保留21位
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8)  & 0x100f00f00f00f00f;
    x = (x | x << 4)  & 0x10c30c30c30c30c3;
    x = (x | x << 2)  & 0x1249249249249249;
    return x;
}

// Helper function: 合并位（用于Morton码解码）
template<typename T>
MUDA_GENERIC T compact_by_3(T x) {
    x &= 0x1249249249249249;
    x = (x ^ (x >> 2))  & 0x10c30c30c30c30c3;
    x = (x ^ (x >> 4))  & 0x100f00f00f00f00f;
    x = (x ^ (x >> 8))  & 0x1f0000ff0000ff;
    x = (x ^ (x >> 16)) & 0x1f00000000ffff;
    x = (x ^ (x >> 32)) & 0x1fffff;
    return x;
}

// 32位版本的位分离
MUDA_GENERIC uint32_t split_by_3_32(uint32_t a) {
    uint32_t x = a & 0x3ff; // 只保留10位
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8)  & 0x300f00f;
    x = (x | x << 4)  & 0x30c30c3;
    x = (x | x << 2)  & 0x9249249;
    return x;
}

// 32位版本的位合并
MUDA_GENERIC uint32_t compact_by_3_32(uint32_t x) {
    x &= 0x9249249;
    x = (x ^ (x >> 2))  & 0x30c30c3;
    x = (x ^ (x >> 4))  & 0x300f00f;
    x = (x ^ (x >> 8))  & 0x30000ff;
    x = (x ^ (x >> 16)) & 0x3ff;
    return x;
}

// Morton编码：将3D坐标编码为Morton码
template<typename CoordinateType, typename VecType>
MUDA_GENERIC CoordinateType encode_morton(const Eigen::Vector<VecType,3>& coord_vec) {
    using UIntType = typename std::conditional<
        sizeof(CoordinateType) == 4, uint32_t, uint64_t>::type;
    
    UIntType x = static_cast<UIntType>(coord_vec(0));
    UIntType y = static_cast<UIntType>(coord_vec(1));
    UIntType z = static_cast<UIntType>(coord_vec(2));
    
    if constexpr (sizeof(CoordinateType) == 4) {
        // 32位：每个坐标最多10位
        return static_cast<CoordinateType>(
            split_by_3_32(x) | (split_by_3_32(y) << 1) | (split_by_3_32(z) << 2)
        );
    } else {
        // 64位：每个坐标最多21位
        return static_cast<CoordinateType>(
            split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2)
        );
    }
}

// Morton解码：将Morton码解码为3D坐标
template<typename CoordinateType, typename VecType>
MUDA_GENERIC Eigen::Vector<VecType, 3> decode_morton(const CoordinateType& morton_coordinate) {
    using UIntType = typename std::conditional<
        sizeof(CoordinateType) == 4, uint32_t, uint64_t>::type;
    
    UIntType morton = static_cast<UIntType>(morton_coordinate);
    
    if constexpr (sizeof(CoordinateType) == 4) {
        // 32位版本
        VecType x = static_cast<VecType>(compact_by_3_32(morton));
        VecType y = static_cast<VecType>(compact_by_3_32(morton >> 1));
        VecType z = static_cast<VecType>(compact_by_3_32(morton >> 2));
        return Eigen::Vector<VecType, 3>(x, y, z);
    } else {
        // 64位版本
        VecType x = static_cast<VecType>(compact_by_3(morton));
        VecType y = static_cast<VecType>(compact_by_3(morton >> 1));
        VecType z = static_cast<VecType>(compact_by_3(morton >> 2));
        return Eigen::Vector<VecType, 3>(x, y, z);
    }
}

// 线性编码：将3D坐标编码为1D索引（行优先）
template<typename GridLayout, typename VecType>
MUDA_GENERIC int linear_encode(const Eigen::Vector<VecType, 3>& vec) {
    // 假设GridLayout有resolution()方法返回Eigen::Vector3i
    auto res_x = GridLayout::nx;
    auto res_y = GridLayout::ny;
    
    
    int x = static_cast<int>(vec(0));
    int y = static_cast<int>(vec(1));
    int z = static_cast<int>(vec(2));
    
    
    return z * res_x*res_y + y * res_x + x;
}

// 线性解码：将1D索引解码为3D坐标
template<typename GridLayout>
MUDA_GENERIC Eigen::Vector3i linear_decode(int linear_code) {
    auto res_x = GridLayout::nx;
    auto res_y = GridLayout::ny;
    
    int x = linear_code % res_x;
    int y = (linear_code / res_x) % res_y;
    int z = linear_code / (res_x * res_y);
    
    return Eigen::Vector3i(x, y, z);
}

// 边界取整：将浮点坐标转换为整数坐标（向下取整）
template<typename VecType>
MUDA_GENERIC Eigen::Vector<int, 3> bound(const Eigen::Vector<VecType, 3>& vec) {
    // 在CUDA中使用floor函数
#ifdef __CUDA_ARCH__
    int x = static_cast<int>(::floor(vec(0)));
    int y = static_cast<int>(::floor(vec(1)));
    int z = static_cast<int>(::floor(vec(2)));
#else
    int x = static_cast<int>(std::floor(vec(0)));
    int y = static_cast<int>(std::floor(vec(1)));
    int z = static_cast<int>(std::floor(vec(2)));
#endif
    
    return Eigen::Vector3i(x, y, z);
}

// 如果需要向最近整数取整的版本
template<typename VecType>
MUDA_GENERIC Eigen::Vector<int, 3> bound_round(const Eigen::Vector<VecType, 3>& vec) {
#ifdef __CUDA_ARCH__
    int x = static_cast<int>(::round(vec(0)));
    int y = static_cast<int>(::round(vec(1)));
    int z = static_cast<int>(::round(vec(2)));
#else
    int x = static_cast<int>(std::round(vec(0)));
    int y = static_cast<int>(std::round(vec(1)));
    int z = static_cast<int>(std::round(vec(2)));
#endif
    
    return Eigen::Vector3i(x, y, z);
}

    
    MUDA_GENERIC static uint64_t expand_bits(uint32_t v) {
    uint64_t x = v;
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8))  & 0x100f00f00f00f00f;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

MUDA_GENERIC static uint32_t compact_bits(uint64_t x) {
    x = x & 0x1249249249249249;
    x = (x | (x >> 2))  & 0x10c30c30c30c30c3;
    x = (x | (x >> 4))  & 0x100f00f00f00f00f;
    x = (x | (x >> 8))  & 0x1f0000ff0000ff;
    x = (x | (x >> 16)) & 0x1f00000000ffff;
    x = (x | (x >> 32)) & 0x1fffff;
    return (uint32_t)x;
}

// 将3D坐标编码为Morton码
MUDA_GENERIC static uint64_t encode_morton(int x, int y, int z) {
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

// 将Morton码解码为3D坐标
MUDA_GENERIC static void decode_morton(uint64_t morton, int* x, int* y, int* z) {
    *x = compact_bits(morton);
    *y = compact_bits(morton >> 1);
    *z = compact_bits(morton >> 2);
}

// 对Morton码进行偏移，返回新的Morton码
MUDA_GENERIC static uint64_t packed_add_morton(uint64_t data, int x, int y, int z) {
    // 解码原始Morton码
    int orig_x, orig_y, orig_z;
    decode_morton(data, &orig_x, &orig_y, &orig_z);
    
    // 应用偏移
    int new_x = orig_x + x;
    int new_y = orig_y + y;
    int new_z = orig_z + z;
    
    // 边界检查（假设坐标范围为非负数）
    if (new_x < 0) new_x = 0;
    if (new_y < 0) new_y = 0;
    if (new_z < 0) new_z = 0;
    
    // 重新编码为Morton码
    return encode_morton(new_x, new_y, new_z);
}

// 对线性编码进行偏移，返回新的线性编码
// 假设线性编码是按照某种3D空间划分规则（如体素网格）进行的
// 这里假设是在一个固定大小的3D网格中进行线性编码
template <size_t nx = 4 , size_t ny = 4 , size_t nz = 4>
MUDA_GENERIC static uint64_t packed_add_linear(int id, int x, int y, int z) {
    

    constexpr int GRID_SIZE_X = nx;
    constexpr int GRID_SIZE_Y = ny;
    constexpr int GRID_SIZE_Z = nz;
    
    // 从线性ID解码出3D坐标
    int orig_z = id / (GRID_SIZE_X * GRID_SIZE_Y);
    int remaining = id % (GRID_SIZE_X * GRID_SIZE_Y);
    int orig_y = remaining / GRID_SIZE_X;
    int orig_x = remaining % GRID_SIZE_X;
    
    // 应用偏移
    int new_x = orig_x + x;
    int new_y = orig_y + y;
    int new_z = orig_z + z;
    
    // 边界检查和环绕处理
    new_x = ((new_x % GRID_SIZE_X) + GRID_SIZE_X) % GRID_SIZE_X;
    new_y = ((new_y % GRID_SIZE_Y) + GRID_SIZE_Y) % GRID_SIZE_Y;
    new_z = ((new_z % GRID_SIZE_Z) + GRID_SIZE_Z) % GRID_SIZE_Z;
    
    // 重新编码为线性ID
    return new_z * (GRID_SIZE_X * GRID_SIZE_Y) + new_y * GRID_SIZE_X + new_x;
}

__host__ __device__ int encode_linear(int x , int y , int z , int nx , int ny , int nz){

    return x + y*nx + z*nx*ny;
    
};


MUDA_GENERIC void decode_linear(int id , int& x , int& y ,int& z, int nx ,int ny, int nz){

    z = id/(nx*ny);
    y = (id - z*nx*ny) / nx;
    x = (id - z*nx*ny - y *nx );

    

}

MUDA_GENERIC  int topology_id_compute(int &cell_id , int cell_dx , int cell_dy , int cell_dz , int nx , int ny ,int nz){

        int cell_x = 0;
        int cell_y = 0;
        int cell_z = 0;

        decode_linear(cell_id, cell_dx, cell_dy, cell_dz, nx, ny, nz);

        int block_id_x_new = (cell_x + cell_dx) / nx + 1;
        int block_id_y_new = (cell_y + cell_dy) / ny + 1;
        int block_id_z_new = (cell_z + cell_dz) / nz + 1;
        
        int cell_id_x_new = (cell_dx + cell_x) % nx;
        int cell_id_y_new = (cell_dy + cell_y) % ny;
        int cell_id_z_new = (cell_dz + cell_z) % nz;

        
        cell_id = encode_linear(cell_id_x_new, cell_id_y_new, cell_id_z_new, nx, ny, nz);
        return encode_linear(block_id_x_new, block_id_y_new, block_id_z_new, nx, ny, nz);

}








}}