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

    
    






}}