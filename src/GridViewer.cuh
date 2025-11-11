#pragma once
#include <muda/ext/eigen/eigen_dense_cxx20.h>
#include <muda/muda.h>
#include "Coordinate.cuh"

namespace muda{
namespace SParseGrid{


    template<bool Isconst , typename GridLayout , typename TransformValueType>
    class GridViewer :ViewerBase<Isconst>{

    using Base = ViewerBase<Isconst>;
    template <typename U>
    using auto_const_t = typename Base::template auto_const_t<U>;
  

    
    template<bool OtherIsConst , typename U , typename V>
    friend class GridViewer;

    public:

    using NonConstView = GridViewer<false, GridLayout, TransformValueType>;
    using ConstView    = GridViewer<true, GridLayout, TransformValueType>;        
    using ThisView     = GridViewer<Isconst, GridLayout, TransformValueType>;

    GridViewer() = default;

    GridViewer(const GridViewer<false, GridLayout, TransformValueType>& other)
        :num_blocks(other.num_blocks),
         transform(other.transform),
         key(other.key),
         value(other.value),
         block_id2key(other.block_id2key),
         topology_cache(other.topology_cache)
    {}

    GridViewer(size_t                    num_blocks ,
               Eigen::Vector<auto_const_t<int> , 3> transform ,
               auto_const_t<uint64_t>*  key ,
               auto_const_t<int>*       value ,
               auto_const_t<uint64_t>*  block_id2key ,
               auto_const_t<int>*       topology_cache )
        :num_blocks(num_blocks),
         transform(transform),
         key(key),
         value(value),
         block_id2key(block_id2key),
         topology_cache(topology_cache)
    {}
    
    template<typename CoordinateType>
    MUDA_GENERIC Index<GridLayout> operator()(Coordinate<GridLayout , CoordinateType> coordinate){
        CoordinateType block_key = coordinate.block_key();
        uint32_t cell_index = coordinate.cell_index();

        CoordinateType hashkey = block_key >> GridLayout::log2_voxel_num;
        int block_id = value[hashkey]; 
        


        return Index<GridLayout>{block_id , cell_index};

    }
    
    template<typename CoordinateType>
    MUDA_GENERIC Index<GridLayout> operator()(const neibour<GridLayout>& neib){

        neib.result();
        int block_id = neib.block_id();
        int cell_index = neib.cell_id();
        int topology_id = neib.topology();

        int block_id_new = topology_cache[block_id * 27 + topology_id];
        
        return Index<GridLayout>{block_id_new , cell_index};
    };
    template<typename VecType>
    MUDA_GENERIC Eigen::Vector<VecType,3> world_to_index(Eigen::Vector<VecType,3> & vec){
        auto index_x = vec/transform(0);
        auto index_y = vec/transform(1);
        auto index_z = vec/transform(2);

        return Eigen::Vector<VecType,3>(index_x,index_y,index_z);
    };

    template<typename CoordinateType , typename VecType>
    MUDA_GENERIC Coordinate<GridLayout,CoordinateType> vec_to_coordinate(Eigen::Vector<VecType,3> &index_vec){

        Coordinate<GridLayout,CoordinateType> coord;
        coord.fromVec(index_vec);
        return coord;
    }; 

    template<typename CoordinateType>
    MUDA_GENERIC Coordinate<GridLayout> make_coordinate(CoordinateType morton_code){
        Coordinate<GridLayout,CoordinateType> coord(morton_code);
        return coord;
    };

    protected:

    size_t num_blocks;
    Eigen::Vector<auto_const_t<int>, 3> transform;

    auto_const_t<uint64_t>* key;
    auto_const_t<int>*      value;
    auto_const_t<uint64_t>* block_id2key;
    auto_const_t<int>*      topology_cache;


    };

}
}