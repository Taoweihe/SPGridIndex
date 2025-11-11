#pragma once
#include <muda/muda.h>
#include <muda/ext/eigen/eigen_dense_cxx20.h>
#include <Eigen/Dense>
namespace muda{
namespace SParseGrid{

    template<bool Isconst , typename GridLayout ,typename TransformValueType>
    class GridView : public ViewBase<Isconst>{
        using Base = ViewBase<Isconst>;
        template<typename U>
        using auto_const_t = typename Base::template auto_const_t<U>;

        template<bool OtherIsConst , typename U , typename V>
        friend class GridView;

        public:

        using NonConstView = GridView<false, GridLayout, TransformValueType>;
        using ConstView    = GridView<true, GridLayout, TransformValueType>;
        using ThisView     = GridView<Isconst, GridLayout, TransformValueType>;


        MUDA_GENERIC GridView() = default;


        MUDA_GENERIC GridView(size_t num_blocks,
                              auto_const_t<Eigen::Vector<TransformValueType, 3>>* trans,
                              auto_const_t<uint64_t>* key,
                              auto_const_t<int>* value,
                              auto_const_t<uint64_t>* block_id2key,
                              auto_const_t<int>* topology_cache)
            : num_blocks(num_blocks),
              transform(trans),
              key(key),
              value(value),
              block_id2key(block_id2key),
              topology_cache(topology_cache)
        {
        };


        


        protected:
        size_t num_blocks;
        Eigen::Vector<auto_const_t<TransformValueType> ,3>* transform;

        auto_const_t<uint64_t>* key;
        auto_const_t<int>*      value;
        auto_const_t<uint64_t>* block_id2key;
        auto_const_t<int>*      topology_cache;
        


};
}}