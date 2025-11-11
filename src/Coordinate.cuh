#pragma once
#include <muda/muda.h>
#include <muda/ext/eigen/eigen_dense_cxx20.h>
#include "utils.cuh"
namespace muda{

    namespace SParseGrid{
    
    template<typename GridLayout>
    struct neibour{

        public:

        neibour() = default;
        neibour(const int& block_id , const int& local_index , const Eigen::Vector3i& offset)
            :m_block_id(block_id) , m_cell_id(local_index) , m_offset(offset){}
        
        MUDA_GENERIC int block_id(){return m_block_id;}
        MUDA_GENERIC int cell_id(){return m_cell_id;}
        MUDA_GENERIC Eigen::Vector3i offset(){return m_offset;}
        MUDA_GENERIC int topology(){return topology_id;};
        
        neibour &result(){

            Eigen::Vector3i linear_vec = linear_decode<GridLayout , int>(m_cell_id);
            linear_vec += m_offset;
            linear_vec(0) = std::abs((linear_vec(0) + GridLayout::nx) % GridLayout::nx);
            linear_vec(1) = std::abs((linear_vec(1) + GridLayout::ny) % GridLayout::ny);
            linear_vec(2) = std::abs((linear_vec(2) + GridLayout::nz) % GridLayout::nz);

            int topology_id_x_new = linear_vec(0) / GridLayout::nx + 1;
            int topology_id_y_new = linear_vec(1) / GridLayout::ny + 1;
            int topology_id_z_new = linear_vec(2) / GridLayout::nz + 1;
            
             topology_id = topology_id_x_new + topology_id_y_new * 3 + topology_id_z_new * 9;
             return *this;
        };   

        int m_block_id;
        int m_cell_id;
        int topology_id;
        Eigen::Vector3i m_offset;

       
        
    };

    // uint32_t or uint64_t
    template< typename GridLayout , typename CoordinateType = uint64_t>
    struct Coordinate{

        Coordinate() = default;
        Coordinate(const CoordinateType& morton_coordinate):morton_coordinate(morton_coordinate){};
        
        template<typename VecType = int>
        MUDA_GENERIC Eigen::Vector<VecType,3> toVec() const{
            Eigen::Vector<VecType,3> coord_vec;
            coord_vec = decode<CoordinateType , VecType>(morton_coordinate);
            return coord_vec;
        }
        
        template<typename VecType>
        MUDA_GENERIC  Coordinate& fromVec(const Eigen::Vector<VecType ,3>& coord_vec){
            Eigen::Vector3i vec = bound_round(coord_vec);
            CoordinateType morton_code = encode<CoordinateType , int>(vec);
            morton_coordinate = morton_code;
            return *this;
        };
        
        MUDA_GENERIC CoordinateType block_key() const{
            return morton_coordinate >> GridLayout::log2_voxel_num;
        }

        MUDA_GENERIC int cell_index() const{
            uint32_t code = morton_coordinate & ((1 << GridLayout::log2_voxel_num) - 1);
            auto Vec = decode_morton<CoordinateType , int>(code) ;
            int local_index = linear_encode<GridLayout , int>(Vec);

        }

        
        /*MUDA_GENERIC neibour<GridLayout> offsetby(Eigen::Vector3i offset) const{
           MUDA_ASSERT(this->toVec<Eigen::Vector3f>().x + offset.x() > 0 &&
                       this->toVec<Eigen::Vector3f>().y + offset.y() > 0 &&
                       this->toVec<Eigen::Vector3f>().z + offset.z() > 0 , "Coordinate offset out of bound");
            return neibour<GridLayout>{block_key() , local_index() , offset};
        }

        MUDA_GENERIC neibour<GridLayout> offsetby(int x , int y , int z) const{
            Eigen::Vector3i offset(x , y , z);
            return neibour<GridLayout>{block_key() , local_index() , offset};
        }*/

        private:
        CoordinateType morton_coordinate;
    };

    template<typename GridLayout>
    struct Index{

        public:

        Index() = default;
        Index(int blockid , int cell_id):m_blockid(blockid) , m_cell_id(cell_id){}

        MUDA_GENERIC int blockid() const{
            return m_blockid;
        }

        MUDA_GENERIC int cell_id() const{
            return m_cell_id;
        }

        MUDA_GENERIC int index_1d() const{
            return m_blockid * GridLayout::voxel_num + m_cell_id;
        }

        MUDA_GENERIC neibour<GridLayout> offsetby(Eigen::Vector3i offset) const{
           
            return neibour<GridLayout>{m_blockid , m_cell_id , offset};
        }


        int m_blockid;
        int m_cell_id;
        
        
    };
    
    
    }


}
