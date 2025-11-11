#pragma once
#include <muda/muda.h>
#include <GridViewer.cuh>
#include <GridView.cuh>
#include <GridLayout.cuh>

namespace muda{

 
    namespace SParseGrid{

    

    template<typename GridLayout , typename TransformValueType>
    class Grid{

        
        public:

        Grid() = default ;
        Grid(const size_t& num_blocks ,
             const Eigen::Vector<TransformValueType , 3>& dx ){
               m_num_blocks = num_blocks;
               key.resize(num_blocks , 0xffffffffffffffff);
               value.resize(num_blocks , -1);
               block_id2key.resize(num_blocks , 0xffffffffffffffff);
               topology_cache.resize(num_blocks * 27 , -1);
               block_counter = 0;
               block_counter_before_adjancent_cache = 0;
             }

        MUDA_HOST void rebuild_map(muda::BufferView<uint64_t> offset_key ,
                                       muda::BufferView<int> index){
                key.fill(0xffffffffffffffff);
                value.fill(-1);
                block_id2key.fill(0xffffffffffffffff);
                topology_cache.fill(-1);
                block_counter = 0;
                block_counter_before_adjancent_cache = 0;

                build_map(offset_key , index);
                                       }
        

        MUDA_HOST void build_map(muda::BufferView<uint64_t> offset_key ,
                                       muda::BufferView<int> index){
            grid_hash_build(offset_key, index);
            std::cout<<"complte_build_hash"<<std::endl;
            counter_cache();
            std::cout<<"begin supple"<<std::endl;
            supplementAdjancentBlocks();
            std::cout<<"supple,emtAdkamcentBlocks"<<std::endl;
            Grid_build_topology();
            std::cout<<"complete_build_topology"<<std::endl;
            
                                       }

        MUDA_HOST void grid_hash_build(muda::BufferView<uint64_t> offset_key ,
                                       muda::BufferView<int> index);
        
        MUDA_GENERIC void counter_cache(){block_counter_before_adjancent_cache = block_counter;}

        MUDA_HOST void Grid_build_topology();

        MUDA_HOST void supplementAdjancentBlocks();

        GridView<0, GridLayout, TransformValueType> view(){
            return GridView<0, GridLayout, TransformValueType>(
                m_num_blocks,
                transform.data(),
                key.data(),
                value.data(),
                block_id2key.data(),
                topology_cache.data()
            );
        }

        GridViewer<0,  GridLayout,  TransformValueType> Viewer(){
            return GridViewer<0, GridLayout, TransformValueType>(
                m_num_blocks,
                transform.data(),
                key.data(),
                value.data(),
                block_id2key.data(),
                topology_cache.data()
            );
        }

        GridView<1, GridLayout, TransformValueType> CView(){
            return GridView<1, GridLayout, TransformValueType>(
                m_num_blocks,
                transform.data(),
                key.data(),
                value.data(),
                block_id2key.data(),
                topology_cache.data()
            );
        }

        GridViewer<1, GridLayout, TransformValueType> CViewer(){
            return GridViewer<1, GridLayout, TransformValueType>(
                m_num_blocks,
                transform.data(),
                key.data(),
                value.data(),
                block_id2key.data(),
                topology_cache.data()
            );
        }

        private:
        
        //transform
        DeviceVar<Eigen::Vector<TransformValueType , 3>> transform;

        //hash
        size_t m_num_blocks;
        muda::DeviceBuffer<uint64_t> key;
        muda::DeviceBuffer<int> value;
        muda::DeviceBuffer<uint64_t> block_id2key;
        muda::DeviceVar<int> block_counter;
        muda::DeviceBuffer<int> topology_cache;
        muda::DeviceVar<int> block_counter_before_adjancent_cache;
        
    };
    

    template<typename GridLayout, typename TransformValueType>
    Grid<GridLayout,TransformValueType> make_grid(const size_t& num_blocks ,
              const Eigen::Vector<TransformValueType , 3>& dx = Eigen::Vector<TransformValueType,3>(1.0f , 1.0f ,1.0f) ){
                  return Grid<GridLayout,TransformValueType>(num_blocks , dx);
              };
    template<typename TransformValueType = float>
    Grid<GridLayout<4,4,4> , TransformValueType> make_grid(const size_t& num_blocks ,
              const Eigen::Vector<TransformValueType , 3>& dx = Eigen::Vector<TransformValueType,3>(1.0f , 1.0f ,1.0f) ){
                  return Grid<GridLayout<4,4,4> ,TransformValueType>(num_blocks , dx);
              };
    
    template<typename TransformValueType , size_t block_nx , size_t block_ny , size_t block_nz>
    Grid<GridLayout<block_nx , block_ny , block_nz> , TransformValueType> make_grid(const size_t& num_blocks ,
              const Eigen::Vector<TransformValueType , 3>& dx = Eigen::Vector<TransformValueType,3>(1.0f , 1.0f ,1.0f) ){
                  return Grid<GridLayout<block_nx , block_ny , block_nz> ,TransformValueType>(num_blocks , dx);
              };
    
    


    template<typename GridLayout , typename TransformValueType>
    MUDA_HOST void Grid<GridLayout,TransformValueType>::grid_hash_build(muda::BufferView<uint64_t> offset_key,
                                                   muda::BufferView<int> index)
    {
        muda::ParallelFor().apply(index.size(),[
        index         = index.viewer().name("particle_index"),
        offset_key    = offset_key.viewer().name("particle_morton_key"),
        tablesize     = m_num_blocks,
        keyTable      = key.viewer().name("hash_key_table"),
        valueTable    = value.viewer().name("hash_value_table"),
        block_id2key  = block_id2key.viewer().name("block_id2morton_key"),
        block_counter = block_counter.viewer().name("block_counter"),
        num_particle  = index.size()
            ] __device__
            (int i ) {

                auto global_thread_id = blockDim.x*blockIdx.x + threadIdx.x;
                
                if(global_thread_id >= num_particle){
                    return;
                }  
                auto particle_index = index(global_thread_id);
                uint64_t key = offset_key(particle_index) >> GridLayout::log2_voxel_num;
                uint64_t hashkey = key % tablesize;
                int block_id = 0;

                while(true){

                    auto key_prev = muda::atomic_cas((unsigned long long int*)keyTable.data() + hashkey ,
                                                    0xffffffffffffffff , (unsigned long long int ) key);
                    if(key_prev == 0xffffffffffffffff){
                        
                        block_id = muda::atomic_add(block_counter.data(), 1);
                        
                        valueTable(hashkey) = block_id;
                        block_id2key(block_id) = key;
                        break;
                    }
                    else if (key_prev == key) {
                        break;
                    }
                    else{

                        hashkey = (hashkey+127)%tablesize;
                    }
                }

                return ;
            }).wait();
            

    }

    template<typename GridLayout , typename TransformValueType>
    MUDA_HOST void Grid<GridLayout, TransformValueType>::supplementAdjancentBlocks(){

        muda::ParallelFor().apply(m_num_blocks , [

        tablesize     = m_num_blocks,
        keyTable      = key.viewer().name("hash_key_table"),
        valueTable    = value.viewer().name("hash_value_table"),
        block_id2key  = block_id2key.viewer().name("block_id2morton_key"),
        topology      = topology_cache.viewer().name("topology_cache"),
        num_block     = block_counter.viewer().name("block_counter"),
        block_counter = block_counter.viewer()


        ] __device__ (int thread_id ) mutable {


            auto global_thread_id = thread_id;
            if(global_thread_id >= num_block){return;}
            
            unsigned int block_id = global_thread_id;
            auto key = block_id2key(block_id);
            
            for(int i = -1 ; i < 2 ; i++){
                for(int j = -1 ; j < 2 ; j++){
                    for(int k = -1 ; k < 2 ; k++){

                        int key_x = 0 ,key_y = 0,key_z = 0;
                        decode_morton(key , &key_x , &key_y ,&key_z);
                        if(key_x + i < 0 || key_y + j < 0 || key_z +k < 0){ continue ;}

                        auto key_neibour = packed_add_morton(key, i, j, k);
                        auto hashkey_neibour = key_neibour % tablesize;
                        

                        while(true){
                        auto key_prev = muda::atomic_cas((unsigned long long int*)keyTable.data() + hashkey_neibour ,
                                                    0xffffffffffffffff , (unsigned long long int ) key_neibour);
                        
                    if(key_prev == 0xffffffffffffffff){
                        
                        auto block_id_new = muda::atomic_add(block_counter.data(), 1);
                        
                        valueTable(hashkey_neibour) = block_id_new;
                        block_id2key(block_id_new) = key_neibour;
                        break;
                    }
                    else if (key_prev == key_neibour) {
                        break;
                    }
                    else{

                        hashkey_neibour = (hashkey_neibour+127)%tablesize;
                    }
                }
                    }
                }
            }
            


        }).wait();
    };

    
    template<typename GridLayout , typename TransformValueType>
    MUDA_HOST void Grid<GridLayout, TransformValueType>::Grid_build_topology(){


        muda::ParallelFor().apply(m_num_blocks , [

        tablesize     = m_num_blocks,
        keyTable      = key.viewer().name("hash_key_table"),
        valueTable    = value.viewer().name("hash_value_table"),
        block_id2key  = block_id2key.viewer().name("block_id2morton_key"),
        topology      = topology_cache.viewer().name("topology_cache"),
        num_block     = block_counter_before_adjancent_cache.viewer().name("block_counter"),
        block_counter = block_counter_before_adjancent_cache.viewer()


        ] __device__ (int thread_id ) mutable {

            auto global_thread_id = thread_id;
            if(global_thread_id >= num_block){return;}
            
            unsigned int block_id = global_thread_id;
            auto key = block_id2key(block_id);
            
            for(int i = -1 ; i < 2 ; i++){
                for(int j = -1 ; j < 2 ; j++){
                    for(int k = -1 ; k < 2 ; k++){
                        int key_x =0 ,key_y = 0,key_z = 0;
                        decode_morton(key , &key_x , &key_y ,&key_z);
                        if(key_x + i < 0 || key_y + j < 0 || key_z +k < 0){ continue ;}


                        auto key_neibour = packed_add_morton(key, i, j, k);
                        auto hashkey_neibour = key_neibour % tablesize;


                        while(keyTable(hashkey_neibour) != key_neibour){
                            hashkey_neibour = (hashkey_neibour+127)%tablesize;
                        }
                        topology(27*block_id + encode_linear(i + 1 , j + 1 , k+1  , 3, 3 ,3)) = valueTable(hashkey_neibour);
                    }
                }
            }
            

            

            
            

        }).wait();






    }
}
}