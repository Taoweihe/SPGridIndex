#include <muda/ext/eigen/eigen_dense_cxx20.h>
#include <Eigen/Dense>
#include <Grid.cuh>
#include <muda/ext/field.h>
#include <utils.cuh>

int main(){

        auto  gridIndex = 
        muda::SParseGrid::make_grid(1000);

        muda::DeviceBuffer<int> index(1000);
        muda::DeviceBuffer<uint64_t> offset_key(1000);
        muda::DeviceBuffer<float> density(1000);
        muda::DeviceBuffer<Eigen::Vector3f> position(1000);
        muda::DeviceBuffer<Eigen::Vector3f> velocity(1000);
        muda::DeviceBuffer<Eigen::Matrix3f> C(1000);
        muda::DeviceBuffer<Eigen::Vector3f> force(1000);

        //initialize particle data

        muda::ParallelFor().apply(10000, 
        [
            index = index.viewer(),
            position = position.viewer(),
            offset_key = offset_key.viewer(),
            density = density.viewer()
            
        ] __device__ (int i) mutable{


            index(i) = i;
            position(i) = Eigen::Vector3f(float(i) * 0.01f+10 , float(i) * 0.01f +10 , float(i) * 0.01f +10);
            offset_key(i) = muda::SParseGrid::encode_morton<uint64_t , float>(position(i));
            density(i) = 1.0f;
        });

        gridIndex.build_map(offset_key, index);

        muda::Field field;
        auto & grid = field["grid"];
        float dt = 0.01f;

        auto builder = grid.SoA();

        auto &mass = builder.entry("mass").scalar<float>();
        auto &f    = builder.entry("f").vector3<float>();
        auto &affine_velocity_matrix    = builder.entry("C").matrix3x3<float>();
        auto &vel = builder.entry("velocity").vector3<float>();

        builder.build();

        grid.resize(1000);


        //test simulation step

        //don‘t use any optimization 

        // compute dt

        dt = 0.03f;

        //rebuilt mapping

        gridIndex.rebuild_map(offset_key, index);

        //P2G

        muda::ParallelFor().apply(10000, 
        [
            index = index.viewer(),
            offset_key = offset_key.viewer(),
            density = density.viewer(),
            position = position.viewer(),
            velocity = velocity.viewer(),
            C = C.viewer(),
            force = force.viewer(),
            gridIndex = gridIndex.Viewer(),
            mass = mass.viewer(),
            f = f.viewer(),
            vel = vel.viewer(),
            affine_velocity_matrix = affine_velocity_matrix.viewer()
            
        ] __device__ (int i) mutable{

            auto particle_index = i;
            auto coord = gridIndex.make_coordinate(offset_key(particle_index));

            auto grid_data_index = gridIndex(coord);

            auto particle_density = 
        });



    }   