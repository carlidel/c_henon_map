#ifndef C_HENON_MAP_H
#define C_HENON_MAP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <tuple>

#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
#include <cuda.h>
#endif

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// x0 functor

struct radial
{
    unsigned int nx;
    double dtheta;
    double dx;

    radial(double _nx, double _dtheta, double _dx);

    template <typename Tuple> __host__ __device__ void operator()(Tuple t);
};

struct grid
{
    unsigned int xlen;
    unsigned int ylen;
    double dx;
    double dy;

    grid(unsigned int _xlen, unsigned int _ylen, double _dx, double _dy);

    template <typename Tuple> __host__ __device__ void operator()(Tuple t);
};

// Henon functor

struct henon_map
{
    const double epsilon_k[7] =
    {
        1.000e-4,
        0.218e-4,
        0.708e-4,
        0.254e-4,
        0.100e-4,
        0.078e-4,
        0.218e-4
    };
    const double Omega_k[7] =
    {
        1 * (2 * M_PI / 868.12),
        2 * (2 * M_PI / 868.12),
        3 * (2 * M_PI / 868.12),
        6 * (2 * M_PI / 868.12),
        7 * (2 * M_PI / 868.12),
        10 * (2 * M_PI / 868.12),
        12 * (2 * M_PI / 868.12)
    };
    const double omega_x0 = 0.168 * 2 * M_PI;
    const double omega_y0 = 0.201 * 2 * M_PI;

    double epsilon;
    double limit = 10000.0;

    unsigned int n_iterations;

    henon_map();
    henon_map(unsigned int _n_iterations, double _epsilon = 0.0, double _limit = 100000.0);

    template <typename Tuple> __host__ __device__ void operator()(Tuple t);
};

class henon_radial
{
public:
    unsigned int n_theta;
    unsigned int n_steps;
    double d_theta;
    double d_x;

    double epsilon;

    thrust::device_vector<double> X, Y, P_X, P_Y;
    thrust::device_vector<double> X_0, Y_0, P_X_0, P_Y_0;
    thrust::device_vector<unsigned int> T, INDEX;

    henon_radial();
    henon_radial(unsigned int _n_theta, unsigned int _n_steps, double _epsilon);
    ~henon_radial();

    void reset();
    std::vector<unsigned int> compute(unsigned int kernel_iterations, unsigned int block_iterations=1);
};

class henon_grid
{
public:
    unsigned int n_x;
    unsigned int n_y;
    double dx;
    double dy;

    double epsilon;

    thrust::device_vector<double> X, Y, P_X, P_Y;
    thrust::device_vector<double> X_0, Y_0, P_X_0, P_Y_0;
    thrust::device_vector<unsigned int> T, INDEX;

    henon_grid();
    henon_grid(unsigned int _n_x, unsigned int _n_y, double _epsilon);
    ~henon_grid();

    void reset();
    std::vector<unsigned int> compute(unsigned int kernel_iterations, unsigned int block_iterations=1);
};

class henon_scan
{
public:
    std::vector<double> x0, y0, px0, py0;

    double epsilon;

    thrust::device_vector<double> X, Y, P_X, P_Y;
    thrust::device_vector<double> X_0, Y_0, P_X_0, P_Y_0;
    thrust::device_vector<unsigned int> T;

    henon_scan();
    henon_scan(std::vector<double> _x0, std::vector<double> _y0, std::vector<double> _px0, std::vector<double> _py0, double _epsilon);
    ~henon_scan();

    void reset();
    std::tuple<std::vector<double>, std::vector<double>, std::vector<unsigned int>> compute(unsigned int kernel_iterations, unsigned int block_iterations = 1);
};

#endif //C_HENON_MAP_H