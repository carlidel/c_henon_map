#include "c_henon_map.h"

// x0 functor

radial::radial(unsigned int _nx, double _dtheta, double _dx) : nx(_nx), dtheta(_dtheta), dx(_dx) {}

template <typename Tuple> __host__ __device__ void radial::operator()(Tuple t)
{
    double theta_I = (thrust::get<2>(t)) / nx;
    double x_I = ((int)thrust::get<2>(t)) % nx;
    thrust::get<0>(t) = dx * (x_I + 1) * cos(dtheta * theta_I);
    thrust::get<1>(t) = dx * (x_I + 1) * sin(dtheta * theta_I);
}


grid::grid(unsigned int _xlen, unsigned int _ylen, double _dx, double _dy) : xlen(_xlen), ylen(_ylen), dx(_dx), dy(_dy) {}

template <typename Tuple> __host__ __device__ void grid::operator()(Tuple t)
{
    double x_I = (thrust::get<2>(t)) / xlen;
    double y_I = (thrust::get<2>(t)) % xlen;
    thrust::get<0>(t) = dx * x_I;
    thrust::get<1>(t) = dy * y_I;
}

// Henon functor

henon_map::henon_map() {}

henon_map::henon_map(unsigned int _n_iterations, double _epsilon, double _limit) : n_iterations(_n_iterations), epsilon(_epsilon), limit(_limit) {}

template <typename Tuple> __host__ __device__ void henon_map::operator()(Tuple t)
{
    double sum = 0;
    double v[4];

    double omega_x;
    double omega_y;
    double cosx;
    double sinx;
    double cosy;
    double siny;

    double temp1;
    double temp2;

    if (thrust::get<0>(t) == 0.0 && thrust::get<1>(t) == 0.0 && thrust::get<2>(t) == 0.0 && thrust::get<3>(t) == 0.0)
    {
        thrust::get<4>(t) += n_iterations;
        return;
    }

    if (thrust::get<5>(t))
    {
        return;
    }

    for (unsigned int i = 0; i < n_iterations; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            sum += epsilon_k[j] * cos(Omega_k[j] * thrust::get<4>(t));
        }
        omega_x = omega_x0 * (1 + epsilon * sum);
        omega_y = omega_y0 * (1 + epsilon * sum);

        cosx = cos(omega_x);
        sinx = sin(omega_x);
        cosy = cos(omega_y);
        siny = sin(omega_y);

        temp1 = (thrust::get<1>(t) + thrust::get<0>(t) * thrust::get<0>(t) - thrust::get<2>(t) * thrust::get<2>(t));
        temp2 = (thrust::get<3>(t) - 2 * thrust::get<0>(t) * thrust::get<2>(t));

        v[0] = cosx * thrust::get<0>(t) + sinx * temp1;
        v[1] = -sinx * thrust::get<0>(t) + cosx * temp1;
        v[2] = cosy * thrust::get<2>(t) + siny * temp2;
        v[3] = -siny * thrust::get<2>(t) + cosy * temp2;

        if (v[0] * v[0] + v[2] * v[2] > limit)
        {
            thrust::get<5>(t) = true;
            return;
        }

        thrust::get<0>(t) = v[0];
        thrust::get<1>(t) = v[1];
        thrust::get<2>(t) = v[2];
        thrust::get<3>(t) = v[3];

        thrust::get<4>(t) += 1;
    }
    return;
}

henon_radial::henon_radial() {}
    
henon_radial::henon_radial(unsigned int _n_theta, unsigned int _n_steps, double _epsilon) : n_theta(_n_theta), n_steps(_n_steps),  epsilon(_epsilon)
{
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
    cudaSetDevice(1);
#endif

    d_theta = M_PI / (2.0 * (n_theta - 1));
    d_x = 1.0 / (n_steps);

    X.resize(n_theta * n_steps);
    Y.resize(n_theta * n_steps);
    X_0.resize(n_theta * n_steps);
    Y_0.resize(n_theta * n_steps);

    thrust::fill(X.begin(), X.end(), 0.0);
    thrust::fill(Y.begin(), Y.end(), 0.0);
    thrust::fill(X_0.begin(), X_0.end(), 0.0);
    thrust::fill(Y_0.begin(), Y_0.end(), 0.0);

    P_X.resize(n_theta * n_steps);
    P_Y.resize(n_theta * n_steps);
    P_X_0.resize(n_theta * n_steps);
    P_Y_0.resize(n_theta * n_steps);

    thrust::fill(P_X.begin(), P_X.end(), 0.0);
    thrust::fill(P_Y.begin(), P_Y.end(), 0.0);
    thrust::fill(P_X_0.begin(), P_X_0.end(), 0.0);
    thrust::fill(P_Y_0.begin(), P_Y_0.end(), 0.0);

    T.resize(n_theta * n_steps);
    thrust::fill(T.begin(), T.end(), 0.0);

    INDEX.resize(n_theta * n_steps);
    thrust::sequence(INDEX.begin(), INDEX.end());

    LOST.resize(n_theta * n_steps);
    thrust::fill(LOST.begin(), LOST.end(), false);

    thrust::for_each
    (
        thrust::make_zip_iterator(thrust::make_tuple(X_0.begin(), Y_0.begin(), INDEX.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(X_0.end(), Y_0.end(), INDEX.end())),
        radial(n_steps, d_theta, d_x)
    );

    X = X_0;
    Y = Y_0;
}

henon_radial::~henon_radial() {}

void henon_radial::reset()
{
    X = X_0;
    Y = Y_0;
    P_X = P_X_0;
    P_Y = P_Y_0;

    thrust::fill(T.begin(), T.end(), 0.0);
    thrust::fill(LOST.begin(), LOST.end(), false);
}

std::vector<unsigned int> henon_radial::compute(unsigned int kernel_iterations, unsigned int block_iterations)
{
    for (unsigned int i = 0; i < block_iterations; i++)
    {
        henon_map temp_hm(kernel_iterations, epsilon);
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(X.begin(), P_X.begin(), Y.begin(), P_Y.begin(), T.begin(), LOST.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(X.end(), P_X.end(), Y.end(), P_Y.end(), T.end(), LOST.end())),
            temp_hm);
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
        cudaDeviceSynchronize();
#endif
    }
    std::vector<unsigned int> times(n_theta * n_steps);
    thrust::copy(T.begin(), T.end(), times.begin());
    return times;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>> henon_radial::get_data()
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> px;
    std::vector<double> py;
    std::vector<unsigned int> t;
    std::vector<bool> lost;

    thrust::copy(X.begin(), X.end(), x.begin());
    thrust::copy(Y.begin(), Y.end(), y.begin());
    thrust::copy(P_X.begin(), P_X.end(), px.begin());
    thrust::copy(P_Y.begin(), P_Y.end(), py.begin());
    thrust::copy(T.begin(), T.end(), t.begin());
    thrust::copy(LOST.begin(), LOST.end(), lost.begin());

    return std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>>(x, y, px, py, t, lost);
}

// henon_grid

henon_grid::henon_grid() {}

henon_grid::henon_grid(unsigned int _n_x, unsigned int _n_y, double _epsilon) : n_x(_n_x), n_y(_n_y), epsilon(_epsilon) 
{
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
    cudaSetDevice(1);
#endif
    dx = 1.0 / (n_x - 1);
    dy = 1.0 / (n_y - 1);

    X.resize(n_x * n_y);
    Y.resize(n_x * n_y);
    X_0.resize(n_x * n_y);
    Y_0.resize(n_x * n_y);

    thrust::fill(X.begin(), X.end(), 0.0);
    thrust::fill(Y.begin(), Y.end(), 0.0);
    thrust::fill(X_0.begin(), X_0.end(), 0.0);
    thrust::fill(Y_0.begin(), Y_0.end(), 0.0);

    P_X.resize(n_x * n_y);
    P_Y.resize(n_x * n_y);
    P_X_0.resize(n_x * n_y);
    P_Y_0.resize(n_x * n_y);

    thrust::fill(P_X.begin(), P_X.end(), 0.0);
    thrust::fill(P_Y.begin(), P_Y.end(), 0.0);
    thrust::fill(P_X_0.begin(), P_X_0.end(), 0.0);
    thrust::fill(P_Y_0.begin(), P_Y_0.end(), 0.0);

    T.resize(n_x * n_y);
    thrust::fill(T.begin(), T.end(), 0.0);

    INDEX.resize(n_x * n_y);
    thrust::sequence(INDEX.begin(), INDEX.end());

    LOST.resize(n_x * n_y);
    thrust::fill(LOST.begin(), LOST.end(), false);

    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(X_0.begin(), Y_0.begin(), INDEX.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(X_0.end(), Y_0.end(), INDEX.end())),
        grid(n_x, n_y, dx, dy));

    X = X_0;
    Y = Y_0;
}

henon_grid::~henon_grid() {}

void henon_grid::reset()
{
    X = X_0;
    Y = Y_0;
    P_X = P_X_0;
    P_Y = P_Y_0;

    thrust::fill(T.begin(), T.end(), 0.0);
    thrust::fill(LOST.begin(), LOST.end(), false);
}

std::vector<unsigned int> henon_grid::compute(unsigned int kernel_iterations, unsigned int block_iterations)
{
    for (unsigned int i = 0; i < block_iterations; i++)
    {
        henon_map temp_hm(kernel_iterations, epsilon);
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(X.begin(), P_X.begin(), Y.begin(), P_Y.begin(), T.begin(), LOST.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(X.end(), P_X.end(), Y.end(), P_Y.end(), T.end(), LOST.end())),
            temp_hm);
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
        cudaDeviceSynchronize();
#endif
    }
    std::vector<unsigned int> times(n_x * n_y);
    thrust::copy(T.begin(), T.end(), times.begin());
    return times;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>> henon_grid::get_data()
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> px;
    std::vector<double> py;
    std::vector<unsigned int> t;
    std::vector<bool> lost;

    thrust::copy(X.begin(), X.end(), x.begin());
    thrust::copy(Y.begin(), Y.end(), y.begin());
    thrust::copy(P_X.begin(), P_X.end(), px.begin());
    thrust::copy(P_Y.begin(), P_Y.end(), py.begin());
    thrust::copy(T.begin(), T.end(), t.begin());
    thrust::copy(LOST.begin(), LOST.end(), lost.begin());

    return std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>>(x, y, px, py, t, lost);
}

// henon_scan

henon_scan::henon_scan() {}

henon_scan::henon_scan(std::vector<double> _x0, std::vector<double> _y0, std::vector<double> _px0, std::vector<double> _py0, double _epsilon) : x0(_x0), y0(_y0), px0(_px0), py0(_py0)
{
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
    cudaSetDevice(1);
#endif

    X = x0;
    Y = y0;
    X_0 = x0;
    Y_0 = y0;

    P_X.resize(x0.size());
    P_Y.resize(x0.size());
    P_X_0.resize(x0.size());
    P_Y_0.resize(x0.size());

    thrust::fill(P_X.begin(), P_X.end(), 0.0);
    thrust::fill(P_Y.begin(), P_Y.end(), 0.0);
    thrust::fill(P_X_0.begin(), P_X_0.end(), 0.0);
    thrust::fill(P_Y_0.begin(), P_Y_0.end(), 0.0);

    T.resize(x0.size());
    thrust::fill(T.begin(), T.end(), 0.0);

    LOST.resize(x0.size());
    thrust::fill(LOST.begin(), LOST.end(), false);
}

henon_scan::~henon_scan() {}

void henon_scan::reset()
{
    X = X_0;
    Y = Y_0;
    P_X = P_X_0;
    P_Y = P_Y_0;

    thrust::fill(T.begin(), T.end(), 0.0);
    thrust::fill(LOST.begin(), LOST.end(), false);
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<unsigned int>> henon_scan::compute(unsigned int kernel_iterations, unsigned int block_iterations)
{
    for (unsigned int i = 0; i < block_iterations; i++)
    {
        henon_map temp_hm(kernel_iterations, epsilon);
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(X.begin(), P_X.begin(), Y.begin(), P_Y.begin(), T.begin(), LOST.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(X.end(), P_X.end(), Y.end(), P_Y.end(), T.end(), LOST.end())),
            temp_hm);
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
        cudaDeviceSynchronize();
#endif
    }
    std::vector<unsigned int> times(X.size());
    thrust::copy(T.begin(), T.end(), times.begin());

    return std::tuple<std::vector<double>, std::vector<double>, std::vector<unsigned int>> (x0, y0, times);
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>> henon_scan::get_data()
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> px;
    std::vector<double> py;
    std::vector<unsigned int> t;
    std::vector<bool> lost;

    thrust::copy(X.begin(), X.end(), x.begin());
    thrust::copy(Y.begin(), Y.end(), y.begin());
    thrust::copy(P_X.begin(), P_X.end(), px.begin());
    thrust::copy(P_Y.begin(), P_Y.end(), py.begin());
    thrust::copy(T.begin(), T.end(), t.begin());
    thrust::copy(LOST.begin(), LOST.end(), lost.begin());

    return std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>> (x, y, px, py, t, lost);
}
