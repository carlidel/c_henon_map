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

henon_map::henon_map(unsigned int _n_iterations, double _epsilon, double _limit, double _om_x0, double _om_y0) : n_iterations(_n_iterations), epsilon(_epsilon), limit(_limit), omega_x0(_om_x0 * 2 * M_PI), omega_y0(_om_y0 * 2 * M_PI) {}

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

        if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] > limit)
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

// Radial Henon Functor

radial_henon_functor::radial_henon_functor() {}

radial_henon_functor::radial_henon_functor(double _dr, double _epsilon, double _limit, double _om_x0, double _om_y0) : dr(_dr), epsilon(_epsilon), limit(_limit), omega_x0(_om_x0 * 2 * M_PI), omega_y0(_om_y0 * 2 * M_PI) {}

template <typename Tuple>
__host__ __device__ void radial_henon_functor::operator()(Tuple t)
{
    // Tuple t = (ALPHA, THETA1, THETA2, STEP, MAX_ITERATIONS)

    double sum;

    double omega_x;
    double omega_y;

    double sin_alpha, cos_alpha, sin_theta1, cos_theta1, sin_theta2, cos_theta2;

    double cosx, sinx, cosy, siny;
    double temp1, temp2;
    
    double x, y, px, py;

    // We start from last known stable point, increase by one
    thrust::get<3>(t) += 1;

    while (true)
    {
        // Convert to Cartesian
        x = thrust::get<3>(t) * dr * cos(thrust::get<0>(t)) * cos(thrust::get<1>(t));
        px = thrust::get<3>(t) * dr * cos(thrust::get<0>(t)) * sin(thrust::get<1>(t));
        y = thrust::get<3>(t) * dr * sin(thrust::get<0>(t)) * cos(thrust::get<2>(t));
        py = thrust::get<3>(t) * dr * sin(thrust::get<0>(t)) * sin(thrust::get<2>(t));

        // Iterate and see what happens...
        for (unsigned int i = 0; i < thrust::get<4>(t); ++i)
        {
            sum = 0;
            for (int j = 0; j < 7; ++j)
            {
                sum += epsilon_k[j] * cos(Omega_k[j] * i);
            }
            omega_x = omega_x0 * (1 + epsilon * sum);
            omega_y = omega_y0 * (1 + epsilon * sum);

            cosx = cos(omega_x);
            sinx = sin(omega_x);
            cosy = cos(omega_y);
            siny = sin(omega_y);

            temp1 = (px + x * x - y * y);
            temp2 = (py - 2 * x * y);

            px = -sinx * x + cosx * temp1;
            py = -siny * y + cosy * temp2;

            x = cosx * x + sinx * temp1;
            y = cosy * y + siny * temp2;

            if (x * x + y * y + px * px + py * py > limit)
            {
                // Particle lost, decrease by one and return
                thrust::get<3>(t) -= 1;
                return;
            }
        }
        // Particle stable, increase by one and go on!
        thrust::get<3>(t) += 1;
    }
}

// Dummy functor

dummy_functor::dummy_functor() {}

template <typename Tuple>
__host__ __device__ void dummy_functor::operator()(Tuple t)
{
    // Tuple t = (ALPHA, THETA1, THETA2, STEP, MAX_ITERATIONS)
    thrust::get<3>(t) = int(1e7) - thrust::get<4>(t);
}

// Henon Radial

henon_radial::henon_radial()
{
}

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

    x.resize(X.size());
    y.resize(Y.size());
    px.resize(P_X.size());
    py.resize(P_Y.size());
    t.resize(T.size());
    lost.resize(LOST.size());
    
    thrust::copy(X.begin(), X.end(), x.begin());
    thrust::copy(Y.begin(), Y.end(), y.begin());
    thrust::copy(P_X.begin(), P_X.end(), px.begin());
    thrust::copy(P_Y.begin(), P_Y.end(), py.begin());
    thrust::copy(T.begin(), T.end(), t.begin());
    thrust::copy(LOST.begin(), LOST.end(), lost.begin());

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>> tup (x, y, px, py, t, lost);
    return tup;
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

    x.resize(X.size());
    y.resize(Y.size());
    px.resize(P_X.size());
    py.resize(P_Y.size());
    t.resize(T.size());
    lost.resize(LOST.size());

    thrust::copy(X.begin(), X.end(), x.begin());
    thrust::copy(Y.begin(), Y.end(), y.begin());
    thrust::copy(P_X.begin(), P_X.end(), px.begin());
    thrust::copy(P_Y.begin(), P_Y.end(), py.begin());
    thrust::copy(T.begin(), T.end(), t.begin());
    thrust::copy(LOST.begin(), LOST.end(), lost.begin());

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>> tup (x, y, px, py, t, lost);
    return tup;
}

// henon_scan

henon_scan::henon_scan() {}

henon_scan::henon_scan(std::vector<double> _x0, std::vector<double> _y0, std::vector<double> _px0, std::vector<double> _py0, double _epsilon) : x0(_x0), y0(_y0), px0(_px0), py0(_py0), epsilon(_epsilon)
{
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
    cudaSetDevice(1);
#endif

    X = x0;
    Y = y0;
    X_0 = x0;
    Y_0 = y0;
    P_X = px0;
    P_Y = py0;
    P_X_0 = px0;
    P_Y_0 = py0; 

    T.resize(x0.size());
    thrust::fill(T.begin(), T.end(), 0);

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

    thrust::fill(T.begin(), T.end(), 0);
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

    std::tuple<std::vector<double>, std::vector<double>, std::vector<unsigned int>> tup (x0, y0, times);
    return tup;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>> henon_scan::get_data()
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> px;
    std::vector<double> py;
    std::vector<unsigned int> t;
    std::vector<bool> lost;

    x.resize(X.size());
    y.resize(Y.size());
    px.resize(P_X.size());
    py.resize(P_Y.size());
    t.resize(T.size());
    lost.resize(LOST.size());

    thrust::copy(X.begin(), X.end(), x.begin());
    thrust::copy(Y.begin(), Y.end(), y.begin());
    thrust::copy(P_X.begin(), P_X.end(), px.begin());
    thrust::copy(P_Y.begin(), P_Y.end(), py.begin());
    thrust::copy(T.begin(), T.end(), t.begin());
    thrust::copy(LOST.begin(), LOST.end(), lost.begin());

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<unsigned int>, std::vector<bool>> tup (x, y, px, py, t, lost);
    return tup;
}

// henon_track

henon_track::henon_track() {}

henon_track::henon_track(double _x0, double _y0, double _px0, double _py0, double _epsilon) : x0(_x0), y0(_y0), px0(_px0), py0(_py0), epsilon(_epsilon)
{
    T = 0;
    x.push_back(x0);
    y.push_back(y0);
    px.push_back(px0);
    py.push_back(py0);
}

henon_track::~henon_track() {}

void henon_track::reset()
{
    T = 0;
    x.clear();
    y.clear();
    px.clear();
    py.clear();
    x.push_back(x0);
    y.push_back(y0);
    px.push_back(px0);
    py.push_back(py0);
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> henon_track::compute(unsigned int iterations)
{
    henon_map temp_hm(1, epsilon);

    thrust::host_vector<double> X, P_X, Y, P_Y;
    thrust::host_vector<unsigned int> TT;
    thrust::host_vector<bool> LOST;

    x.reserve(x.size() + iterations);
    px.reserve(x.size() + iterations);
    y.reserve(x.size() + iterations);
    py.reserve(x.size() + iterations);

    X.push_back(x[0]);
    Y.push_back(y[0]);
    P_X.push_back(px[0]);
    P_Y.push_back(py[0]);
    TT.push_back(0);
    LOST.push_back(false);

    for (unsigned int i = 0; i < iterations; i++)
    {
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(X.begin(), P_X.begin(), Y.begin(), P_Y.begin(), TT.begin(), LOST.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(X.end(), P_X.end(), Y.end(), P_Y.end(), TT.end(), LOST.end())),
            temp_hm);
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
        cudaDeviceSynchronize();
#endif
        x.push_back(X[0]);
        px.push_back(P_X[0]);
        y.push_back(Y[0]);
        py.push_back(P_Y[0]);

        if (LOST[0])
            break;
    }

    X.clear();
    Y.clear();
    P_X.clear();
    P_Y.clear();
    TT.clear();
    LOST.clear();

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> tup (x, y, px, py);
    return tup;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> henon_track::get_data()
{
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> tup (x, y, px, py);
    return tup;
}

// radial scan

radial_scan::radial_scan() {}

radial_scan::radial_scan(double _dr, std::vector<double> _alpha, std::vector<double> _theta1, std::vector<double> _theta2, double _epsilon, double _omx, double _omy) : dr(_dr), alpha(_alpha), theta1(_theta1), theta2(_theta2), epsilon(_epsilon), omx(_omx), omy(_omy)
{
    ALPHA = alpha;
    THETA1 = theta1;
    THETA2 = theta2;

    STEP.resize(ALPHA.size());
    thrust::fill(STEP.begin(), STEP.end(), 0);

    MAX_ITERATIONS.resize(ALPHA.size());

    radiuses.resize(alpha.size());
}

void radial_scan::reset()
{
    thrust::fill(STEP.begin(), STEP.end(), 0);
    
    for (auto &x : radiuses)
        x.clear();
}

std::vector<std::vector<double>> radial_scan::compute(std::vector<unsigned int> time_samples)
{
    radial_henon_functor functor(dr, epsilon, 100.0, omx, omy);
    for (auto t : time_samples)
    {
        thrust::fill(MAX_ITERATIONS.begin(), MAX_ITERATIONS.end(), t);
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(
                ALPHA.begin(),
                THETA1.begin(),
                THETA2.begin(),
                STEP.begin(),
                MAX_ITERATIONS.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                ALPHA.end(),
                THETA1.end(),
                THETA2.end(),
                STEP.end(),
                MAX_ITERATIONS.end())),
            functor);
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
        cudaDeviceSynchronize();
#endif
        for(unsigned int i = 0; i < alpha.size(); i++)
        {
            radiuses[i].push_back(STEP[i] * dr);
        }
    }
    return radiuses;
}

std::vector<std::vector<double>> radial_scan::dummy_compute(std::vector<unsigned int> time_samples)
{
    dummy_functor functor;
    for (auto t : time_samples)
    {
        thrust::fill(MAX_ITERATIONS.begin(), MAX_ITERATIONS.end(), t);
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(
                ALPHA.begin(),
                THETA1.begin(),
                THETA2.begin(),
                STEP.begin(),
                MAX_ITERATIONS.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                ALPHA.end(),
                THETA1.end(),
                THETA2.end(),
                STEP.end(),
                MAX_ITERATIONS.end())),
            functor);
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_OMP
        cudaDeviceSynchronize();
#endif
        for (unsigned int i = 0; i < alpha.size(); i++)
        {
            radiuses[i].push_back(STEP[i] * dr);
        }
    }
    return radiuses;
}

std::vector<std::vector<double>> radial_scan::get_data()
{
    return radiuses;
}
