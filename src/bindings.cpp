#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

# include "c_henon_map.h"

// PYTHON BINDING
PYBIND11_MODULE(c_henon_map, m)
{
    py::class_<henon_radial>(m, "henon_radial")
        .def(py::init<unsigned int, unsigned int, double>())
        .def("reset", &henon_radial::reset)
        .def("compute", &henon_radial::compute)
        .def("get_data", &henon_radial::get_data);

    py::class_<henon_grid>(m, "henon_grid")
        .def(py::init<unsigned int, unsigned int, double>())
        .def("reset", &henon_grid::reset)
        .def("compute", &henon_grid::compute)
        .def("get_data", &henon_grid::get_data);

    py::class_<henon_scan>(m, "henon_scan")
        .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, double>())
        .def("reset", &henon_scan::reset)
        .def("compute", &henon_scan::compute)
        .def("get_data", &henon_scan::get_data);
}