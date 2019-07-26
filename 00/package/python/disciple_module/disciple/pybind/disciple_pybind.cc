#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "plus.h"

PYBIND11_MODULE(disciple_pybind, m) {
    m.def("plus", [](int x, int y) {
        return plus(x, y);
    }, "simple");
    m.attr("__version__") = "dev";
}
