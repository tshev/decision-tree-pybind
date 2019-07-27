#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "decision_tree.h"
using namespace pybind11::literals;
namespace py = pybind11;

struct decision_tree_csl  {
    typedef double value_type;
    typedef decision_tree<value_type, gini<value_type>> decision_tree_type;
    decision_tree_type dt;
    size_t random_state;
    size_t max_unique_values_per_feature;

    decision_tree_csl(py::kwargs kw) {
        if (!kw) { return; }
        if (kw.contains("random_state")) {
            random_state = kw["random_state"].cast<size_t>();
        } else {
            random_state = 0;
        }
        if (kw.contains("max_unique_values_per_feature")) {
            max_unique_values_per_feature = kw["max_unique_values_per_feature"].cast<size_t>();
        } else {
            max_unique_values_per_feature = 1000;
        }
    }

    decision_tree_csl(size_t random_state, size_t max_unique_values_per_feature) : random_state(random_state), max_unique_values_per_feature(max_unique_values_per_feature) {}
};


PYBIND11_MODULE(decision_tree_pybind, m) {
    // TODO check strides of py::array_t
    py::class_<decision_tree_csl>(m, "DecisionTree")
        .def(py::init<size_t, size_t>())
        .def(py::init<py::kwargs>())
        .def("fit", [](decision_tree_csl& cls, py::array_t<double> data, py::array_t<int> target) {
            py::buffer_info xbuff = data.request();
            size_t rows = xbuff.shape[0];
            size_t cols  = xbuff.shape[1];

            double* x_data = (double*)xbuff.ptr;
            iterator2<double> first0(x_data, rows);
            iterator2<double> last0(x_data + rows * cols, rows);

            auto ybuff = target.request();
            int* y_data = (int*)ybuff.ptr;
            matrix <double> m((const double*)x_data, rows, cols);
            decision_tree_csl::decision_tree_type dt_new(cls.random_state, cls.max_unique_values_per_feature, m.begin(), m.end(), y_data, y_data + rows);
            cls.dt = std::move(dt_new);
        })
        .def("predict", [](decision_tree_csl& cls, py::array_t<double> x) {
            auto buff = x.request();
            if (buff.ndim != 2) {
                throw std::runtime_error("X should have 2 dimensions");
            }
            size_t rows = buff.shape[0];
            size_t cols = buff.shape[1];
            matrix<double> data_col_major((const double*)buff.ptr, rows, cols);

            py::array_t<int> predictions(rows);
            cls.dt(data_col_major.begin(), data_col_major.end(), (int*)predictions.request().ptr);
            return predictions;
            // Old version:
            // std::vector<int> predictions(rows);
            // cls.dt(data_col_major.begin(), data_col_major.end(), std::begin(predictions));
        })
        ;
    m.attr("__version__") = "dev";
}
