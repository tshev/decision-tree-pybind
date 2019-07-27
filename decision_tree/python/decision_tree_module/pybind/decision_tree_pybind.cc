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
        .def("fit", [](decision_tree_csl& cls, const py::array_t<double>& x, const py::array_t<int>& target) {
            if (x.ndim() != 2) {
                throw std::runtime_error("x should have 2 dimensions");
            }

            size_t rows = x.shape(0);
            size_t cols  = x.shape(1);

            const double* x_data = (const double*)x.data();
            matrix <double> m(x_data, rows, cols);

            const int* y_data = (const int*)target.data();
            std::vector<int> y(y_data, y_data + rows);

            decision_tree_csl::decision_tree_type dt_new(cls.random_state, cls.max_unique_values_per_feature, m.begin(), m.end(), y.begin(), y.end());
            cls.dt = std::move(dt_new);
        })
        .def("predict", [](decision_tree_csl& cls, const py::array_t<double>& x) {
            if (x.ndim() != 2) {
                throw std::runtime_error("x should have 2 dimensions");
            }
            size_t rows = x.shape(0);
            size_t cols = x.shape(1);
            matrix<double> data_col_major((const double*)x.data(), rows, cols);

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
