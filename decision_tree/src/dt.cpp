#include "decision_tree.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        if (argc == 0) {
            std::cerr << "oops" << std::endl;
            std::exit(1);
        }
        std::cerr << argv[0] << " PATH_TO_TRAIN_SAMPLE PATH_TO_TEST_SAMPLE" << std::endl;
        std::exit(2);
    }

    std::fstream istream_xy_train(argv[1]);
    matrix<double> x_train;
    istream_xy_train >> x_train;
    std::vector<int> y_train;
    istream_xy_train >> y_train;

    assert(x_train.rows() != 0 && x_train.cols() != 0 && y_train.size() == x_train.rows());

    size_t seed = 10;
    size_t max_unique_values_per_feature = 100;
    
    decision_tree<double, gini<double>> dt2(seed, max_unique_values_per_feature, x_train.begin(), x_train.end(), y_train.begin(), y_train.end());
    auto dt = std::move(dt2);

    std::vector<int> y_predicted(y_train.size());

    std::fstream istream_xy_test(argv[2]);
    matrix<double> x_test;
    istream_xy_test >> x_test;
    std::vector<int> y_test;
    istream_xy_test >> y_test;

    dt(x_test.begin(), x_test.end(), y_predicted.begin());

    std::cout << "precision=" << precision(y_predicted.begin(), y_predicted.end(), y_test.begin(), 0.0) << std::endl;;
}
