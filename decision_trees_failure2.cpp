#include <vector>
#include <random>
#include <tuple>
#include <algorithm>
#include <iostream>

template<typename T>
class matrix {
    std::vector<T> values;
    size_t rows_;
    size_t cols_;
public:
    matrix(size_t rows, size_t cols) : values(rows * cols), rows_(rows), cols_(cols) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T& operator()(size_t i, size_t j) {
        return values[i * cols_ + j];
    }

    const T& operator()(size_t i, size_t j) const {
        return values[i * cols_ + j];
    }

    struct iterator2 {
        T* values;
        size_t cols;

        friend
        inline
        bool operator==(const iterator2& x, const iterator2& y) {
            return x.values == y.values && x.cols == y.cols;
        }

        friend
        inline
        bool operator!=(const iterator2& x, const iterator2& y) {
            return !(x == y);
        }
       
        iterator2& operator++() {
            values += cols;
            return *this;
        }

        T& operator*() {
            return *values;
        }

        const T& operator*() const {
            return *values;
        }

        T* begin() {
            return values;
        }

        T* end() {
            return values + cols;
        }

        friend
        inline
        size_t operator-(const iterator2& x, const iterator2& y) {
            return (x.values - y.values) / x.cols;
        }
    };


    iterator2 begin() {
        T* d = &values[0];
        return iterator2{d, rows_};
    }

    iterator2 end() {
        T* d = &values[0];
        return iterator2{d + values.size(), rows_};
    }
};

struct gini {

    template<typename It, typename B>
    double operator()(It f, It l, B& b) {
        double n = l - f;
        if (n == 0.0) { return 0.0; }
        std::for_each(f, l, [&b](const auto& x) {
            ++b[std::get<1>(x)];
        });
        double result = 1.0 - std::accumulate(std::begin(b), std::end(b), 0.0, [n](double r, size_t freq) {
            double x = freq / n;
            return r + x * x; 
        });
        return result;
    }
};


template<typename Criterion>
class decision_tree {
    Criterion criterion;
    std::mt19937 g;

    struct node_t {
        size_t feature_id;
        double value;
        node_t* left;
        node_t* right;
    };

public:
    template<typename It0, typename It1>
    decision_tree(size_t seed, size_t tests_n, It0 f0, It0 l0, It1 f1, It1 l1) : g(seed) {
        build_tree(tests_n, f0, l0, f1, l1);
    }


    template<typename It0, typename It1>
    void build_tree(size_t tests_n, It0 f0, It0 l0, It1 f1, It1 l1) {
        // Initial thoughts: column-based matrix should be good for performance
        // Problem : partition by rows.
        //
        size_t rows_n = f0.end() - f0.begin();
        double rows_nf = rows_n;
        size_t cols_n = l0 - f0;
        std::vector<size_t> frequencies(cols_n);

        std::vector<std::tuple<double, size_t>> feature(rows_n);
        std::vector<double> unique_values(rows_n);
        size_t max_n = 100;

        std::vector<std::pair<double, size_t>> pivots;
        while (true) {
            pivots.resize(0);
            size_t column_index = 0;
            for (auto it = f0; it != l0; ++it, ++column_index) {
                size_t i = 0;
                std::transform(it.begin(), it.end(), f1, std::begin(feature), [&i](auto x, auto y) { return std::make_tuple(x, y); });
                std::sort(std::begin(feature), std::end(feature), [](const auto& x, const auto& y) {
                    return std::get<0>(x) < std::get<0>(y);
                });

                std::copy(it.begin(), it.end(), std::begin(unique_values));
                auto last = std::unique(std::begin(unique_values), std::end(unique_values));
                std::shuffle(std::begin(unique_values), last, g);

                auto feature_first = std::begin(feature);
                auto feature_last = std::end(feature);
                double pivot = std::accumulate(std::begin(unique_values), last, std::numeric_limits<double>::max(), [&frequencies, feature_first, feature_last, this](double r, const auto& pivot) {
                    auto feature_middle = std::lower_bound(feature_first, feature_last, pivot, [](const auto& l, const auto& x) {
                        return std::get<0>(l) < x;
                    });
                        
                    double count = double(feature_last - feature_first);
                    std::fill(std::begin(frequencies), std::end(frequencies), 0);
                    auto criterion_left = criterion(feature_first, feature_middle, frequencies);
                    std::fill(std::begin(frequencies), std::end(frequencies), 0);
                    auto criterion_right = criterion(feature_middle, feature_last, frequencies);
                    auto criterion_value = double(feature_middle - feature_first) / count * criterion_left + double(feature_last - feature_middle) / count * criterion_right;
                    return std::min(r, criterion_value);
                });
                std::cout << "P = " << pivot << std::endl; 
                pivots.emplace_back(pivot, column_index);
            }
            auto min_e = std::min_element(std::begin(pivots), std::end(pivots));
            break;
        }
    }
};

int main() {
    matrix<double> m(4, 2);
    m(0, 0) = 1.0; m(0, 1) = 1.0;
    m(1, 0) = 2.0; m(1, 1) = 3.0;
    m(2, 0) = 4.0; m(2, 1) = 5.0;
    m(2, 0) = 4.0; m(2, 1) = 5.0;
    std::vector<int> y = {0, 1, 1};
    size_t tests_n = 64;
    decision_tree<gini> dt(11, tests_n, m.begin(), m.end(), y.begin(), y.end());

}
