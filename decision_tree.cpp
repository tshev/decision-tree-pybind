// Quick and dirty cross-platform C++14/C++17 implementation for Decision Trees c
#include <vector>
#include <fstream>
#include <cassert>
#include <random>
#include <tuple>
#include <algorithm>
#include <iostream>

template<typename I>
inline
I& operator>>(I&& cin, std::vector<int>& x) {
    size_t n;
    cin >> n;
    x.resize(n);
    for (size_t i = 0; i < n; ++i) {
        cin >> x[i];
    }
    return cin;
}


template<typename I>
inline
I& operator<<(I&& cout, const std::vector<int>& x) {
    cout << x.size() << std::endl;;
    for (auto a : x) {
        cout << a << " ";
    }
    return cout;
}


template<typename T>
class matrix {
    std::vector<T> values;
    size_t rows_;
    size_t cols_;
public:
    matrix() {}
    matrix(size_t rows, size_t cols) : values(rows * cols), rows_(rows), cols_(cols) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    T& operator()(size_t i, size_t j) {
        return values[j * rows_ + i];
    }

    const T& operator()(size_t i, size_t j) const {
        return values[j * rows_ + i];
    }

    void resize(size_t rows, size_t cols) {
        values.resize(rows * cols);
        rows_ = rows;
        cols_ = cols;
    }

    template<typename O>
    friend
    inline
    O& operator<<(O& o, const matrix& m) {

        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                o << m(i, j) << " ";
            }
            o << std::endl;
        }
        return o;
    }
 
    template<typename O>
    friend
    inline
    O& operator>>(O& o, matrix& m) {
        size_t rows;
        size_t cols;
        o >> rows >> cols;
        m.resize(rows, cols);
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                o >> m(i, j);
            }
        }
        return o;
    }
   

    struct iterator2 {
        T* values;
        size_t rows;

        friend
        inline
        bool operator==(const iterator2& x, const iterator2& y) {
            return x.values == y.values && x.rows == y.rows;
        }

        friend
        inline
        bool operator!=(const iterator2& x, const iterator2& y) {
            return !(x == y);
        }
       
        iterator2& operator++() {
            values += rows;
            return *this;
        }

        T& operator*() { // oops (wrong)
            return *values;
        }

        const T& operator*() const {  // oops (wrong)
            return *values;
        }

        T* begin() {
            return values; // probably, dereferencing should return
        }

        T* end() {
            return values + rows;
        }

        friend
        inline
        size_t operator-(const iterator2& x, const iterator2& y) {
            return (x.values - y.values) / x.rows;
        }

        friend
        inline
        iterator2 operator+(const iterator2& x, size_t y) {
            return iterator2{x.values + x.rows * y, x.rows};
        }

        friend
        inline
        iterator2 operator+(size_t y, const iterator2& x) {
            return iterator2{x.values + x.rows * y, x.rows};
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

template<typename T>
struct gini {
    typedef T value_type;

    template<typename It, typename B>
    value_type operator()(It f, It l, B& b) {
        value_type n = l - f;
        if (n == 0.0) { return 0.0; } // ooops, don't like it
        std::for_each(f, l, [&b](const auto& x) {
            ++b[std::get<1>(x)];
        });
        value_type result = 1.0 - std::accumulate(std::begin(b), std::end(b), 0.0, [n](value_type r, size_t freq) {
            value_type x = freq / n;
            return r + x * x; 
        });
        return result;
    }
};

template<typename ForwardIterator, typename UnaryPredicate>
ForwardIterator partition_semistable_position(ForwardIterator first, ForwardIterator last, UnaryPredicate predicate) {
    while (first != last && !predicate(first)) { ++first; }

    if (first == last) {
        return first;
    }

    ForwardIterator fast = first;
    ++fast;
    while (fast != last) {
        if (predicate(fast)) {
            ++fast;
        } else {
            std::swap(*first, *fast);
            ++first;
            ++fast;
        }
    }
    return first;
}

template<typename It, typename F>
void for_each_position(It first, It last, F func) {
    while (first != last) { 
        func(first);
        ++first;
    }
}

template<typename It0, typename It1, typename T, typename B>
size_t partition_sample(It0 f0, It0 l0, It1 f1, It1 l1, size_t left_stride, size_t right_stride, size_t feature_id, T threshold, B mask) {
    /// had a local mask (possible source of slow down)
    auto range = f0 + feature_id;
    auto first = range.begin() + left_stride;
    auto last =  range.begin() + right_stride;
    for_each_position(first, last, [mask, first, threshold](auto const x) { mask[x - first] = (*x > threshold);});
  
    size_t result = 0;
    while (f0 != l0) {
        auto first = f0.begin() + left_stride;
        auto last =  f0.begin() + right_stride;
        auto pivot0 = partition_semistable_position(first, last, [mask, first](auto pos) { return mask[pos - first]; });
        result = pivot0 - f0.begin();
        ++f0;
    }

    auto f11 = f1 + left_stride;
    auto l11 = f1 + right_stride;
    partition_semistable_position(f11, l11, [&mask, f11](auto pos) { return mask[pos - f11]; });
    return result;
}

template<typename T, typename Criterion>
struct decision_tree {
    typedef T value_type;

    Criterion criterion;
    std::mt19937 g;

    struct node_t {
        size_t feature_id;
        size_t dominant_class;
        value_type value;
        node_t* left = nullptr;
        node_t* right = nullptr;
    };

    node_t* root;

    struct reducer {
        // is was a huge lambda function (possible source of slow down)
        typedef typename std::vector<std::tuple<value_type, size_t>>::iterator iterator;

        std::vector<size_t>* frequencies;
        iterator feature_first;
        iterator feature_last;
        double count;
        Criterion criterion; 

        template<typename T0, typename T1>
        auto operator()(const T0& r, const T1& pivot) {
            auto feature_middle = std::partition(feature_first, feature_last, [pivot](auto x) { return std::get<0>(x) <= pivot; });

            std::fill(std::begin(*frequencies), std::end(*frequencies), 0);
            auto criterion_left = criterion(feature_first, feature_middle, *frequencies);

            std::fill(std::begin(*frequencies), std::end(*frequencies), 0);
            auto criterion_right = criterion(feature_middle, feature_last, *frequencies);

            auto criterion_value = value_type(feature_middle - feature_first) / count * criterion_left + value_type(feature_last - feature_middle) / count * criterion_right;

            std::for_each(feature_first, feature_middle, [this](const auto& x) { ++(*frequencies)[std::get<1>(x)]; });

            size_t dominant_class = std::max_element(std::begin(*frequencies), std::end(*frequencies)) - std::begin(*frequencies);
            auto right = std::make_tuple(criterion_value, pivot, dominant_class);

            return std::min(r, right);
        }
    };
public:
    template<typename It0, typename It1>
    decision_tree(size_t seed, size_t max_unique_values_per_feature, It0 f0, It0 l0, It1 f1, It1 l1) : g(seed) {
        this->root = new node_t;
        try {
            build_tree(max_unique_values_per_feature, f0, l0, f1, l1, 0ul, f0.end() - f0.begin(), this->root);
        } catch(...) {
            delete_tree(root);
        }
    }

    template<typename It0, typename It1>
    bool build_tree(size_t max_unique_values_per_feature, It0 f0, It0 l0, It1 f1, It1 l1, size_t left_stride, size_t right_stride, node_t* root_node) {
        // Initial thoughts:
        // 1. Column-based matrix should be good for performance.
        // 2. Problem: partition by rows.
        // 3. Column-based matrix is less efficient for inference.
        // 4. Build a Decision tree as a separate data structure or even a few separate data structures.
        // 5. Write this code without recursiion (It will save tones of memory allocations).
        // 6. Implement tree with arrays (Will save a few more tones of memory allocations).
        // 7. Have less than 10 memory allocations for the whole training process.

        std::vector<size_t> frequencies(std::accumulate(f1, l1, 0, [](auto r, auto x) { return std::max(r, x); }) + 1);
        size_t rows_n = f0.end() - f0.begin();
        std::vector<std::tuple<value_type, size_t>> feature(rows_n);
        std::vector<value_type> unique_values(rows_n);

        std::vector<bool> mask(rows_n);

        std::vector<std::tuple<value_type, value_type, size_t, size_t>> pivots;
        pivots.reserve(l0 - f0);

        size_t column_index = 0;

        for (auto it = f0; it != l0; ++it, ++column_index) {
            auto it_first = it.begin() + left_stride;
            auto it_last = it.begin() + right_stride;
            auto feature_first = std::begin(feature);

            auto unique_values_first = std::begin(unique_values);
            auto unique_values_last = std::copy(it_first, it_last, unique_values_first);
            std::sort(unique_values_first, unique_values_last);
            unique_values_last = std::unique(unique_values_first, unique_values_last);

            std::shuffle(std::begin(unique_values), unique_values_last, g);

            size_t offset = std::min(max_unique_values_per_feature, size_t(unique_values_last - unique_values_first));
            auto feature_last = std::transform(it_first, it_last, f1 + left_stride, feature_first, [](auto x, auto y) { return std::make_tuple(x, y); });
            value_type count = value_type(feature_last - feature_first);

            auto initial = std::make_tuple(std::numeric_limits<value_type>::max(), value_type(0.0), size_t(0));
            auto pivot = std::accumulate(unique_values_first, unique_values_first + offset, initial, reducer{&frequencies, feature_first, feature_last, count, criterion});
            pivots.emplace_back(std::get<0>(pivot), std::get<1>(pivot), std::get<2>(pivot), column_index);
        }

        auto suboptimal_pivot = std::min_element(std::begin(pivots), std::end(pivots), [](const auto& x, const auto& y) { return std::get<0>(x) < std::get<1>(y);});

        if (suboptimal_pivot == std::end(pivots)) {
            return false;
        }

        root_node->value = std::get<1>(*suboptimal_pivot);
        root_node->dominant_class = std::get<2>(*suboptimal_pivot);
        root_node->feature_id = std::get<3>(*suboptimal_pivot);

        size_t middle_stride = partition_sample(f0, l0, f1, l1, left_stride, right_stride, root_node->feature_id, root_node->value, std::begin(mask)); 

        if (middle_stride != left_stride && middle_stride != right_stride) {
            root_node->left = new node_t;
            try {
                if (!build_tree(max_unique_values_per_feature, f0, l0, f1, l1, left_stride, middle_stride, root_node ->left)) {
                    delete root_node->left;
                    root_node->left = nullptr;
                }
            } catch(...) {
                delete root_node->left;
            }
            root_node->right = new node_t;
            try {
                if (!build_tree(max_unique_values_per_feature, f0, l0, f1, l1, middle_stride, right_stride, root_node->right)) {
                    delete root_node->right;
                    root_node->right = nullptr;
                }
            } catch(...) {
                delete root_node->right;
            }
        }
        return true;
    }

    void delete_tree(node_t* root_node) {
        if (root_node == nullptr) return;
        node_t* l = root_node ->left;
        node_t* r = root_node->right;
        delete root_node;
        delete_tree(l);
        delete_tree(r);
    }

     ~decision_tree() {
        delete_tree(this->root);
     }

    template<typename It>
    size_t operator()(It first, node_t* r) {
        if (first[r->feature_id] < r->value) {
            if (r->left == nullptr) {
                return r->dominant_class;
            }
           return (*this)(first, r->left);
        }
        if (r->right == nullptr) {
            return r->dominant_class;
        }
        return (*this)(first, r->right);
    }

    template<typename It, typename OutputIterator>
    OutputIterator operator()(It first, It last, OutputIterator out) {
        size_t n = first.end() - first.begin();
        std::vector<double> item(last - first);
        for (size_t i = 0; i < n; ++i) {
            item.resize(0);
            for (auto it = first; it != last; ++it) {
                item.push_back(*(it.begin() + i));
            }
            *out = this->operator()(std::begin(item));
            ++out;
        }
        return out;
    }
    
    template<typename It>
    size_t operator()(It first) {
        return this->operator()(first, root);
    }
};

template<typename It0, typename It1, typename T, typename P>
T count_if(It0 f0, It0 l0, It1 f1, T n, P p) {
    while (f0 != l0) {
        if (p(*f0, *f1)) {
            ++n;
        }
        ++f0;
        ++f1;
    }
    return n;
}

template<typename It0, typename It1, typename T>
inline
double precision(It0 first0, It0 last0, It1 first1, T x) {
    return count_if(first0, last0, first1, x, [](const auto& x, const auto& y) { return x == y; }) / T(last0 - first0);
}


int main() {
    std::fstream istream_xy_train("m1.txt");
    matrix<double> x_train;
    istream_xy_train >> x_train;
    std::vector<int> y_train;
    istream_xy_train >> y_train;
    assert(x_train.rows() != 0 && x_train.cols() != 0 && y_train.size() == x_train.rows());

    size_t seed = 3;
    size_t max_unique_values_per_feature = 100;
    
    decision_tree<double, gini<double>> dt(seed, max_unique_values_per_feature, x_train.begin(), x_train.end(), y_train.begin(), y_train.end());

    std::vector<int> y_predicted(y_train.size());
    dt(x_train.begin(), x_train.end(), y_predicted.begin());

    std::cout << "precision=" << precision(y_predicted.begin(), y_predicted.end(), y_train.begin(), 0.0) << std::endl;;
}
