/**
 * Quadrature-TreeSHAP for SHAP TreeEnsemble (path-dependent, background-weighted).
 *
 * Port of the fixed 8-point Gauss–Legendre formulation from XGBoost 3.3+
 * (src/predictor/interpretability/shap.cc, quadrature.h). Uses node_sample_weight
 * for branch probabilities instead of XGBoost SumHess cover.
 *
 * Include after tree_shap.h.
 */
#ifndef QUADRATURE_TREE_SHAP_H
#define QUADRATURE_TREE_SHAP_H

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace quadrature_tree_shap {

constexpr std::size_t kPoints = 8;
constexpr tfloat kUnseen = static_cast<tfloat>(-999);
constexpr tfloat kMinChildWeight = static_cast<tfloat>(1e-12);
constexpr tfloat kPi = static_cast<tfloat>(3.141592653589793238462643383279502884);

struct QuadratureRule {
    tfloat nodes[kPoints];
    tfloat weights[kPoints];
};

using QuadratureBuffer = std::array<tfloat, kPoints>;

inline tfloat branch_weight(tfloat child_weight, tfloat parent_weight) {
    if (parent_weight <= 0) {
        return static_cast<tfloat>(0.5);
    }
    tfloat w = child_weight / parent_weight;
    if (w < kMinChildWeight) {
        return kMinChildWeight;
    }
    return w;
}

inline double legendre_polynomial(std::size_t n, double x) {
    double p0 = 1.0;
    if (n == 0) return p0;
    double p1 = x;
    if (n == 1) return p1;
    for (std::size_t k = 2; k <= n; ++k) {
        double pk = ((2.0 * static_cast<double>(k) - 1.0) * x * p1 -
                     (static_cast<double>(k) - 1.0) * p0) /
                    static_cast<double>(k);
        p0 = p1;
        p1 = pk;
    }
    return p1;
}

inline double legendre_derivative(std::size_t n, double x, double pn) {
    double n_d = static_cast<double>(n);
    return n_d * (x * pn - legendre_polynomial(n - 1, x)) / (x * x - 1.0);
}

inline QuadratureRule make_endpoint_quadrature() {
    constexpr std::size_t kN = kPoints;
    constexpr double kConvergenceEps = 1e-15;
    QuadratureRule rule;

    for (std::size_t i = 0; i < kN; ++i) {
        double theta = kPi * (static_cast<double>(i) + 0.75) / (static_cast<double>(kN) + 0.5);
        double x = std::cos(theta);
        for (std::size_t iter = 0; iter < 64; ++iter) {
            double pn = legendre_polynomial(kN, x);
            double dpn = legendre_derivative(kN, x, pn);
            double dx = pn / dpn;
            x -= dx;
            if (std::abs(dx) < kConvergenceEps) {
                break;
            }
        }

        double pn = legendre_polynomial(kN, x);
        double dpn = legendre_derivative(kN, x, pn);
        double w = 2.0 / ((1.0 - x * x) * dpn * dpn);
        double s = 0.5 * (x + 1.0);
        double ws = 0.5 * w;
        std::size_t out_idx = kN - 1 - i;
        rule.nodes[out_idx] = static_cast<tfloat>(s * s);
        rule.weights[out_idx] = static_cast<tfloat>(2.0 * s * ws);
    }
    return rule;
}

inline const QuadratureRule &get_quadrature_rule() {
    static const QuadratureRule kRule = make_endpoint_quadrature();
    return kRule;
}

inline void add_in_place(QuadratureBuffer *lhs, const QuadratureBuffer &rhs) {
    for (std::size_t i = 0; i < kPoints; ++i) {
        (*lhs)[i] += rhs[i];
    }
}

inline tfloat extract_quadrature_delta(const QuadratureRule &rule, const QuadratureBuffer &h_vals,
                                         tfloat p_enter, tfloat p_exit) {
    tfloat acc = 0;
    if (p_enter != 1) {
        tfloat alpha_enter = p_enter - 1;
        for (std::size_t i = 0; i < kPoints; ++i) {
            acc += alpha_enter * h_vals[i] / (1 + alpha_enter * rule.nodes[i]);
        }
    }
    if (p_exit != 1) {
        tfloat alpha_exit = p_exit - 1;
        for (std::size_t i = 0; i < kPoints; ++i) {
            acc -= alpha_exit * h_vals[i] / (1 + alpha_exit * rule.nodes[i]);
        }
    }
    return acc;
}

inline unsigned hot_child_index(unsigned node_index, const TreeEnsemble &tree, const tfloat *x,
                                const bool *x_missing) {
    const unsigned split_index = tree.features[node_index];
    if (x_missing[split_index]) {
        return tree.children_default[node_index];
    }
    if (tree.threshold_types[node_index] == 0 && x[split_index] <= tree.thresholds[node_index]) {
        return tree.children_left[node_index];
    }
    if (tree.threshold_types[node_index] == 1 &&
        category_in_threshold(tree.thresholds[node_index], x[split_index])) {
        return tree.children_left[node_index];
    }
    return tree.children_right[node_index];
}

inline void write_weighted_leaf_return(const TreeEnsemble &tree, const QuadratureRule &rule,
                                       unsigned node_index, unsigned output_index,
                                       const QuadratureBuffer &c_vals, tfloat w_prod,
                                       QuadratureBuffer *out_h) {
    const tfloat leaf_scale = w_prod * tree.values[node_index * tree.num_outputs + output_index];
    for (std::size_t i = 0; i < kPoints; ++i) {
        (*out_h)[i] = c_vals[i] * leaf_scale * rule.weights[i];
    }
}

struct QuadratureTreeRunner {
    const TreeEnsemble &tree;
    const tfloat *x;
    const bool *x_missing;
    const QuadratureRule &rule;
    std::vector<tfloat> *path_prob;
    tfloat *phi;
    unsigned output_index;

    tfloat child_weight(unsigned parent, unsigned child) const {
        return branch_weight(tree.node_sample_weights[child], tree.node_sample_weights[parent]);
    }

    void visit_child(unsigned split_node, unsigned child_node, tfloat child_weight_val, bool satisfies,
                     const QuadratureBuffer &c_vals, tfloat w_prod, QuadratureBuffer *out_h) {
        const unsigned split_index = tree.features[split_node];
        tfloat p_old = (*path_prob)[split_index];

        tfloat p_e = 0;
        if (p_old == kUnseen) {
            p_e = satisfies ? static_cast<tfloat>(1) / child_weight_val : 0;
        } else {
            p_e = satisfies ? p_old / child_weight_val : 0;
        }

        QuadratureBuffer c_child = c_vals;
        tfloat alpha_e = p_e - 1;
        for (std::size_t i = 0; i < kPoints; ++i) {
            c_child[i] *= 1 + alpha_e * rule.nodes[i];
        }

        if (p_old != kUnseen) {
            tfloat alpha_old = p_old - 1;
            if (alpha_old != 0) {
                for (std::size_t i = 0; i < kPoints; ++i) {
                    c_child[i] /= 1 + alpha_old * rule.nodes[i];
                }
            }
        }

        (*path_prob)[split_index] = p_e;
        run_node(child_node, c_child, w_prod * child_weight_val, out_h);
        phi[split_index * tree.num_outputs + output_index] +=
            extract_quadrature_delta(rule, *out_h, p_e, p_old == kUnseen ? 1 : p_old);
        (*path_prob)[split_index] = p_old;
    }

    void run_node(unsigned node_index, const QuadratureBuffer &c_vals, tfloat w_prod,
                  QuadratureBuffer *out_h) {
        if (tree.is_leaf(node_index)) {
            write_weighted_leaf_return(tree, rule, node_index, output_index, c_vals, w_prod, out_h);
            return;
        }

        const unsigned left = tree.children_left[node_index];
        const unsigned right = tree.children_right[node_index];
        const tfloat left_weight = child_weight(node_index, left);
        const tfloat right_weight = child_weight(node_index, right);
        const bool goes_left = hot_child_index(node_index, tree, x, x_missing) == left;

        QuadratureBuffer right_h{};
        visit_child(node_index, left, left_weight, goes_left, c_vals, w_prod, out_h);
        visit_child(node_index, right, right_weight, !goes_left, c_vals, w_prod, &right_h);
        add_in_place(out_h, right_h);
    }

    void run() {
        if (tree.is_leaf(0)) {
            return;
        }

        QuadratureBuffer c_init{};
        c_init.fill(1);
        QuadratureBuffer h_vals{};
        run_node(0, c_init, 1, &h_vals);
    }
};

inline void reset_path_prob(std::vector<tfloat> *path_prob) {
    std::fill(path_prob->begin(), path_prob->end(), kUnseen);
}

inline void quadrature_tree_shap(const TreeEnsemble &tree, const ExplanationDataset &data,
                                 tfloat *out_contribs, std::vector<tfloat> *path_prob) {
    const QuadratureRule &rule = get_quadrature_rule();
    const unsigned bias_offset = data.M * tree.num_outputs;

    for (unsigned j = 0; j < tree.num_outputs; ++j) {
        out_contribs[bias_offset + j] += tree.values[j];
    }

    if (tree.is_leaf(0)) {
        return;
    }

    reset_path_prob(path_prob);

    for (unsigned j = 0; j < tree.num_outputs; ++j) {
        QuadratureTreeRunner runner{tree, data.X, data.X_missing, rule, path_prob,
                                    out_contribs, j};
        runner.run();
        reset_path_prob(path_prob);
    }
}

inline void dense_tree_quadrature_path_dependent(const TreeEnsemble &trees,
                                                 const ExplanationDataset &data,
                                                 tfloat *out_contribs) {
    TreeEnsemble tree;
    ExplanationDataset instance;
    std::vector<tfloat> path_prob(data.M, kUnseen);

    for (unsigned i = 0; i < data.num_X; ++i) {
        tfloat *instance_out_contribs =
            out_contribs + static_cast<unsigned long long>(i) * (data.M + 1) * trees.num_outputs;
        data.get_x_instance(instance, i);

        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            trees.get_tree(tree, j);
            quadrature_tree_shap(tree, instance, instance_out_contribs, &path_prob);
        }

        for (unsigned j = 0; j < trees.num_outputs; ++j) {
            instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset[j];
        }
    }
}

}  // namespace quadrature_tree_shap

#endif  // QUADRATURE_TREE_SHAP_H
