#ifndef SRC_LIBSLIC3R_AABBTREELINES_HPP_
#define SRC_LIBSLIC3R_AABBTREELINES_HPP_

#include "libslic3r/Point.hpp"
#include "libslic3r/EdgeGrid.hpp"
#include "libslic3r/AABBTreeIndirect.hpp"
#include "libslic3r/Line.hpp"

namespace Slic3r {

struct EdgeGridWrapper {
    EdgeGridWrapper(coord_t edge_width, std::vector<Points> lines) :
            lines(lines), edge_width(edge_width) {

        grid.create(this->lines, edge_width, true);
        grid.calculate_sdf();
    }

    bool signed_distance(const Point &point, coordf_t point_width, coordf_t &dist_out) const {
        coordf_t tmp_dist_out;
        bool found = grid.signed_distance(point, edge_width, tmp_dist_out);
        dist_out = tmp_dist_out - edge_width / 2 - point_width / 2;
        return found;
    }

    EdgeGrid::Grid grid;
    std::vector<Points> lines;
    coord_t edge_width;
};

namespace AABBTreeLines {

namespace detail {

// Nothing to do with COVID-19 social distancing.
template<typename ALineType, typename ATreeType, typename AVectorType>
struct IndexedLinesDistancer {
    using LineType = ALineType;
    using TreeType = ATreeType;
    using VectorType = AVectorType;

    const std::vector<LineType> &lines;
    const TreeType &tree;

    const VectorType origin;
};

template<typename VectorType, typename LineType>
static inline VectorType closest_point_to_line(const VectorType &point, const LineType &line) {
    using Scalar = typename Vector::Scalar;
    VectorType nearest_point;
    line_alg::distance_to_squared(line, point, nearest_point);
    return nearest_point;
}

template<typename IndexedLinesDistancerType, typename Scalar>
static inline Scalar squared_distance_to_indexed_lines_recursive(
        IndexedLinesDistancerType &distancer,
        size_t node_idx,
        Scalar low_sqr_d,
        Scalar up_sqr_d,
        size_t &i,
        Eigen::PlainObjectBase<typename IndexedLinesDistancerType::VectorType> &c)
        {
    using Vector = typename IndexedLinesDistancerType::VectorType;

    if (low_sqr_d > up_sqr_d)
        return low_sqr_d;

    // Save the best achieved hit.
    auto set_min = [&i, &c, &up_sqr_d](const Scalar sqr_d_candidate, const size_t i_candidate,
            const Vector &c_candidate) {
        if (sqr_d_candidate < up_sqr_d) {
            i = i_candidate;
            c = c_candidate;
            up_sqr_d = sqr_d_candidate;
        }
    };

    const auto &node = distancer.tree.node(node_idx);
    assert(node.is_valid());
    if (node.is_leaf())
    {
        const auto &line = distancer.lines[node.idx];
        Vector c_candidate = closest_point_to_line<Vector, typename IndexedLinesDistancerType::LineType>(
                distancer.origin, line);
        set_min((c_candidate - distancer.origin).squaredNorm(), node.idx, c_candidate);
    }
    else
    {
        size_t left_node_idx = node_idx * 2 + 1;
        size_t right_node_idx = left_node_idx + 1;
        const auto &node_left = distancer.tree.node(left_node_idx);
        const auto &node_right = distancer.tree.node(right_node_idx);
        assert(node_left.is_valid());
        assert(node_right.is_valid());

        bool looked_left = false;
        bool looked_right = false;
        const auto &look_left = [&]()
                {
                    size_t i_left;
                    Vector c_left = c;
                    Scalar sqr_d_left = squared_distance_to_indexed_lines_recursive(distancer, left_node_idx,
                            low_sqr_d, up_sqr_d, i_left, c_left);
                    set_min(sqr_d_left, i_left, c_left);
                    looked_left = true;
                };
        const auto &look_right = [&]()
                {
                    size_t i_right;
                    Vector c_right = c;
                    Scalar sqr_d_right = squared_distance_to_indexed_lines_recursive(distancer, right_node_idx,
                            low_sqr_d, up_sqr_d, i_right, c_right);
                    set_min(sqr_d_right, i_right, c_right);
                    looked_right = true;
                };

        // must look left or right if in box
        using BBoxScalar = typename IndexedLinesDistancerType::TreeType::BoundingBox::Scalar;
        if (node_left.bbox.contains(distancer.origin.template cast<BBoxScalar>()))
            look_left();
        if (node_right.bbox.contains(distancer.origin.template cast<BBoxScalar>()))
            look_right();
        // if haven't looked left and could be less than current min, then look
        Scalar left_up_sqr_d = node_left.bbox.squaredExteriorDistance(distancer.origin);
        Scalar right_up_sqr_d = node_right.bbox.squaredExteriorDistance(distancer.origin);
        if (left_up_sqr_d < right_up_sqr_d) {
            if (!looked_left && left_up_sqr_d < up_sqr_d)
                look_left();
            if (!looked_right && right_up_sqr_d < up_sqr_d)
                look_right();
        } else {
            if (!looked_right && right_up_sqr_d < up_sqr_d)
                look_right();
            if (!looked_left && left_up_sqr_d < up_sqr_d)
                look_left();
        }
    }
    return up_sqr_d;
}
}

// Build a balanced AABB Tree over a vector of float lines, balancing the tree
// on centroids of the lines.
// Epsilon is applied to the bounding boxes of the AABB Tree to cope with numeric inaccuracies
// during tree traversal.
template<typename LineType>
inline AABBTreeIndirect::Tree<2, typename LineType::Scalar> build_aabb_tree_over_indexed_lines(
        const std::vector<LineType> &lines,
        //FIXME do we want to apply an epsilon?
        const float eps = 0)
        {
    using TreeType = AABBTreeIndirect::Tree<2, typename LineType::Scalar>;
//    using              CoordType      = typename TreeType::CoordType;
    using VectorType = typename TreeType::VectorType;
    using BoundingBox = typename TreeType::BoundingBox;

    struct InputType {
        size_t idx() const {
            return m_idx;
        }
        const BoundingBox& bbox() const {
            return m_bbox;
        }
        const VectorType& centroid() const {
            return m_centroid;
        }

        size_t m_idx;
        BoundingBox m_bbox;
        VectorType m_centroid;
    };

    std::vector<InputType> input;
    input.reserve(lines.size());
    const VectorType veps(eps, eps);
    for (size_t i = 0; i < lines.size(); ++i) {
        const LineType &line = lines[i];
        InputType n;
        n.m_idx = i;
        n.m_centroid = (line.a + line.b) * 0.5;
        n.m_bbox = BoundingBox(line.a, line.a);
        n.m_bbox.extend(line.b);
        n.m_bbox.min() -= veps;
        n.m_bbox.max() += veps;
        input.emplace_back(n);
    }

    TreeType out;
    out.build(std::move(input));
    return out;
}

// Finding a closest line, its closest point and squared distance to the closest point
// Returns squared distance to the closest point or -1 if the input is empty.
template<typename LineType, typename TreeType, typename VectorType>
inline typename VectorType::Scalar squared_distance_to_indexed_lines(
        const std::vector<LineType> &lines,
        const TreeType &tree,
        const VectorType &point,
        size_t &hit_idx_out,
        Eigen::PlainObjectBase<VectorType> &hit_point_out)
        {
    using Scalar = typename VectorType::Scalar;
    auto distancer = detail::IndexedLinesDistancer<LineType, TreeType, VectorType>
            { lines, tree, point };
    return tree.empty() ?
                          Scalar(-1) :
                          detail::squared_distance_to_indexed_lines_recursive(distancer, size_t(0), Scalar(0),
                                  std::numeric_limits<Scalar>::infinity(), hit_idx_out, hit_point_out);
}

}

}

#endif /* SRC_LIBSLIC3R_AABBTREELINES_HPP_ */
