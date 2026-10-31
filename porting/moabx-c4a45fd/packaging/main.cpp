// Out-of-tree consumer: must compile and link against installed headers and
// the installed library only, with no reference to the MOABX source tree.
#include "moabx/mesh/csr_graph.hpp"
#include "moabx/pp/runtime.hpp"
#include "moabx/pp/parallel_for.hpp"
#include <atomic>
#include <cstdio>

int main() {
  // Triangle: 0-1, 0-2, 1-2 as COO.
  const std::vector<moabx::mesh::index_type> rows{0, 0, 1, 1, 2, 2};
  const std::vector<moabx::mesh::index_type> cols{1, 2, 0, 2, 0, 1};
  const auto g = moabx::mesh::csr_graph::from_coo(3, rows, cols);

  if (g.row_count() != 3 || g.nnz() != 6) {
    std::printf("FAIL: row_count=%zu nnz=%zu\n", g.row_count(), g.nnz());
    return 1;
  }
  if (g.row(0).size() != 2) {
    std::printf("FAIL: vertex 0 degree=%zu\n", g.row(0).size());
    return 1;
  }

  moabx::pp::runtime_scope scope;
  const auto exec = moabx::pp::executor::make_default();
  std::atomic<int> seen{0};
  moabx::pp::parallel_for(exec, {.league_size = 8},
                          [&](int) { seen.fetch_add(1, std::memory_order_relaxed); });
  if (seen.load() != 8) {
    std::printf("FAIL: parallel_for visited %d of 8\n", seen.load());
    return 1;
  }

  std::printf("OK csr_graph rows=%zu nnz=%zu, parallel_for visited %d\n",
              g.row_count(), g.nnz(), seen.load());
  return 0;
}
