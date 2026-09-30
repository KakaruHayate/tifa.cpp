#include "g2p/select.h"

#include <algorithm>
#include <vector>

namespace tifa_ggml::internal::g2p {

std::vector<int> first_choices(const CandidateGrid & grid) {
    std::vector<int> out(static_cast<std::size_t>(grid.W), 0);
    for (int w = 0; w < grid.W; ++w) {
        for (int c = 0; c < grid.C; ++c) {
            if (grid.candidates[static_cast<std::size_t>(w) * grid.C + c]) {
                out[static_cast<std::size_t>(w)] = c + 1;   // 1-based
                break;
            }
        }
    }
    return out;
}

MaterializedPaths materialize_paths(const CandidateGrid & grid,
                                    const std::vector<int> & choices)
{
    const int P = grid.P;
    const int C = grid.C;

    // ---- extract_tokens: pick the chosen column of every row ---------------
    std::vector<int> tokens(static_cast<std::size_t>(P), 0);
    std::vector<int> local_groups(static_cast<std::size_t>(P), 0);
    std::vector<int> owners(static_cast<std::size_t>(P), 0);
    for (int p = 0; p < P; ++p) {
        const int w = grid.words[static_cast<std::size_t>(p)];   // 1-based, 0 = pad
        owners[static_cast<std::size_t>(p)] = w;
        if (w <= 0) continue;
        const int choice = (static_cast<std::size_t>(w - 1) < choices.size())
            ? choices[static_cast<std::size_t>(w - 1)] : 0;
        if (choice <= 0 || choice > C) continue;
        const std::size_t idx = static_cast<std::size_t>(p) * C + (choice - 1);
        tokens[static_cast<std::size_t>(p)]       = grid.paths[idx];
        local_groups[static_cast<std::size_t>(p)] = grid.groups[idx];
    }

    // ---- compact_sequences: left-pack non-zero tokens, keep capacity -------
    const std::size_t n = tokens.size();
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const bool ka = tokens[a] == 0;   // padding sorts last, stable otherwise
        const bool kb = tokens[b] == 0;
        if (ka != kb) return !ka;
        return false;
    });

    std::vector<int> ct(n), cw(n), cg(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = order[i];
        ct[i] = tokens[src];
        cw[i] = owners[src];
        cg[i] = local_groups[src];
    }

    // ---- number the groups globally ---------------------------------------
    MaterializedPaths out;
    out.tokens = ct;
    out.words  = cw;
    out.groups.assign(n, 0);
    int running = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (ct[i] == 0) break;                       // padding tail
        const bool same_prev = i > 0 &&
            cw[i] == cw[i - 1] && cg[i] == cg[i - 1];
        if (!same_prev) ++running;
        out.groups[i] = running;
        ++out.count;
    }
    return out;
}

}  // namespace tifa_ggml::internal::g2p
