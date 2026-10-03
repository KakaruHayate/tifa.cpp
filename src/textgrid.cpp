#include "tifa_ggml/textgrid.h"

#include "tifa_ggml/errors.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace tifa_ggml {

SkipHandling parse_skip_handling(const std::string & s) {
    if (s == "discard")  return SkipHandling::Discard;
    if (s == "omit")     return SkipHandling::Omit;
    if (s == "preserve") return SkipHandling::Preserve;
    throw InvalidArgument("unknown skip handling '" + s +
                          "' (expected discard | omit | preserve)");
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

namespace {

// Shortest representation that round-trips: %g with 17 significant digits
// would be ugly, so print with 3 decimals (all our values are ms-rounded) and
// fall back to more digits when needed.
std::string fmt_double(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", v);
    double parsed = std::strtod(buf, nullptr);
    if (std::fabs(parsed - v) > 1e-9) {
        std::snprintf(buf, sizeof(buf), "%.17g", v);
    }
    std::string s(buf);
    // Trim redundant trailing zeros ("0.500" -> "0.5", "0.000" -> "0").
    if (s.find('.') != std::string::npos && s.find_first_of("eE") == std::string::npos) {
        std::size_t last = s.find_last_not_of('0');
        if (last != std::string::npos && s[last] == '.') --last;
        s.erase(last + 1);
    }
    return s;
}

std::string quote_text(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');
    for (char c : text) {
        if (c == '"') out.push_back('"');   // doubled quotes (Praat convention)
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// Port of the Python `textgrid` package's `IntervalTier._fillInTheGaps(null)`,
// which upstream TIFA gets for free because it serialises through that library.
// The decoder's gap states (`G_i` in modules/decoding.py) are frames no token
// claims, so the spans -- and therefore the tiers -- legitimately contain
// holes; Praat's IntervalTier format, however, is meant to run continuously.
// Upstream turns every uncovered stretch into an empty-labelled interval and
// documents it ("Unlabeled gaps ... are filled with \"\"", README).  Emitting
// the holes verbatim instead produces TextGrids that Praat tolerates but that
// strict consumers (vLabeler's `continuous: true` labeler) reject.
std::vector<TextGridInterval> fill_tier_gaps(
        const std::vector<TextGridInterval> & intervals, double xmax) {
    std::vector<TextGridInterval> out;
    out.reserve(intervals.size() + 4);
    double prev = 0.0;
    for (const auto & iv : intervals) {
        if (prev < iv.xmin - 1e-9) out.push_back({prev, iv.xmin, ""});
        out.push_back(iv);
        if (iv.xmax > prev) prev = iv.xmax;
    }
    if (prev < xmax - 1e-9) out.push_back({prev, xmax, ""});
    return out;
}

}  // namespace

std::string format_textgrid(const std::vector<TextGridTier> & tiers, double xmax) {
    std::ostringstream os;
    os << "File type = \"ooTextFile\"\n";
    os << "Object class = \"TextGrid\"\n";
    os << "\n";
    os << "xmin = 0\n";
    os << "xmax = " << fmt_double(xmax) << "\n";
    os << "tiers? <exists>\n";
    os << "size = " << tiers.size() << "\n";
    os << "item []:\n";
    for (std::size_t ti = 0; ti < tiers.size(); ++ti) {
        const auto & tier = tiers[ti];
        os << "\titem [" << (ti + 1) << "]:\n";
        os << "\t\tclass = \"IntervalTier\"\n";
        os << "\t\tname = " << quote_text(tier.name) << "\n";
        os << "\t\txmin = 0\n";
        os << "\t\txmax = " << fmt_double(xmax) << "\n";
        const std::vector<TextGridInterval> intervals = fill_tier_gaps(tier.intervals, xmax);
        os << "\t\tintervals: size = " << intervals.size() << "\n";
        for (std::size_t ii = 0; ii < intervals.size(); ++ii) {
            const auto & iv = intervals[ii];
            os << "\t\t\tintervals [" << (ii + 1) << "]:\n";
            os << "\t\t\t\txmin = " << fmt_double(iv.xmin) << "\n";
            os << "\t\t\t\txmax = " << fmt_double(iv.xmax) << "\n";
            os << "\t\t\t\ttext = " << quote_text(iv.text) << "\n";
        }
    }
    return os.str();
}

void write_textgrid_file(const std::string & path,
                         const std::vector<TextGridTier> & tiers,
                         double xmax)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) throw Error("cannot open TextGrid for writing: " + path);
    const std::string text = format_textgrid(tiers, xmax);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) throw Error("failed writing TextGrid: " + path);
}

// ---------------------------------------------------------------------------
// Tier construction (port of SaveTextGridCallback.on_predict_batch_end)
// ---------------------------------------------------------------------------

namespace {

double round3(double v) { return std::round(v * 1000.0) / 1000.0; }

// Port of `_preserve_skipped_spans`: place 1 ms skipped intervals inside the
// permitted gaps, working in integer milliseconds.
void preserve_skipped_spans(std::vector<TextGridInterval> & intervals,
                            const std::vector<std::int32_t> & groups,
                            double & total_duration)
{
    const std::size_t N = intervals.size();
    struct Ms { long onset, offset; };
    std::vector<Ms> original(N);
    for (std::size_t i = 0; i < N; ++i) {
        original[i].onset  = std::lround(intervals[i].xmin * 1000.0);
        original[i].offset = std::lround(intervals[i].xmax * 1000.0);
    }
    long duration_ms = std::lround(total_duration * 1000.0);

    std::vector<std::pair<long, long>> out;
    std::size_t i = 0;
    while (i < N) {
        const long left = out.empty() ? 0 : out.back().second;
        long onset = original[i].onset;
        long offset = original[i].offset;
        if (offset > onset) {
            onset = std::max(onset, left);
            out.emplace_back(onset, std::max(offset, onset + 1));
            ++i;
            continue;
        }
        std::size_t end = i + 1;
        while (end < N && original[end].onset >= original[end].offset) ++end;
        const long right = std::max(left,
            end < N ? original[end].onset : duration_ms);
        std::size_t gap_index = i;
        while (gap_index <= end && gap_index != 0 && gap_index != N &&
               groups[gap_index - 1] == groups[gap_index]) {
            ++gap_index;
        }
        const std::size_t left_count = std::min(gap_index, end) - i;
        const std::size_t right_count = end - i - left_count;
        for (std::size_t k = 0; k < left_count; ++k) {
            out.emplace_back(left + static_cast<long>(k), left + static_cast<long>(k) + 1);
        }
        const long right_start = std::max(left + static_cast<long>(left_count),
                                          right - static_cast<long>(right_count));
        for (std::size_t k = 0; k < right_count; ++k) {
            out.emplace_back(right_start + static_cast<long>(k),
                             right_start + static_cast<long>(k) + 1);
        }
        i = end;
    }

    if (!out.empty()) duration_ms = std::max(duration_ms, out.back().second);
    intervals.clear();
    for (const auto & iv : out) {
        intervals.push_back({static_cast<double>(iv.first) / 1000.0,
                             static_cast<double>(iv.second) / 1000.0, ""});
    }
    total_duration = static_cast<double>(duration_ms) / 1000.0;
}

}  // namespace

bool build_alignment_tiers(const AlignResult & result,
                           SkipHandling skip_handling,
                           double spectrogram_seconds,
                           std::vector<TextGridTier> & tiers_out,
                           double & xmax_out)
{
    const std::size_t N0 = result.spans.size();
    if (N0 == 0) return false;

    constexpr double kEps = 0.001;

    // Step 1-2: round to milliseconds.
    std::vector<double> onsets(N0), offsets(N0);
    for (std::size_t i = 0; i < N0; ++i) {
        onsets[i]  = round3(result.spans[i].onset);
        offsets[i] = round3(result.spans[i].offset);
    }
    // Step 3: zero-width detection.
    std::vector<std::size_t> zero_idx;
    for (std::size_t i = 0; i < N0; ++i) {
        if (onsets[i] >= offsets[i]) zero_idx.push_back(i);
    }

    std::vector<std::string>  labels = result.labels;
    std::vector<std::int32_t> words  = result.words;
    std::vector<std::int32_t> groups = result.groups;
    double total_duration = round3(spectrogram_seconds);

    // Step 4: skip handling.
    if (!zero_idx.empty()) {
        if (skip_handling == SkipHandling::Discard) return false;
        if (skip_handling == SkipHandling::Omit) {
            std::vector<std::size_t> keep;
            keep.reserve(N0);
            for (std::size_t i = 0; i < N0; ++i) {
                if (!std::binary_search(zero_idx.begin(), zero_idx.end(), i)) keep.push_back(i);
            }
            if (keep.empty()) return false;
            std::vector<double> on, off;
            std::vector<std::string> lb;
            std::vector<std::int32_t> wd, gr;
            for (std::size_t i : keep) {
                on.push_back(onsets[i]);
                off.push_back(offsets[i]);
                lb.push_back(labels[i]);
                wd.push_back(words[i]);
                gr.push_back(groups[i]);
            }
            onsets.swap(on);
            offsets.swap(off);
            labels.swap(lb);
            words.swap(wd);
            groups.swap(gr);
        } else {
            std::vector<TextGridInterval> iv(N0);
            for (std::size_t i = 0; i < N0; ++i) iv[i] = {onsets[i], offsets[i], ""};
            preserve_skipped_spans(iv, groups, total_duration);
            onsets.resize(iv.size());
            offsets.resize(iv.size());
            for (std::size_t i = 0; i < iv.size(); ++i) {
                onsets[i]  = iv[i].xmin;
                offsets[i] = iv[i].xmax;
            }
        }
    }
    const std::size_t N = onsets.size();
    if (N == 0) return false;

    // Step 5: phone intervals with monotone clamping and +1 ms inflation.
    std::vector<double> phone_on(N), phone_off(N);
    for (std::size_t n = 0; n < N; ++n) {
        double onset = onsets[n];
        if (n > 0 && onset < phone_off[n - 1]) onset = phone_off[n - 1];
        double offset = offsets[n];
        if (offset <= onset) offset = onset + kEps;
        phone_on[n]  = onset;
        phone_off[n] = offset;
    }
    if (phone_off[N - 1] > total_duration) total_duration = phone_off[N - 1];

    // Step 6: aggregate words / groups, then append the phones tier.
    tiers_out.clear();
    struct TierSpec { const char * name; const std::vector<std::int32_t> * owners; };
    const std::vector<std::int32_t> texts_owner =
        words;   // semantic word id per phone (1-based)

    std::vector<std::pair<double, double>> intervals;
    std::vector<std::int32_t> orders;
    for (int pass = 0; pass < 2; ++pass) {
        const std::vector<std::int32_t> & owners = (pass == 0) ? words : groups;
        intervals.clear();
        orders.clear();
        std::size_t i = 0;
        while (i < N) {
            const std::int32_t owner = owners[i];
            std::size_t j = i + 1;
            while (j < N && owners[j] == owner) ++j;
            double onset = phone_on[i];
            double offset = phone_off[j - 1];
            if (!intervals.empty() && onset < intervals.back().second) {
                onset = intervals.back().second;
            }
            if (offset <= onset) offset = onset + kEps;
            intervals.emplace_back(onset, offset);
            orders.push_back(owner);
            i = j;
        }
        TextGridTier tier;
        tier.name = (pass == 0) ? "texts" : "words";
        const std::vector<std::string> & texts =
            (pass == 0) ? result.word_texts : result.group_scripts;
        for (std::size_t k = 0; k < intervals.size(); ++k) {
            const std::int32_t owner = orders[k];
            const std::string label =
                (owner >= 1 && static_cast<std::size_t>(owner) <= texts.size())
                    ? texts[static_cast<std::size_t>(owner) - 1]
                    : std::string();
            tier.intervals.push_back({intervals[k].first, intervals[k].second, label});
        }
        tiers_out.push_back(std::move(tier));
    }
    (void)texts_owner;

    TextGridTier phones_tier;
    phones_tier.name = "phones";
    for (std::size_t n = 0; n < N; ++n) {
        phones_tier.intervals.push_back({phone_on[n], phone_off[n], labels[n]});
    }
    tiers_out.push_back(std::move(phones_tier));

    xmax_out = total_duration;
    return true;
}

}  // namespace tifa_ggml
