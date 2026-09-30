#pragma once

// Praat TextGrid writing (the annotation format TIFA emits).
//
// A TextGrid file holds one or more IntervalTiers; TIFA writes three, in this
// order: `texts` (semantic words), `words` (pronunciation groups) and
// `phones`.  Unlabelled gaps are written as empty intervals.

#include "tifa_ggml/types.h"

#include <string>
#include <vector>

namespace tifa_ggml {

struct TextGridInterval {
    double      xmin  = 0.0;
    double      xmax  = 0.0;
    std::string text;
};

struct TextGridTier {
    std::string                  name;
    std::vector<TextGridInterval> intervals;
};

// Serialize a TextGrid in the "long" ooTextFile format Praat (and the Python
// `textgrid` package) round-trips.
std::string format_textgrid(const std::vector<TextGridTier> & tiers, double xmax);

// Write `format_textgrid(...)` to `path` (UTF-8, LF line endings).
void write_textgrid_file(const std::string & path,
                         const std::vector<TextGridTier> & tiers,
                         double xmax);

// Build the three tiers from an alignment result.
//
// Follows SaveTextGridCallback: spans are rounded to milliseconds, zero-width
// spans are handled per `skip_handling`, phone intervals are made monotone,
// and the `texts` / `words` tiers are aggregated by word / group id.
//
// Returns false when the sample must be skipped entirely (skip_handling ==
// Discard and at least one zero-width span exists).  `xmax` is updated to the
// final interval end when it exceeds the spectrogram duration.
bool build_alignment_tiers(const AlignResult & result,
                           SkipHandling skip_handling,
                           double spectrogram_seconds,
                           std::vector<TextGridTier> & tiers_out,
                           double & xmax_out);

}  // namespace tifa_ggml
