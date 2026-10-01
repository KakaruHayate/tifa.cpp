#pragma once

// Reading known phone sequences from the common annotation formats:
//   * a whitespace-separated list (inline or a .txt / .lab file)
//   * a Praat TextGrid (the `phones` tier, or --phones-tier NAME)
//   * a DiffSinger `transcriptions.csv` row (the ph_seq column)

#include "breath/breath.h"

#include <string>
#include <vector>

namespace tifa_cli {

struct PhoneSequence {
    std::vector<std::string> phones;
    // Optional per-phone grouping (1-based, non-decreasing).  Empty = one
    // group per phone.
    std::vector<int> groups;
};

// "AP zh e n" / newline-separated text.
PhoneSequence parse_phone_list(const std::string & text);

// Read a whitespace-separated phone list from a file (.txt / .lab).
PhoneSequence read_phones_file(const std::string & path);

// Read a TextGrid and return the intervals of `tier_name` (default "phones").
// Empty intervals (blank text) are skipped.  Throws on parse failure.
PhoneSequence read_textgrid_tier(const std::string & path,
                                 const std::string & tier_name);

// Read `transcriptions.csv` (DiffSinger format) and return the `ph_seq`
// column of the row whose `name` equals `key`.  Throws when not found.
PhoneSequence read_transcriptions_csv(const std::string & path,
                                      const std::string & key);

// A phones-tier interval with its real timeline position (what the G2P
// reader's PhoneSequence throws away and the 2PASS merge needs).
struct TimedInterval {
    double      xmin = 0.0;
    double      xmax = 0.0;
    std::string text;
};

// Fold the AP/SP segments of a breath run into the phone timeline of a
// first-pass alignment: every segment longer than `min_insert` becomes its
// own interval (phones overlapping it are split around it; stretches the
// annotation already labels with the same symbol are skipped), segments in
// the gaps (leading/trailing silence, inter-phrase pauses) are inserted as
// they are, adjacent same-label intervals are collapsed, and unlabelled
// stretches are dropped.  Output is sorted by time; `inserted_out` (when
// non-null) receives how many segments were inserted.
std::vector<TimedInterval> merge_breath_into_phones(
        const std::vector<TimedInterval> & phones,
        const std::vector<tifa_ggml::BreathSegment> & segments,
        double min_insert, std::size_t * inserted_out);

}  // namespace tifa_cli
