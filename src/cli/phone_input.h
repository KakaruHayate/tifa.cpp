#pragma once

// Reading known phone sequences from the common annotation formats:
//   * a whitespace-separated list (inline or a .txt / .lab file)
//   * a Praat TextGrid (the `phones` tier, or --phones-tier NAME)
//   * a DiffSinger `transcriptions.csv` row (the ph_seq column)

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

}  // namespace tifa_cli
