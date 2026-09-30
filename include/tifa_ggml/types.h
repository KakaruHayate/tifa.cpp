#pragma once

// Public value types for TIFA forced alignment.

#include <cstdint>
#include <string>
#include <vector>

namespace tifa_ggml {

// One predicted phoneme span (seconds).
struct Span {
    float onset  = 0.0f;
    float offset = 0.0f;
};

// Alignment request: a known phone sequence (already language-resolved labels,
// e.g. "zh/e" or bare "e" for the default language), plus the phone→group map.
struct AlignRequest {
    // One label per token.  Labels may carry a language prefix ("en/iy").
    std::vector<std::string> phones;
    // Optional group id per token (1-based, strictly non-decreasing); empty =
    // one group per phone.  Phones in the same group may not be separated by
    // a gap (see modules/decoding.py `_apply_groups_flat`).
    std::vector<std::int32_t> groups;
    // Language whose symbol prefix is dropped in output labels ("" = keep all).
    std::string language;
    // Label for the single semantic word spanned by the sequence (the `texts`
    // tier); empty when unknown.
    std::string word_text;

    float       skip_penalty  = 0.5f;    // raw cosine cost per skipped phone
    std::string skip_handling = "omit";  // "discard" | "omit" | "preserve"
    std::string score_unit    = "none";  // "none" for a fixed phone sequence
};

// Alignment request for an already-resolved token sequence (the G2P path).
struct AlignTokenRequest {
    std::vector<std::int32_t> tokens;        // vocabulary ids
    std::vector<std::string>  labels;        // display label per token
    std::vector<std::int32_t> words;         // 1-based semantic word id per token
    std::vector<std::int32_t> groups;        // 1-based group id per token (non-decreasing)
    std::vector<std::string>  word_texts;    // indexed by word id - 1
    std::vector<std::string>  group_scripts; // indexed by group id - 1

    std::string language;
    float       skip_penalty  = 0.5f;
    std::string skip_handling = "omit";
    std::string score_unit    = "none";
};

// Reference-free diagnosis metrics (TIFA --stat), plus the alignment itself.
struct AlignResult {
    std::vector<std::string>  labels;     // phone labels after language stripping
    std::vector<Span>         spans;      // per phone, in seconds (raw decode)
    std::vector<std::int32_t> words;      // semantic word id per phone (1-based)
    std::vector<std::int32_t> groups;     // group id per phone

    int   num_frames = 0;                 // valid mel frames (maskT sum)
    float agreement  = 0.0f;              // mean softmax probability of the input tokens
    float confidence = 0.0f;              // mean in-span cosine similarity
    float determinacy = 0.0f;             // evidence concentration vs competitors
    float monotonicity = 0.0f;            // evidence pointing to current/later tokens

    // Per-phone alternate pronunciations chosen by the scoring pass (1-based
    // candidate ids); empty when scoring is disabled or unambiguous.
    std::vector<std::int32_t> choices;

    // Tier labels: `word_texts[word_id - 1]` labels the `texts` tier,
    // `group_scripts[group_id - 1]` labels the `words` tier.
    std::vector<std::string> word_texts;
    std::vector<std::string> group_scripts;
};

// TextGrid skip handling for zero-width spans.
enum class SkipHandling { Discard, Omit, Preserve };

SkipHandling parse_skip_handling(const std::string & s);   // throws InvalidArgument

}  // namespace tifa_ggml
