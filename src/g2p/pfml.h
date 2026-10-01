#pragma once

// PFML — Pronunciation Flow Markup Language (openvpi/g2pflow).
//
// PFML embeds language scopes, fixed word boundaries and final phonemes in
// ordinary transcript text; upstream requires it as the interchange format
// for G2P input.  A fragment may be as small as a single word:
//
//   <scope language="ja">東京へ行く</scope>
//   <word language="zh" script="zhong" phonemes="zh ong">重庆</word>
//   <phoneme>ong</phoneme>
//
// No root tag is required.  Direct phonemes bypass G2P entirely (they are the
// final phones the model aligns against); text without a `phonemes` attribute
// goes through the pipeline, with the scope/word language routing converter
// selection and "<lang>/<phoneme>" vocabulary resolution.
//
// Internal API (not exported from the library).

#include "g2p.h"

#include <string>
#include <vector>

namespace tifa_ggml::internal::g2p {

// True when `text` contains a PFML element tag (`<scope`, `<word` or
// `<phoneme`).  Plain text — including a stray '<' from prose — is not PFML.
bool looks_like_pfml(const std::string & text);

// Convert a PFML fragment into words.  `languages` is the CLI allow-list,
// used for fragments (and plain runs) that declare no language of their own.
// Throws InvalidArgument on malformed markup (with the byte offset).
std::vector<Word> convert_pfml(const Pipeline & pipeline, const std::string & text,
                               const std::vector<std::string> & languages);

}  // namespace tifa_ggml::internal::g2p
