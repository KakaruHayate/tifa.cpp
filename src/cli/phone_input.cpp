#include "phone_input.h"

#include "tifa_ggml/errors.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace tifa_cli {

using tifa_ggml::Error;
using tifa_ggml::InvalidArgument;

namespace {

std::string trim(const std::string & s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

}  // namespace

PhoneSequence parse_phone_list(const std::string & text) {
    PhoneSequence out;
    std::istringstream is(text);
    std::string tok;
    while (is >> tok) out.phones.push_back(tok);
    return out;
}

PhoneSequence read_phones_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("cannot open phone list: " + path);
    std::ostringstream buf;
    buf << in.rdbuf();
    PhoneSequence seq = parse_phone_list(buf.str());
    if (seq.phones.empty()) throw Error("no phones found in " + path);
    return seq;
}

PhoneSequence read_textgrid_tier(const std::string & path,
                                 const std::string & tier_name)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("cannot open TextGrid: " + path);

    // Minimal tolerant parser: walk `item [n]:` blocks, keep the one whose
    // `name = "..."` matches, then collect its intervals in file order.
    PhoneSequence out;
    std::string line;
    bool in_target_tier = false;
    bool in_tier = false;
    bool have_interval = false;
    double xmin = 0.0, xmax = 0.0;
    std::string text;

    auto flush = [&]() {
        if (have_interval && !trim(text).empty()) {
            out.phones.push_back(trim(text));
        }
        have_interval = false;
        text.clear();
    };

    while (std::getline(in, line)) {
        const std::string t = trim(line);
        if (t.rfind("item [", 0) == 0) {
            flush();
            in_tier = true;
            in_target_tier = false;
            continue;
        }
        if (!in_tier) continue;

        if (t.rfind("name =", 0) == 0) {
            std::string v = trim(t.substr(6));
            if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
                v = v.substr(1, v.size() - 2);
            }
            in_target_tier = (v == tier_name);
            continue;
        }
        if (t.rfind("intervals [", 0) == 0) {
            flush();
            have_interval = true;
            continue;
        }
        if (!in_target_tier || !have_interval) continue;

        // `key = value` with one optional space around '=' (Praat's own
        // writer, and ours, both emit `text = "..."` — parse by position of
        // '=' so either spelling works)
        const std::size_t eq = t.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(t.substr(0, eq));
        const std::string value = trim(t.substr(eq + 1));

        if (key == "xmin") {
            xmin = std::atof(value.c_str());
        } else if (key == "xmax") {
            xmax = std::atof(value.c_str());
        } else if (key == "text") {
            std::string v = value;
            if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
                v = v.substr(1, v.size() - 2);
            }
            // unescape doubled quotes
            std::string unescaped;
            for (std::size_t i = 0; i < v.size(); ++i) {
                unescaped.push_back(v[i]);
                if (v[i] == '"' && i + 1 < v.size() && v[i + 1] == '"') ++i;
            }
            text = unescaped;
        }
    }
    flush();
    (void)xmin;
    (void)xmax;

    if (out.phones.empty()) {
        throw Error("no non-empty intervals found in tier '" + tier_name +
                    "' of " + path);
    }
    return out;
}

PhoneSequence read_transcriptions_csv(const std::string & path,
                                      const std::string & key)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("cannot open transcriptions csv: " + path);

    std::string header;
    if (!std::getline(in, header)) throw Error("empty csv: " + path);
    std::vector<std::string> cols;
    {
        std::istringstream hs(header);
        std::string c;
        while (std::getline(hs, c, ',')) cols.push_back(trim(c));
    }
    int name_col = -1, ph_col = -1;
    for (std::size_t i = 0; i < cols.size(); ++i) {
        if (cols[i] == "name") name_col = static_cast<int>(i);
        if (cols[i] == "ph_seq") ph_col = static_cast<int>(i);
    }
    if (name_col < 0 || ph_col < 0) {
        throw InvalidArgument("csv must have 'name' and 'ph_seq' columns: " + path);
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        // Split on commas, honouring quoted fields (CSV values here never
        // contain commas, but be forgiving).
        std::vector<std::string> fields;
        std::string cur;
        bool in_quotes = false;
        for (char ch : line) {
            if (ch == '"') { in_quotes = !in_quotes; continue; }
            if (ch == ',' && !in_quotes) { fields.push_back(cur); cur.clear(); continue; }
            cur.push_back(ch);
        }
        fields.push_back(cur);
        if (static_cast<int>(fields.size()) <= std::max(name_col, ph_col)) continue;
        std::string name = trim(fields[static_cast<std::size_t>(name_col)]);
        if (name == key) {
            return parse_phone_list(fields[static_cast<std::size_t>(ph_col)]);
        }
    }
    throw Error("identifier '" + key + "' not found in " + path);
}

}  // namespace tifa_cli

namespace tifa_cli {

// Fold the AP/SP segments into the phone timeline: every long-enough segment
// becomes its own interval, phones overlapping it are split around it, and
// segments living in the gaps (leading/trailing silence, inter-phrase pauses)
// are inserted as they are.  Output is sorted by time and tiles the file.
std::vector<TimedInterval> merge_breath_into_phones(
        const std::vector<TimedInterval> & phones,
        const std::vector<tifa_ggml::BreathSegment> & segments,
        double min_insert, std::size_t * inserted_out) {
    struct Insert { double xmin, xmax; const std::string * label; };
    std::vector<Insert> inserts;
    for (const tifa_ggml::BreathSegment & s : segments) {
        if ((s.label == "AP" || s.label == "SP" || s.label == "EP") && s.end - s.start >= min_insert) {
            inserts.push_back({s.start, s.end, &s.label});
        }
    }

    std::vector<TimedInterval> out;
    std::size_t inserted = 0;
    for (const TimedInterval & p : phones) {
        if (p.text.empty()) continue;   // unlabelled stretches carry no phone
        std::vector<const Insert *> inside;
        for (const Insert & e : inserts) {
            // the annotation may already mark this stretch (an AP the first
            // pass placed itself, a hand-tagged SP): don't double it
            if (p.text == *e.label) continue;
            if (e.xmin >= p.xmin - 1e-9 && e.xmax <= p.xmax + 1e-9) {
                inside.push_back(&e);
            }
        }
        std::sort(inside.begin(), inside.end(),
                  [](const Insert * a, const Insert * b) { return a->xmin < b->xmin; });
        double cursor = p.xmin;
        for (const Insert * e : inside) {
            if (e->xmin - cursor > 1e-4) {
                out.push_back(TimedInterval{cursor, e->xmin, p.text});
            }
            out.push_back(TimedInterval{e->xmin, e->xmax, *e->label});
            ++inserted;
            cursor = e->xmax;
        }
        if (p.xmax - cursor > 1e-4) {
            out.push_back(TimedInterval{cursor, p.xmax, p.text});
        }
    }
    // Segments that do not touch any labelled phone (leading/trailing
    // silence, inter-phrase gaps) are inserted as they are.  Containment in a
    // labelled phone is handled above; a segment straddling a phone boundary
    // is dropped rather than forced in, so the phone timeline stays
    // authoritative.  Unlabelled stretches never count as covered.
    for (const Insert & e : inserts) {
        bool overlaps = false;
        for (const TimedInterval & p : phones) {
            if (p.text.empty()) continue;
            if (e.xmax > p.xmin + 1e-6 && e.xmin < p.xmax - 1e-6) {
                overlaps = true;
                break;
            }
        }
        if (!overlaps) {
            out.push_back(TimedInterval{e.xmin, e.xmax, *e.label});
            ++inserted;
        }
    }
    std::sort(out.begin(), out.end(),
              [](const TimedInterval & a, const TimedInterval & b) {
                  return a.xmin < b.xmin;
              });
    // Adjacent same-label intervals collapse: an insert landing next to an
    // identical phone (any label, truly touching), or the detector's onset
    // lagging the aligner's placement of the same breath by a few frames
    // (breath labels only — 100 ms covers the lag, and fusing two *real*
    // adjacent phones of any other label is not ours to decide).
    constexpr double kBreathMergeGap = 0.1;
    const auto is_breath_label = [](const std::string & text) {
        return text == "AP" || text == "SP" || text == "EP" || text == "br" || text == "sil"
            || text == "pau";
    };
    std::vector<TimedInterval> deduped;
    for (TimedInterval & interval : out) {
        if (!deduped.empty() && deduped.back().text == interval.text) {
            const double gap = interval.xmin - deduped.back().xmax;
            if (gap < 1e-4 || (gap < kBreathMergeGap && is_breath_label(interval.text))) {
                deduped.back().xmax = std::max(deduped.back().xmax, interval.xmax);
                continue;
            }
        }
        deduped.push_back(std::move(interval));
    }
    out = std::move(deduped);
    if (inserted_out != nullptr) *inserted_out = inserted;
    return out;
}

}  // namespace tifa_cli
