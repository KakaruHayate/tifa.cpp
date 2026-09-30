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

        if (t.rfind("xmin =", 0) == 0) {
            xmin = std::atof(trim(t.substr(5)).c_str());
        } else if (t.rfind("xmax =", 0) == 0) {
            xmax = std::atof(trim(t.substr(5)).c_str());
        } else if (t.rfind("text =", 0) == 0) {
            std::string v = trim(t.substr(5));
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
