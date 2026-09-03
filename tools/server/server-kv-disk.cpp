// On-disk KV cache (--kv-cache-dir).
//
// Prefill on some machines costs minutes while the KV state of the same conversation reads back from
// an NVMe in a couple of seconds, so it is worth spending disk to never prefill the same prefix twice.
// Each conversation is fingerprinted with sha1(model tag + serialized prompt tokens) and its sequence
// state written under that name; a returning conversation is restored into a free slot instead of
// being reprocessed. Entries are evicted least-recently-used once the store exceeds --kv-cache-max.
//
// Three files make up one entry:
//
//   <key>.kv   target context sequence state (llama_state_seq_save_file, carries the token list)
//   <key>.dft  draft context sequence state, only when speculative decoding is enabled
//   <key>.idx  small index record, written last and deleted first, so it doubles as a commit marker
//
// The state blobs are multi-GiB, so the index keeps the tokens resident: a lookup can measure the
// longest common prefix against every stored conversation without reading a single byte of state.

#include "server-task.h"

#include "common.h"
#include "llama.h"
#include "server-common.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

//
// sha1
//

struct sha1_state {
    uint32_t h[5]  = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    uint64_t n_len = 0;
    uint8_t  buf[64];
    size_t   n_buf = 0;
};

inline uint32_t sha1_rol(uint32_t v, int b) {
    return (v << b) | (v >> (32 - b));
}

void sha1_compress(sha1_state & s, const uint8_t * p) {
    uint32_t w[80];

    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t(p[4*i + 0]) << 24) | (uint32_t(p[4*i + 1]) << 16) |
               (uint32_t(p[4*i + 2]) <<  8) | (uint32_t(p[4*i + 3]));
    }
    for (int i = 16; i < 80; i++) {
        w[i] = sha1_rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    }

    uint32_t a = s.h[0], b = s.h[1], c = s.h[2], d = s.h[3], e = s.h[4];

    for (int i = 0; i < 80; i++) {
        uint32_t f, k;

        if (i < 20) {
            f = (b & c) | ((~b) & d);         k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ c ^ d;                    k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);  k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;                    k = 0xCA62C1D6;
        }

        const uint32_t t = sha1_rol(a, 5) + f + e + k + w[i];

        e = d;
        d = c;
        c = sha1_rol(b, 30);
        b = a;
        a = t;
    }

    s.h[0] += a; s.h[1] += b; s.h[2] += c; s.h[3] += d; s.h[4] += e;
}

void sha1_update(sha1_state & s, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;

    s.n_len += n;

    while (n > 0) {
        const size_t take = std::min(n, sizeof(s.buf) - s.n_buf);

        memcpy(s.buf + s.n_buf, p, take);

        s.n_buf += take;
        p       += take;
        n       -= take;

        if (s.n_buf == sizeof(s.buf)) {
            sha1_compress(s, s.buf);
            s.n_buf = 0;
        }
    }
}

std::string sha1_hex(sha1_state & s) {
    const uint64_t n_bits = s.n_len * 8;

    const uint8_t pad = 0x80;
    sha1_update(s, &pad, 1);

    const uint8_t zero = 0x00;
    while (s.n_buf != 56) {
        sha1_update(s, &zero, 1);
    }

    uint8_t len[8];
    for (int i = 0; i < 8; i++) {
        len[i] = (uint8_t) (n_bits >> (56 - 8*i));
    }
    sha1_update(s, len, 8);

    char out[41];
    for (int i = 0; i < 5; i++) {
        snprintf(out + 8*i, 9, "%08x", s.h[i]);
    }

    return std::string(out, 40);
}

// the fingerprint the user asked for: the conversation, bound to the model that produced it
std::string kv_fingerprint(const std::string & model_tag, const std::vector<char> & packed) {
    sha1_state s;

    sha1_update(s, model_tag.data(), model_tag.size());

    const uint8_t sep = 0;
    sha1_update(s, &sep, 1);

    sha1_update(s, packed.data(), packed.size());

    return sha1_hex(s);
}

//
// index record
//

constexpr char     KV_IDX_MAGIC[8] = { 'K','V','C','A','C','H','E','1' };
constexpr uint32_t KV_IDX_VERSION  = 1;

int64_t kv_now() {
    return std::filesystem::file_time_type::clock::now().time_since_epoch().count();
}

llama_tokens kv_packed_to_tokens(const std::vector<char> & packed) {
    llama_tokens out(packed.size() / sizeof(llama_token));

    if (!out.empty()) {
        memcpy(out.data(), packed.data(), out.size() * sizeof(llama_token));
    }

    return out;
}

bool kv_write_idx(const std::string & path, const std::vector<char> & packed, uint64_t bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }

    const uint64_t n_packed = packed.size();

    f.write(KV_IDX_MAGIC, sizeof(KV_IDX_MAGIC));
    f.write((const char *) &KV_IDX_VERSION, sizeof(KV_IDX_VERSION));
    f.write((const char *) &n_packed, sizeof(n_packed));
    f.write((const char *) &bytes,    sizeof(bytes));
    f.write(packed.data(), packed.size());

    f.close();

    return f.good();
}

bool kv_read_idx(const std::string & path, std::vector<char> & packed, uint64_t & bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    char     magic[8];
    uint32_t version  = 0;
    uint64_t n_packed = 0;

    f.read(magic, sizeof(magic));
    f.read((char *) &version,  sizeof(version));
    f.read((char *) &n_packed, sizeof(n_packed));
    f.read((char *) &bytes,    sizeof(bytes));

    if (!f || memcmp(magic, KV_IDX_MAGIC, sizeof(magic)) != 0 || version != KV_IDX_VERSION) {
        return false;
    }

    // a truncated index is indistinguishable from a corrupt one; both are simply dropped
    if (n_packed > (1ull << 34) || n_packed % sizeof(llama_token) != 0) {
        return false;
    }

    packed.resize(n_packed);
    f.read(packed.data(), n_packed);

    return f.good() || (size_t) f.gcount() == n_packed;
}

} // namespace

namespace {

// a failed restore must leave the slot genuinely empty in every context, otherwise the next request
// would compute a common prefix against cells that were never installed
void kv_reset_slot(server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);

    if (ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
    }

    prompt.clear();
}

} // namespace

std::string server_prompt_cache::disk_path(const std::string & key) const {
    return (std::filesystem::path(dir) / key).string();
}

size_t server_prompt_cache::disk_size() const {
    size_t res = 0;

    for (const auto & e : disk) {
        res += e.bytes;
    }

    return res;
}

bool server_prompt_cache::disk_init(const std::string & dir, const std::string & model_tag, size_t limit_size, size_t min_tokens, bool has_mtmd) {
    this->dir             = dir;
    this->model_tag       = model_tag;
    this->disk_limit_size = limit_size;
    this->disk_min_tokens = min_tokens;
    this->disk_has_mtmd   = has_mtmd;

    std::error_code ec;

    std::filesystem::create_directories(dir, ec);
    if (ec) {
        SRV_WRN("kv cache: cannot create directory '%s': %s\n", dir.c_str(), ec.message().c_str());
        this->dir.clear();
        return false;
    }

    // an entry is committed by its .idx; anything else left over is from an interrupted write
    std::vector<std::filesystem::path> orphans;

    for (const auto & de : std::filesystem::directory_iterator(dir, ec)) {
        const auto & path = de.path();

        if (path.extension() == ".tmp") {
            orphans.push_back(path);
            continue;
        }

        if (path.extension() != ".idx") {
            continue;
        }

        const std::string key = path.stem().string();

        std::vector<char> packed;
        uint64_t          bytes = 0;

        if (!kv_read_idx(path.string(), packed, bytes)) {
            SRV_WRN("kv cache: dropping unreadable index %s\n", key.c_str());
            orphans.push_back(path);
            continue;
        }

        if (!std::filesystem::exists(disk_path(key) + ".kv", ec)) {
            SRV_WRN("kv cache: dropping index %s with no state file\n", key.c_str());
            orphans.push_back(path);
            continue;
        }

        server_prompt_cache_disk_entry entry;

        entry.key      = key;
        entry.tokens   = server_tokens::deserialize(kv_packed_to_tokens(packed), has_mtmd);
        entry.n_packed = packed.size() / sizeof(llama_token);
        entry.bytes    = bytes;

        const auto mtime = std::filesystem::last_write_time(path, ec);
        entry.t_last = ec ? 0 : mtime.time_since_epoch().count();

        disk.push_back(std::move(entry));
    }

    if (ec) {
        SRV_WRN("kv cache: cannot scan directory '%s': %s\n", dir.c_str(), ec.message().c_str());
        this->dir.clear();
        return false;
    }

    for (const auto & path : orphans) {
        std::filesystem::remove(path, ec);

        // .tmp files carry the full state, their siblings are removed with them
        const std::string base = (path.parent_path() / path.stem()).string();
        std::filesystem::remove(base + ".kv",  ec);
        std::filesystem::remove(base + ".dft", ec);
    }

    std::sort(disk.begin(), disk.end(), [](const auto & a, const auto & b) { return a.t_last < b.t_last; });

    SRV_INF("kv cache: '%s' holds %zu conversation(s), %.3f GiB (limit %.3f GiB)\n",
            dir.c_str(), disk.size(), disk_size() / (1024.0*1024.0*1024.0),
            disk_limit_size / (1024.0*1024.0*1024.0));

    disk_prune();

    return true;
}

void server_prompt_cache::disk_prune() {
    if (dir.empty() || disk_limit_size == 0) {
        return;
    }

    std::error_code ec;

    while (!disk.empty() && disk_size() > disk_limit_size) {
        // disk is kept ordered by t_last, so the front is the least recently used
        auto it = std::min_element(disk.begin(), disk.end(),
                [](const auto & a, const auto & b) { return a.t_last < b.t_last; });

        SRV_WRN("kv cache: evicting %s (%zu tokens, %.3f GiB), store over limit\n",
                it->key.substr(0, 12).c_str(), it->tokens.size(), it->bytes / (1024.0*1024.0*1024.0));

        // .idx first: an entry stops existing the moment its commit marker is gone
        std::filesystem::remove(disk_path(it->key) + ".idx", ec);
        std::filesystem::remove(disk_path(it->key) + ".kv",  ec);
        std::filesystem::remove(disk_path(it->key) + ".dft", ec);

        disk.erase(it);
    }
}

bool server_prompt_cache::disk_store(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    if (dir.empty() || prompt.tokens.size() < disk_min_tokens) {
        return false;
    }

    std::vector<char> packed;
    try {
        packed = prompt.tokens.serialize();
    } catch (const std::exception & err) {
        SRV_WRN("kv cache: cannot serialize prompt: %s\n", err.what());
        return false;
    }

    const std::string key = kv_fingerprint(model_tag, packed);

    // already stored: this is the common case when a slot is saved twice without generating
    for (auto & e : disk) {
        if (e.key == key) {
            e.t_last = kv_now();
            return false;
        }
    }

    const std::string base = disk_path(key);

    std::error_code ec;
    uint64_t bytes = 0;

    const int64_t t_start = ggml_time_us();

    // written under .tmp and renamed, so a crash can never leave a half-written state behind a
    // valid index. The token list travels inside the state file itself.
    {
        const size_t n = llama_state_seq_save_file(ctx_tgt, (base + ".kv.tmp").c_str(), id_slot,
                (const llama_token *) packed.data(), packed.size() / sizeof(llama_token));
        if (n == 0) {
            SRV_WRN("kv cache: failed to write state for %s\n", key.substr(0, 12).c_str());
            std::filesystem::remove(base + ".kv.tmp", ec);
            return false;
        }

        bytes += n;
    }

    if (ctx_dft) {
        const size_t n = llama_state_seq_save_file(ctx_dft, (base + ".dft.tmp").c_str(), id_slot,
                (const llama_token *) packed.data(), packed.size() / sizeof(llama_token));
        if (n == 0) {
            SRV_WRN("kv cache: failed to write draft state for %s\n", key.substr(0, 12).c_str());
            std::filesystem::remove(base + ".kv.tmp",  ec);
            std::filesystem::remove(base + ".dft.tmp", ec);
            return false;
        }

        bytes += n;
    }

    std::filesystem::rename(base + ".kv.tmp", base + ".kv", ec);
    if (ec) {
        SRV_WRN("kv cache: cannot commit state for %s: %s\n", key.substr(0, 12).c_str(), ec.message().c_str());
        std::filesystem::remove(base + ".kv.tmp",  ec);
        std::filesystem::remove(base + ".dft.tmp", ec);
        return false;
    }

    if (ctx_dft) {
        std::filesystem::rename(base + ".dft.tmp", base + ".dft", ec);
    }

    // the index is the commit marker, so it goes last
    if (!kv_write_idx(base + ".idx", packed, bytes)) {
        SRV_WRN("kv cache: cannot write index for %s\n", key.substr(0, 12).c_str());
        std::filesystem::remove(base + ".kv",  ec);
        std::filesystem::remove(base + ".dft", ec);
        return false;
    }

    const double t_ms = (ggml_time_us() - t_start) / 1000.0;

    // supersede earlier turns of this same conversation: any stored prompt that is a prefix of the
    // one just written is reachable through it, so keeping it only costs disk
    for (auto it = disk.begin(); it != disk.end();) {
        if (it->tokens.size() < prompt.tokens.size() &&
            it->tokens.get_common_prefix(prompt.tokens) == it->tokens.size()) {
            SRV_TRC("kv cache: superseding %s (%zu tokens)\n", it->key.substr(0, 12).c_str(), it->tokens.size());

            std::filesystem::remove(disk_path(it->key) + ".idx", ec);
            std::filesystem::remove(disk_path(it->key) + ".kv",  ec);
            std::filesystem::remove(disk_path(it->key) + ".dft", ec);

            it = disk.erase(it);
        } else {
            ++it;
        }
    }

    server_prompt_cache_disk_entry entry;

    entry.key      = key;
    entry.tokens   = prompt.tokens.clone();
    entry.n_packed = packed.size() / sizeof(llama_token);
    entry.bytes    = bytes;
    entry.t_last   = kv_now();

    disk.push_back(std::move(entry));

    SRV_INF("kv cache: stored %s, %zu tokens, %.3f GiB in %.1f ms (%zu entries, %.3f GiB total)\n",
            key.substr(0, 12).c_str(), prompt.tokens.size(), bytes / (1024.0*1024.0*1024.0), t_ms,
            disk.size(), disk_size() / (1024.0*1024.0*1024.0));

    disk_prune();

    return true;
}

bool server_prompt_cache::disk_load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    if (dir.empty() || tokens_new.empty()) {
        return false;
    }

    const int lcp_cur = prompt.tokens.get_common_prefix(tokens_new);

    // same rule as the in-memory tier: only move if it both keeps more of the stored context and
    // covers more of the incoming prompt than what the slot already holds
    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_cur) / prompt.tokens.size() : -1.0f;
    float f_sim_best  = float(lcp_cur) / tokens_new.size();

    auto it_best = disk.end();

    for (auto it = disk.begin(); it != disk.end(); ++it) {
        if (it->tokens.empty()) {
            continue;
        }

        const int lcp = it->tokens.get_common_prefix(tokens_new);

        const float f_keep = float(lcp) / it->tokens.size();
        const float f_sim  = float(lcp) / tokens_new.size();

        // restoring gigabytes to reuse a sliver of them is slower than prefilling the sliver
        if (f_keep < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep && f_sim_best < f_sim) {
            f_keep_best = f_keep;
            f_sim_best  = f_sim;

            it_best = it;
        }
    }

    if (it_best == disk.end()) {
        return false;
    }

    const std::string base = disk_path(it_best->key);

    const int64_t t_start = ggml_time_us();

    llama_tokens packed(std::max<size_t>(1, it_best->n_packed));
    size_t       n_packed = 0;

    // llama_state_seq_load_file clears the destination sequence before installing cells, so a
    // failure here cannot leave another conversation's KV behind under these tokens
    const size_t n_read = llama_state_seq_load_file(ctx_tgt, (base + ".kv").c_str(), id_slot,
            packed.data(), packed.size(), &n_packed);

    if (n_read == 0) {
        SRV_WRN("kv cache: failed to restore %s, dropping it\n", it_best->key.substr(0, 12).c_str());

        std::error_code ec;
        std::filesystem::remove(base + ".idx", ec);
        std::filesystem::remove(base + ".kv",  ec);
        std::filesystem::remove(base + ".dft", ec);

        disk.erase(it_best);

        kv_reset_slot(prompt, ctx_tgt, ctx_dft, id_slot);

        return false;
    }

    packed.resize(n_packed);

    if (ctx_dft) {
        llama_tokens packed_dft(std::max<size_t>(1, it_best->n_packed));
        size_t       n_packed_dft = 0;

        if (llama_state_seq_load_file(ctx_dft, (base + ".dft").c_str(), id_slot,
                    packed_dft.data(), packed_dft.size(), &n_packed_dft) == 0) {
            SRV_WRN("kv cache: no draft state for %s, it will be rebuilt\n", it_best->key.substr(0, 12).c_str());
        }
    }

    server_tokens restored;
    try {
        restored = server_tokens::deserialize(packed, disk_has_mtmd);
    } catch (const std::exception & err) {
        SRV_WRN("kv cache: cannot deserialize tokens for %s: %s\n", it_best->key.substr(0, 12).c_str(), err.what());
        kv_reset_slot(prompt, ctx_tgt, ctx_dft, id_slot);
        return false;
    }

    if (!restored.validate(ctx_tgt)) {
        SRV_WRN("kv cache: %s holds tokens this model cannot represent, dropping it\n", it_best->key.substr(0, 12).c_str());

        std::error_code ec;
        std::filesystem::remove(base + ".idx", ec);
        std::filesystem::remove(base + ".kv",  ec);
        std::filesystem::remove(base + ".dft", ec);

        disk.erase(it_best);

        kv_reset_slot(prompt, ctx_tgt, ctx_dft, id_slot);

        return false;
    }

    const double t_ms = (ggml_time_us() - t_start) / 1000.0;

    SRV_INF("kv cache: restored %s into slot %d, %zu tokens, %.3f GiB in %.1f ms (f_keep = %.3f, f_sim = %.3f)\n",
            it_best->key.substr(0, 12).c_str(), id_slot, restored.size(),
            it_best->bytes / (1024.0*1024.0*1024.0), t_ms, f_keep_best, f_sim_best);

    // checkpoints describe the state this slot used to hold, not the one just installed
    prompt.checkpoints.clear();
    prompt.tokens = std::move(restored);

    it_best->t_last = kv_now();

    std::error_code ec;
    std::filesystem::last_write_time(base + ".idx", std::filesystem::file_time_type::clock::now(), ec);

    return true;
}
