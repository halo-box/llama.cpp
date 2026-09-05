// On-disk KV cache (--kv-cache-dir). See server-kv-disk.h for the keying scheme.
//
// Prefill on this class of machine costs minutes where the same KV state reads back from NVMe in
// seconds, so it is worth spending disk to never prefill a conversation twice. Measured on
// gemma-4-26B-A4B at q8_0 KV: 15.4 KiB/token, so a 22.9k-token conversation is 344 MiB that writes
// in 45 ms and reads in 47 ms, against 18.4 s to re-prefill it.
//
// An entry is three files:
//
//   <key>.kv   target context state (llama_state_seq_save_file, carries the token list)
//   <key>.dft  draft context state, when speculative decoding is on
//   <key>.idx  index record, written last and deleted first, so it is the commit marker
//
// llama_state_seq_save_file / _load_file are the same primitives the /slots endpoints expose, but
// nothing here goes through HTTP: they are called in-process, on the inference thread, driven by
// the conversation key rather than by hand.

#include "server-kv-disk.h"
#include "server-task.h"

#include "common.h"
#include "llama.h"
#include "server-common.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <vector>

//
// sha1
//

namespace {

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
            f = (b & c) | ((~b) & d);        k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ c ^ d;                   k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;                   k = 0xCA62C1D6;
        }

        const uint32_t t = sha1_rol(a, 5) + f + e + k + w[i];

        e = d; d = c; c = sha1_rol(b, 30); b = a; a = t;
    }

    s.h[0] += a; s.h[1] += b; s.h[2] += c; s.h[3] += d; s.h[4] += e;
}

const char * HEX = "0123456789abcdef";

} // namespace

void sha1_update(sha1_state & s, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;

    s.n_len += n;

    while (n > 0) {
        const size_t take = std::min(n, sizeof(s.buf) - s.n_buf);

        memcpy(s.buf + s.n_buf, p, take);

        s.n_buf += (uint32_t) take;
        p       += take;
        n       -= take;

        if (s.n_buf == sizeof(s.buf)) {
            sha1_compress(s, s.buf);
            s.n_buf = 0;
        }
    }
}

std::string sha1_hex(sha1_state s) {
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

std::string sha1_state_to_hex(const sha1_state & s) {
    uint8_t raw[sizeof(sha1_state)];
    memcpy(raw, &s, sizeof(s));

    std::string out;
    out.reserve(sizeof(raw) * 2);

    for (size_t i = 0; i < sizeof(raw); i++) {
        out += HEX[raw[i] >> 4];
        out += HEX[raw[i] & 0xf];
    }

    return out;
}

bool sha1_state_from_hex(const std::string & hex, sha1_state & s) {
    if (hex.size() != sizeof(sha1_state) * 2) {
        return false;
    }

    uint8_t raw[sizeof(sha1_state)];

    for (size_t i = 0; i < sizeof(raw); i++) {
        const char hi = hex[2*i], lo = hex[2*i + 1];

        const auto nib = [](char ch) -> int {
            if (ch >= '0' && ch <= '9') return ch - '0';
            if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
            return -1;
        };

        const int a = nib(hi), b = nib(lo);
        if (a < 0 || b < 0) {
            return false;
        }

        raw[i] = (uint8_t) ((a << 4) | b);
    }

    memcpy(&s, raw, sizeof(s));

    if (s.n_buf >= sizeof(s.buf)) {
        return false;
    }

    return true;
}

void server_kv_add_message(sha1_state & s, const std::string & role, const std::string & content) {
    sha1_update(s, role.data(), role.size());
    sha1_update(s, "\0", 1);
    sha1_update(s, content.data(), content.size());
    sha1_update(s, "\x01", 1);
}

//
// on-disk store
//

namespace {

constexpr char     KV_IDX_MAGIC[8] = { 'K','V','C','A','C','H','E','2' };
constexpr uint32_t KV_IDX_VERSION  = 2;

int64_t kv_now() {
    return std::filesystem::file_time_type::clock::now().time_since_epoch().count();
}

// file_time_type ticks are implementation defined; derive the tick rate rather than assume it
int64_t kv_ticks_per_sec() {
    using clock = std::filesystem::file_time_type::clock;
    return std::chrono::duration_cast<clock::duration>(std::chrono::seconds(1)).count();
}

bool kv_write_idx(const std::string & path, uint64_t n_packed, uint64_t bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }

    f.write(KV_IDX_MAGIC, sizeof(KV_IDX_MAGIC));
    f.write((const char *) &KV_IDX_VERSION, sizeof(KV_IDX_VERSION));
    f.write((const char *) &n_packed, sizeof(n_packed));
    f.write((const char *) &bytes,    sizeof(bytes));
    f.close();

    return f.good();
}

bool kv_read_idx(const std::string & path, uint64_t & n_packed, uint64_t & bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    char     magic[8];
    uint32_t version = 0;

    f.read(magic, sizeof(magic));
    f.read((char *) &version,  sizeof(version));
    f.read((char *) &n_packed, sizeof(n_packed));
    f.read((char *) &bytes,    sizeof(bytes));

    if (!f || memcmp(magic, KV_IDX_MAGIC, sizeof(magic)) != 0 || version != KV_IDX_VERSION) {
        return false;
    }

    return n_packed <= (1ull << 32);
}

llama_tokens kv_packed_to_tokens(const std::vector<char> & packed) {
    llama_tokens out(packed.size() / sizeof(llama_token));

    if (!out.empty()) {
        memcpy(out.data(), packed.data(), out.size() * sizeof(llama_token));
    }

    return out;
}

// Context checkpoints have to travel with the state.
//
// On a sliding-window model the sequence state only carries the SWA window, so after a restore
// llama_memory_seq_pos_min sits above pos_min_thold and the server will only reuse the prompt if a
// checkpoint reaches further back. Without these the whole restored prefix is discarded and the
// conversation is re-prefilled - which is exactly the state the in-memory tier avoids by copying
// prompt.checkpoints alongside the blob. [TAG_KV_DISK_CKPT]
void kv_write_vec(std::ofstream & f, const std::vector<uint8_t> & v) {
    const uint64_t n = v.size();
    f.write((const char *) &n, sizeof(n));
    if (n) {
        f.write((const char *) v.data(), n);
    }
}

bool kv_read_vec(std::ifstream & f, std::vector<uint8_t> & v) {
    uint64_t n = 0;
    f.read((char *) &n, sizeof(n));
    if (!f || n > (1ull << 34)) {
        return false;
    }
    v.resize(n);
    if (n) {
        f.read((char *) v.data(), n);
    }
    return (bool) f;
}

uint64_t kv_write_checkpoints(const std::string & path, const std::list<common_prompt_checkpoint> & ckpts) {
    if (ckpts.empty()) {
        return 0;
    }

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return 0;
    }

    const uint32_t n = (uint32_t) ckpts.size();
    f.write((const char *) &n, sizeof(n));

    for (const auto & c : ckpts) {
        f.write((const char *) &c.n_tokens, sizeof(c.n_tokens));
        f.write((const char *) &c.pos_min,  sizeof(c.pos_min));
        f.write((const char *) &c.pos_max,  sizeof(c.pos_max));

        kv_write_vec(f, c.data_tgt);
        kv_write_vec(f, c.data_dft);
        kv_write_vec(f, c.data_spec);
    }

    const uint64_t bytes = (uint64_t) f.tellp();
    f.close();

    return f.good() ? bytes : 0;
}

bool kv_read_checkpoints(const std::string & path, std::list<common_prompt_checkpoint> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false; // no checkpoints stored, not an error
    }

    uint32_t n = 0;
    f.read((char *) &n, sizeof(n));
    if (!f || n > 1024) {
        return false;
    }

    for (uint32_t i = 0; i < n; i++) {
        common_prompt_checkpoint c;

        f.read((char *) &c.n_tokens, sizeof(c.n_tokens));
        f.read((char *) &c.pos_min,  sizeof(c.pos_min));
        f.read((char *) &c.pos_max,  sizeof(c.pos_max));

        if (!f || !kv_read_vec(f, c.data_tgt) || !kv_read_vec(f, c.data_dft) || !kv_read_vec(f, c.data_spec)) {
            out.clear();
            return false;
        }

        // the task that made it is long gone; -1 keeps the eviction rule in create_checkpoint honest
        c.id_task = -1;

        out.push_back(std::move(c));
    }

    return true;
}

// a failed restore must leave the slot genuinely empty in every context, otherwise the next request
// would measure a prefix against cells that were never installed
void kv_reset_slot(server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);

    if (ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
    }

    prompt.clear();
}

} // namespace

std::string server_prompt_cache::disk_key(const std::string & conv_key) const {
    // bind the conversation to the weights, so two models sharing a directory cannot collide
    sha1_state s;

    sha1_update(s, model_tag.data(), model_tag.size());
    sha1_update(s, "\0", 1);
    sha1_update(s, conv_key.data(), conv_key.size());

    return sha1_hex(s);
}

std::string server_prompt_cache::disk_path(const std::string & key) const {
    return (std::filesystem::path(dir) / key).string();
}

size_t server_prompt_cache::disk_size() const {
    size_t res = 0;

    for (const auto & [_, e] : disk) {
        res += e.bytes;
    }

    return res;
}

void server_prompt_cache::disk_erase(const std::string & key) {
    std::error_code ec;

    // .idx first: an entry stops existing the moment its commit marker is gone
    std::filesystem::remove(disk_path(key) + ".idx",  ec);
    std::filesystem::remove(disk_path(key) + ".kv",   ec);
    std::filesystem::remove(disk_path(key) + ".dft",  ec);
    std::filesystem::remove(disk_path(key) + ".ckpt", ec);

    disk.erase(key);
}

bool server_prompt_cache::disk_init(const std::string & dir, const std::string & model_tag, size_t limit_size, int64_t ttl_s, bool has_mtmd) {
    this->dir             = dir;
    this->model_tag       = model_tag;
    this->disk_limit_size = limit_size;
    this->disk_ttl_s      = ttl_s;
    this->disk_has_mtmd   = has_mtmd;

    std::error_code ec;

    std::filesystem::create_directories(dir, ec);
    if (ec) {
        SRV_WRN("kv cache: cannot create directory '%s': %s\n", dir.c_str(), ec.message().c_str());
        this->dir.clear();
        return false;
    }

    std::vector<std::filesystem::path> orphans;

    for (const auto & de : std::filesystem::directory_iterator(dir, ec)) {
        const auto & path = de.path();

        if (path.extension() == ".tmp") {
            orphans.push_back(path); // an interrupted write
            continue;
        }

        if (path.extension() != ".idx") {
            continue;
        }

        const std::string key = path.stem().string();

        uint64_t n_packed = 0;
        uint64_t bytes    = 0;

        if (!kv_read_idx(path.string(), n_packed, bytes) ||
            !std::filesystem::exists(disk_path(key) + ".kv", ec)) {
            SRV_WRN("kv cache: dropping incomplete entry %s\n", key.substr(0, 12).c_str());
            orphans.push_back(path);
            continue;
        }

        server_prompt_cache_disk_entry entry;

        entry.n_packed = n_packed;
        entry.bytes    = bytes;

        const auto mtime = std::filesystem::last_write_time(path, ec);
        entry.t_last = ec ? kv_now() : mtime.time_since_epoch().count();

        disk.emplace(key, entry);
    }

    for (const auto & path : orphans) {
        const std::string base = (path.parent_path() / path.stem()).string();

        std::filesystem::remove(path, ec);
        std::filesystem::remove(base + ".kv",   ec);
        std::filesystem::remove(base + ".dft",  ec);
        std::filesystem::remove(base + ".ckpt", ec);
    }

    SRV_INF("kv cache: '%s' holds %zu conversation(s), %.3f GiB (limit %.3f GiB, ttl %lld h)\n",
            dir.c_str(), disk.size(), disk_size() / (1024.0*1024.0*1024.0),
            disk_limit_size / (1024.0*1024.0*1024.0), (long long) (disk_ttl_s / 3600));

    disk_prune();

    return true;
}

void server_prompt_cache::disk_prune() {
    if (dir.empty()) {
        return;
    }

    // age first: a stale conversation is dropped whether or not the store is over its limit
    if (disk_ttl_s > 0) {
        const int64_t cutoff = kv_now() - disk_ttl_s * kv_ticks_per_sec();

        for (auto it = disk.begin(); it != disk.end();) {
            if (it->second.t_last < cutoff) {
                SRV_INF("kv cache: expiring %s (%.3f GiB, older than %lld h)\n",
                        it->first.substr(0, 12).c_str(), it->second.bytes / (1024.0*1024.0*1024.0),
                        (long long) (disk_ttl_s / 3600));

                const std::string key = it->first;
                ++it;
                disk_erase(key);
            } else {
                ++it;
            }
        }
    }

    if (disk_limit_size == 0) {
        return;
    }

    while (!disk.empty() && disk_size() > disk_limit_size) {
        auto lru = std::min_element(disk.begin(), disk.end(),
                [](const auto & a, const auto & b) { return a.second.t_last < b.second.t_last; });

        SRV_INF("kv cache: evicting %s (%.3f GiB), store over its %.3f GiB limit\n",
                lru->first.substr(0, 12).c_str(), lru->second.bytes / (1024.0*1024.0*1024.0),
                disk_limit_size / (1024.0*1024.0*1024.0));

        disk_erase(lru->first);
    }
}

bool server_prompt_cache::disk_store(const server_prompt & prompt, const std::string & conv_key, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    if (dir.empty() || conv_key.empty() || prompt.tokens.size() < disk_min_tokens) {
        return false;
    }

    const std::string key = disk_key(conv_key);

    if (disk.count(key)) {
        disk[key].t_last = kv_now(); // already stored, e.g. a regenerated turn
        return false;
    }

    std::vector<char> packed;
    try {
        packed = prompt.tokens.serialize();
    } catch (const std::exception & err) {
        SRV_WRN("kv cache: cannot serialize prompt: %s\n", err.what());
        return false;
    }

    const std::string base = disk_path(key);

    std::error_code ec;
    uint64_t bytes = 0;

    const int64_t t_start = ggml_time_us();

    // written under .tmp and renamed, so a crash can never leave a half-written state behind a
    // valid index. The token list travels inside the state file itself.
    const size_t n_tgt = llama_state_seq_save_file(ctx_tgt, (base + ".kv.tmp").c_str(), id_slot,
            (const llama_token *) packed.data(), packed.size() / sizeof(llama_token));

    if (n_tgt == 0) {
        SRV_WRN("kv cache: failed to write state for %s\n", key.substr(0, 12).c_str());
        std::filesystem::remove(base + ".kv.tmp", ec);
        return false;
    }

    bytes += n_tgt;

    if (ctx_dft) {
        const size_t n_dft = llama_state_seq_save_file(ctx_dft, (base + ".dft.tmp").c_str(), id_slot,
                (const llama_token *) packed.data(), packed.size() / sizeof(llama_token));

        if (n_dft == 0) {
            SRV_WRN("kv cache: failed to write draft state for %s\n", key.substr(0, 12).c_str());
            std::filesystem::remove(base + ".kv.tmp",  ec);
            std::filesystem::remove(base + ".dft.tmp", ec);
            return false;
        }

        bytes += n_dft;
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

    // checkpoints go with the state: on an SWA model the prefix is unusable without them
    bytes += kv_write_checkpoints(base + ".ckpt", prompt.checkpoints);

    const uint64_t n_packed = packed.size() / sizeof(llama_token);

    // the index is the commit marker, so it goes last
    if (!kv_write_idx(base + ".idx", n_packed, bytes)) {
        SRV_WRN("kv cache: cannot write index for %s\n", key.substr(0, 12).c_str());
        std::filesystem::remove(base + ".kv",   ec);
        std::filesystem::remove(base + ".dft",  ec);
        std::filesystem::remove(base + ".ckpt", ec);
        return false;
    }

    server_prompt_cache_disk_entry entry;

    entry.n_packed = n_packed;
    entry.bytes    = bytes;
    entry.t_last   = kv_now();

    disk[key] = entry;

    SRV_INF("kv cache: stored %s, %zu tokens, %.3f GiB in %.1f ms (%zu entries, %.3f GiB total)\n",
            key.substr(0, 12).c_str(), prompt.tokens.size(), bytes / (1024.0*1024.0*1024.0),
            (ggml_time_us() - t_start) / 1000.0, disk.size(), disk_size() / (1024.0*1024.0*1024.0));

    disk_prune();

    return true;
}

bool server_prompt_cache::disk_load(server_prompt & prompt, const std::string & conv_key, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    if (dir.empty() || conv_key.empty()) {
        return false;
    }

    // prune before the read, so an expired conversation is never resurrected
    disk_prune();

    const std::string key = disk_key(conv_key);

    auto it = disk.find(key);
    if (it == disk.end()) {
        return false;
    }

    const std::string base = disk_path(key);

    const int64_t t_start = ggml_time_us();

    llama_tokens packed(std::max<uint64_t>(1, it->second.n_packed));
    size_t       n_packed = 0;

    // llama_state_seq_load_file clears the destination sequence before installing cells, so a
    // failure here cannot leave another conversation's KV behind under these tokens
    const size_t n_read = llama_state_seq_load_file(ctx_tgt, (base + ".kv").c_str(), id_slot,
            packed.data(), packed.size(), &n_packed);

    if (n_read == 0) {
        SRV_WRN("kv cache: failed to restore %s, dropping it\n", key.substr(0, 12).c_str());
        kv_reset_slot(prompt, ctx_tgt, ctx_dft, id_slot);
        disk_erase(key);
        return false;
    }

    packed.resize(n_packed);

    if (ctx_dft) {
        llama_tokens packed_dft(std::max<uint64_t>(1, it->second.n_packed));
        size_t       n_dft = 0;

        if (llama_state_seq_load_file(ctx_dft, (base + ".dft").c_str(), id_slot,
                    packed_dft.data(), packed_dft.size(), &n_dft) == 0) {
            SRV_WRN("kv cache: no draft state for %s, it will be rebuilt\n", key.substr(0, 12).c_str());
        }
    }

    server_tokens restored;
    try {
        restored = server_tokens::deserialize(packed, disk_has_mtmd);
    } catch (const std::exception & err) {
        SRV_WRN("kv cache: cannot deserialize tokens for %s: %s\n", key.substr(0, 12).c_str(), err.what());
        kv_reset_slot(prompt, ctx_tgt, ctx_dft, id_slot);
        disk_erase(key);
        return false;
    }

    if (!restored.validate(ctx_tgt)) {
        SRV_WRN("kv cache: %s holds tokens this model cannot represent, dropping it\n", key.substr(0, 12).c_str());
        kv_reset_slot(prompt, ctx_tgt, ctx_dft, id_slot);
        disk_erase(key);
        return false;
    }

    // The slot's own checkpoints describe the state it used to hold; replace them with the ones
    // stored alongside this conversation. On an SWA model the restored sequence only carries the
    // window, so without these the server discards the whole prefix and re-prefills. [TAG_KV_DISK_CKPT]
    std::list<common_prompt_checkpoint> ckpts;
    if (!kv_read_checkpoints(base + ".ckpt", ckpts)) {
        ckpts.clear();
    }

    const size_t n_ckpt = ckpts.size();

    SRV_INF("kv cache: restored %s into slot %d, %zu tokens, %zu checkpoint(s), %.3f GiB in %.1f ms\n",
            key.substr(0, 12).c_str(), id_slot, restored.size(), n_ckpt,
            it->second.bytes / (1024.0*1024.0*1024.0), (ggml_time_us() - t_start) / 1000.0);

    prompt.checkpoints = std::move(ckpts);
    prompt.tokens      = std::move(restored);

    it->second.t_last = kv_now();

    std::error_code ec;
    std::filesystem::last_write_time(base + ".idx", std::filesystem::file_time_type::clock::now(), ec);

    return true;
}
