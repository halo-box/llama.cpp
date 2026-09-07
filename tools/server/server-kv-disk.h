#pragma once

// On-disk KV cache: fingerprint a conversation, restore its KV state instead of re-prefilling it.
//
// The key is an exact sha1 of the conversation, not a prefix match. That works because the lookup
// hashes the conversation *truncated at the last assistant message*, which is byte-identical to
// what the previous turn stored:
//
//   turn 1 arrives [sys, u1]              -> no assistant message yet, no lookup
//                                            generate A1, store under key([sys, u1, A1])
//   turn 2 arrives [sys, u1, A1, u2]      -> truncate at last assistant -> key([sys, u1, A1]) HIT
//                                            generate A2, store under key([sys, u1, A1, u2, A2])
//
// Hashing the whole arriving conversation instead would never hit: every request carries one more
// message than anything already stored.

#include <cstddef>
#include <cstdint>
#include <string>

//
// sha1
//

struct sha1_state {
    uint32_t h[5]  = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    uint64_t n_len = 0;
    uint8_t  buf[64] = {0};
    uint32_t n_buf = 0;
};

void        sha1_update(sha1_state & s, const void * data, size_t n);
std::string sha1_hex(sha1_state s); // by value: finalising must not consume the running state

// the running state travels from the HTTP layer (where the messages are) to the slot (where the
// reply is), so it has to survive a trip through the task's json
std::string sha1_state_to_hex(const sha1_state & s);
bool        sha1_state_from_hex(const std::string & hex, sha1_state & s);

//
// conversation keys
//

// Feeds one message into a running conversation digest. Deliberately a plain concatenation, so a
// digest can be extended with the assistant's reply once it exists.
void server_kv_add_message(sha1_state & s, const std::string & role, const std::string & content);
