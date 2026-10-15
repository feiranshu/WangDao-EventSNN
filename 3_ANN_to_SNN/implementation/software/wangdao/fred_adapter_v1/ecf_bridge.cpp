// C ABI around the unmodified Prophesee Apache-2.0 ECF codec in vendor/.
#include "ecf_codec.h"
#include <cstring>
#include <cstddef>
#include <cstdint>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
static_assert(sizeof(ECF::EventCD) == 16 && offsetof(ECF::EventCD, t) == 8,
              "The HDF5 event ABI must be 16 bytes with timestamp at byte 8");

API long long fred_ecf_decode(const unsigned char* input, size_t size,
                             ECF::EventCD* output, size_t capacity) {
    if (!input || size < 12 || !output) return -1;
    std::uint32_t header;
    std::memcpy(&header, input, sizeof(header));
    auto count = header >> 2;
    if (!count || count > 65535 || count > capacity) return -2;
    try {
        ECF::Decoder decoder;
        return static_cast<long long>(decoder(input, input + size, output) / sizeof(ECF::EventCD));
    } catch (...) { return -3; }
}

API long long fred_ecf_encode(const ECF::EventCD* input, size_t count,
                             unsigned char* output, size_t capacity) {
    if (!input || !output || !count || count > 65535) return -1;
    try {
        ECF::Encoder encoder;
        if (capacity < encoder.getCompressedSize()) return -2;
        return static_cast<long long>(encoder(input, input + count, output));
    } catch (...) { return -3; }
}
