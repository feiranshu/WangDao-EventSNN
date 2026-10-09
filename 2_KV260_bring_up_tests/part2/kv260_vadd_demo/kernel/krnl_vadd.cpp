#include <cstdint>

extern "C" void krnl_vadd(const std::int32_t* a,
                           const std::int32_t* b,
                           std::int32_t* c,
                           std::uint32_t count) {
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=c bundle=control
#pragma HLS INTERFACE s_axilite port=count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

vadd_loop:
    for (std::uint32_t i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        c[i] = a[i] + b[i];
    }
}
