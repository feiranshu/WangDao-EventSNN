#include "snn_accel.h"

static uint32_t read_reg(const snn_accel_t *device, uint32_t offset) {
    return device->base[offset / sizeof(uint32_t)];
}

static void write_reg(snn_accel_t *device, uint32_t offset, uint32_t value) {
    device->base[offset / sizeof(uint32_t)] = value;
}

static uint64_t read_counter(const snn_accel_t *device, uint32_t low_offset) {
    uint32_t high_before;
    uint32_t low;
    uint32_t high_after;
    do {
        high_before = read_reg(device, low_offset + 4U);
        low = read_reg(device, low_offset);
        high_after = read_reg(device, low_offset + 4U);
    } while (high_before != high_after);
    return ((uint64_t)high_after << 32) | low;
}

bool snn_pack_aer_event(uint16_t x, uint16_t y, uint8_t polarity,
                        uint8_t time_bin, uint16_t slice_id,
                        uint64_t *packet_out) {
    if (packet_out == NULL || x >= 320U || y >= 256U || polarity > 1U || time_bin >= 16U) {
        return false;
    }
    *packet_out = ((uint64_t)SNN_PACKET_EVENT) |
                  ((uint64_t)x << 2) |
                  ((uint64_t)y << 11) |
                  ((uint64_t)polarity << 19) |
                  ((uint64_t)time_bin << 20) |
                  ((uint64_t)slice_id << 24);
    return true;
}

uint64_t snn_pack_control_packet(enum snn_packet_type type, uint16_t slice_id) {
    return ((uint64_t)type & 0x3U) | ((uint64_t)slice_id << 24);
}

void snn_configure(snn_accel_t *device, enum snn_control_mode mode,
                   uint32_t density_low, uint32_t density_high, bool enable) {
    if (device == NULL || device->base == NULL || density_low >= density_high) {
        return;
    }
    write_reg(device, SNN_REG_DENSITY_LOW, density_low);
    write_reg(device, SNN_REG_DENSITY_HIGH, density_high);
    write_reg(device, SNN_REG_CONTROL, ((uint32_t)mode & 0x3U) | (enable ? (1U << 4) : 0U));
}

void snn_soft_reset(snn_accel_t *device) {
    uint32_t control = read_reg(device, SNN_REG_CONTROL);
    write_reg(device, SNN_REG_CONTROL, control | (1U << 2));
    write_reg(device, SNN_REG_CONTROL, control & ~(1U << 2));
}

void snn_clear_errors(snn_accel_t *device) {
    uint32_t control = read_reg(device, SNN_REG_CONTROL);
    write_reg(device, SNN_REG_CONTROL, control | (1U << 3));
    write_reg(device, SNN_REG_CONTROL, control & ~(1U << 3));
}

void snn_read_counters(const snn_accel_t *device, snn_counters_t *counters) {
    if (device == NULL || device->base == NULL || counters == NULL) {
        return;
    }
    counters->status = read_reg(device, SNN_REG_STATUS);
    counters->events = read_counter(device, SNN_REG_EVENT_LO);
    counters->cycles = read_counter(device, SNN_REG_CYCLE_LO);
    counters->cache_hits = read_counter(device, SNN_REG_CACHE_HIT_LO);
    counters->external_reads = read_counter(device, SNN_REG_EXT_READ_LO);
    counters->saturations = read_counter(device, SNN_REG_SAT_LO);
}

