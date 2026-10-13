#ifndef SNN_ACCEL_H
#define SNN_ACCEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum snn_packet_type {
    SNN_PACKET_EVENT = 0,
    SNN_PACKET_SLICE_START = 1,
    SNN_PACKET_SLICE_END = 2,
    SNN_PACKET_RESET = 3,
};

enum snn_control_mode {
    SNN_MODE_EXACT = 0,
    SNN_MODE_APPROXIMATE = 1,
    SNN_MODE_AUTO = 2,
};

enum snn_register_offset {
    SNN_REG_CONTROL = 0x00,
    SNN_REG_STATUS = 0x04,
    SNN_REG_DENSITY_LOW = 0x08,
    SNN_REG_DENSITY_HIGH = 0x0C,
    SNN_REG_EVENT_LO = 0x10,
    SNN_REG_EVENT_HI = 0x14,
    SNN_REG_CYCLE_LO = 0x18,
    SNN_REG_CYCLE_HI = 0x1C,
    SNN_REG_CACHE_HIT_LO = 0x20,
    SNN_REG_CACHE_HIT_HI = 0x24,
    SNN_REG_EXT_READ_LO = 0x28,
    SNN_REG_EXT_READ_HI = 0x2C,
    SNN_REG_SAT_LO = 0x30,
    SNN_REG_SAT_HI = 0x34,
};

typedef struct {
    volatile uint32_t *base;
} snn_accel_t;

typedef struct {
    uint64_t events;
    uint64_t cycles;
    uint64_t cache_hits;
    uint64_t external_reads;
    uint64_t saturations;
    uint32_t status;
} snn_counters_t;

bool snn_pack_aer_event(uint16_t x, uint16_t y, uint8_t polarity,
                        uint8_t time_bin, uint16_t slice_id,
                        uint64_t *packet_out);
uint64_t snn_pack_control_packet(enum snn_packet_type type, uint16_t slice_id);
void snn_configure(snn_accel_t *device, enum snn_control_mode mode,
                   uint32_t density_low, uint32_t density_high, bool enable);
void snn_soft_reset(snn_accel_t *device);
void snn_clear_errors(snn_accel_t *device);
void snn_read_counters(const snn_accel_t *device, snn_counters_t *counters);

#ifdef __cplusplus
}
#endif

#endif

