/*
 * PS-side fixed H2 refiner for the KV260 Proposal-SNN pipeline.
 *
 * The PL produces one signed Q12 logit vector (16 words) per patch.  This
 * program consumes that capture together with a compact, precomputed event
 * representation, performs Proposal-SNN heatmap routing on the PS, executes
 * the frozen int8 H2 refiner, and writes one (candidate, prediction) byte pair
 * per event.  It is deliberately self-contained and uses no Python runtime.
 */

#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

enum {
    EVENT_MAGIC = 0x32445645U,
    H2_MAGIC = 0x32444857U,
    FILE_VERSION = 1U,
    EVENT_RECORD_BYTES = 36U,
    EVENT_FEATURE_DIM = 28U,
    REFINER_FEATURE_DIM = 32U,
    EVENT_WINDOWS = 80U,
    PATCHES_PER_WINDOW = 336U,
    PATCH_OUTPUT_DIM = 16U,
    HEATMAP_HEIGHT = 34U,
    HEATMAP_WIDTH = 44U,
    ACCUMULATOR_BITS = 24U,
    SCALE_FRACTION_BITS = 20U,
    LOGIT_FRACTION_BITS = 16U,
    PROTOTYPES = 2U,
};

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t event_count;
    uint32_t record_bytes;
    uint32_t windows;
    uint32_t sensor_width;
    uint32_t sensor_height;
    uint32_t feature_dim;
} EventHeader;

typedef struct __attribute__((packed)) {
    uint16_t x;
    uint16_t y;
    uint8_t window;
    uint8_t within_q8;
    uint8_t label;
    uint8_t name;
    uint8_t features[EVENT_FEATURE_DIM];
} EventRecord;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t layer_count;
    uint32_t accumulator_bits;
    uint32_t scale_fraction_bits;
    uint32_t logit_fraction_bits;
    uint32_t prototypes;
    uint32_t input_dim;
    uint32_t output_dim;
    float proposal_threshold;
    float event_threshold;
} H2Header;

typedef struct {
    uint32_t input_dim;
    uint32_t output_dim;
    int8_t *weights;
    int32_t *multipliers;
} H2Layer;

typedef struct {
    H2Header header;
    H2Layer layers[3];
} H2Model;

typedef struct __attribute__((packed)) {
    uint8_t candidate;
    uint8_t prediction;
} PredictionRecord;

_Static_assert(sizeof(EventRecord) == EVENT_RECORD_BYTES, "event record layout");
_Static_assert(sizeof(PredictionRecord) == 2U, "prediction record layout");

static void fail_message(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void fail_errno(const char *message, const char *path) {
    fprintf(stderr, "%s %s: %s\n", message, path, strerror(errno));
    exit(EXIT_FAILURE);
}

static void *checked_alloc(size_t count, size_t size) {
    if (size != 0U && count > SIZE_MAX / size) {
        fail_message("allocation size overflow");
    }
    void *value = calloc(count, size);
    if (value == NULL && count != 0U) {
        fail_message("out of memory");
    }
    return value;
}

static uint64_t monotonic_nanoseconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        fail_errno("clock_gettime failed", "CLOCK_MONOTONIC_RAW");
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static void write_runtime_json(const char *path, const H2Model *model,
                               uint32_t events, uint64_t candidates,
                               const uint64_t effective_macs[3], long process_max_rss_kib,
                               uint64_t input_load_ns, uint64_t compute_ns,
                               uint64_t output_fsync_ns, uint64_t total_ns) {
    FILE *stream = fopen(path, "w");
    if (stream == NULL) fail_errno("cannot open H2 runtime report", path);
    fprintf(stream, "{\n");
    fprintf(stream, "  \"schema_version\": 1,\n");
    fprintf(stream, "  \"status\": \"ok\",\n");
    fprintf(stream, "  \"measurement_scope\": \"KV260 PS fixed H2 refiner\",\n");
    fprintf(stream, "  \"clock\": \"CLOCK_MONOTONIC_RAW\",\n");
    fprintf(stream, "  \"timing_units\": \"nanoseconds\",\n");
    fprintf(stream, "  \"events\": %" PRIu32 ",\n", events);
    fprintf(stream, "  \"candidates\": %" PRIu64 ",\n", candidates);
    uint64_t parameters = 0U;
    uint64_t nonzero_parameters = 0U;
    uint64_t scale_count = 0U;
    for (uint32_t layer_index = 0U; layer_index < model->header.layer_count; ++layer_index) {
        const H2Layer *layer = &model->layers[layer_index];
        const uint64_t layer_parameters = (uint64_t)layer->input_dim * layer->output_dim;
        parameters += layer_parameters;
        scale_count += layer->output_dim;
        for (uint64_t index = 0U; index < layer_parameters; ++index) {
            if (layer->weights[index] != 0) ++nonzero_parameters;
        }
    }
    const uint64_t effective_total = effective_macs[0] + effective_macs[1] + effective_macs[2];
    fprintf(stream, "  \"B_model_complexity\": {\n");
    fprintf(stream, "    \"A01_parameter_count\": %" PRIu64 ",\n", parameters);
    fprintf(stream, "    \"nonzero_parameter_count\": %" PRIu64 ",\n", nonzero_parameters);
    fprintf(stream, "    \"A02_static_storage_bytes\": %" PRIu64 ",\n", parameters + scale_count * 4U);
    fprintf(stream, "    \"weight_storage_bytes_int8\": %" PRIu64 ",\n", parameters);
    fprintf(stream, "    \"scale_storage_bytes_float32\": %" PRIu64 ",\n", scale_count * 4U);
    fprintf(stream, "    \"A03_theoretical_work_memory_bytes_per_call\": 176,\n");
    fprintf(stream, "    \"process_max_rss_kib\": %ld,\n", process_max_rss_kib);
    fprintf(stream, "    \"process_max_rss_note\": \"whole H2 process; not algorithm-only tensor memory\",\n");
    fprintf(stream, "    \"A04_dense_MAC\": %" PRIu64 ",\n", candidates * parameters);
    fprintf(stream, "    \"A05_effective_MAC\": %" PRIu64 ",\n", effective_total);
    fprintf(stream, "    \"effective_MAC_by_layer\": [%" PRIu64 ", %" PRIu64 ", %" PRIu64 "],\n",
            effective_macs[0], effective_macs[1], effective_macs[2]);
    fprintf(stream, "    \"A06_activation_quantizations\": %" PRIu64 ",\n", candidates * UINT64_C(96));
    fprintf(stream, "    \"A06_probability_evaluations\": %" PRIu64 ",\n", candidates);
    fprintf(stream, "    \"effective_operation_note\": \"nonzero activation times nonzero weight; dense C loops are not claimed to skip work\"\n");
    fprintf(stream, "  },\n");
    fprintf(stream, "  \"timing\": {\n");
    fprintf(stream, "    \"input_load_ns\": %" PRIu64 ",\n", input_load_ns);
    fprintf(stream, "    \"h2_compute_ns\": %" PRIu64 ",\n", compute_ns);
    fprintf(stream, "    \"prediction_write_fsync_ns\": %" PRIu64 ",\n", output_fsync_ns);
    fprintf(stream, "    \"total_ns\": %" PRIu64 "\n", total_ns);
    fprintf(stream, "  },\n");
    fprintf(stream, "  \"limitations\": [\n");
    fprintf(stream, "    \"input reads may be served by the Linux page cache\",\n");
    fprintf(stream, "    \"this is PS software time and must not be reported as PL compute time\"\n");
    fprintf(stream, "  ]\n");
    fprintf(stream, "}\n");
    int persist_error = 0;
    if (fflush(stream) != 0) persist_error = 1;
    if (fsync(fileno(stream)) != 0) persist_error = 1;
    if (fclose(stream) != 0) persist_error = 1;
    if (persist_error) fail_errno("cannot persist H2 runtime report", path);
}

static void read_exact(FILE *stream, void *destination, size_t bytes, const char *what) {
    if (fread(destination, 1U, bytes, stream) != bytes) {
        fprintf(stderr, "short or failed read: %s\n", what);
        exit(EXIT_FAILURE);
    }
}

static EventRecord *read_events(const char *path, uint32_t *count_out) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        fail_errno("cannot open event file", path);
    }
    EventHeader header;
    read_exact(stream, &header, sizeof(header), "event header");
    if (header.magic != EVENT_MAGIC || header.version != FILE_VERSION ||
        header.record_bytes != EVENT_RECORD_BYTES || header.windows != EVENT_WINDOWS ||
        header.sensor_width != 346U || header.sensor_height != 260U ||
        header.feature_dim != EVENT_FEATURE_DIM) {
        fclose(stream);
        fail_message("event file header does not match the PS H2 contract");
    }
    EventRecord *events = checked_alloc(header.event_count, sizeof(*events));
    read_exact(stream, events, (size_t)header.event_count * sizeof(*events), "event records");
    fclose(stream);
    *count_out = header.event_count;
    return events;
}

static void free_h2(H2Model *model) {
    for (uint32_t index = 0U; index < 3U; ++index) {
        free(model->layers[index].weights);
        free(model->layers[index].multipliers);
        model->layers[index].weights = NULL;
        model->layers[index].multipliers = NULL;
    }
}

static H2Model read_h2(const char *path) {
    H2Model model;
    memset(&model, 0, sizeof(model));
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        fail_errno("cannot open H2 weight file", path);
    }
    read_exact(stream, &model.header, sizeof(model.header), "H2 header");
    if (model.header.magic != H2_MAGIC || model.header.version != FILE_VERSION ||
        model.header.layer_count != 3U || model.header.accumulator_bits != ACCUMULATOR_BITS ||
        model.header.scale_fraction_bits != SCALE_FRACTION_BITS ||
        model.header.logit_fraction_bits != LOGIT_FRACTION_BITS ||
        model.header.prototypes != PROTOTYPES || model.header.input_dim != REFINER_FEATURE_DIM ||
        model.header.output_dim != 4U) {
        fclose(stream);
        fail_message("H2 weight header does not match the fixed PS contract");
    }
    for (uint32_t index = 0U; index < model.header.layer_count; ++index) {
        H2Layer *layer = &model.layers[index];
        read_exact(stream, &layer->input_dim, sizeof(layer->input_dim), "H2 layer input dimension");
        read_exact(stream, &layer->output_dim, sizeof(layer->output_dim), "H2 layer output dimension");
        if (layer->input_dim == 0U || layer->output_dim == 0U ||
            layer->input_dim > 64U || layer->output_dim > 64U) {
            fclose(stream);
            free_h2(&model);
            fail_message("invalid H2 layer dimensions");
        }
        layer->weights = checked_alloc((size_t)layer->input_dim * layer->output_dim,
                                       sizeof(*layer->weights));
        layer->multipliers = checked_alloc(layer->output_dim, sizeof(*layer->multipliers));
        read_exact(stream, layer->weights,
                   (size_t)layer->input_dim * layer->output_dim * sizeof(*layer->weights),
                   "H2 layer weights");
        read_exact(stream, layer->multipliers,
                   (size_t)layer->output_dim * sizeof(*layer->multipliers),
                   "H2 layer multipliers");
    }
    fclose(stream);
    if (model.layers[0].input_dim != REFINER_FEATURE_DIM ||
        model.layers[0].output_dim != 64U || model.layers[1].input_dim != 64U ||
        model.layers[1].output_dim != 32U || model.layers[2].input_dim != 32U ||
        model.layers[2].output_dim != 4U) {
        free_h2(&model);
        fail_message("H2 layer dimensions do not match 32-64-32-4");
    }
    return model;
}

static int32_t signed_round_divide(__int128 value, int64_t divisor) {
    const int negative = value < 0;
    const __int128 magnitude = negative ? -value : value;
    const __int128 rounded = (magnitude + divisor / 2) / divisor;
    const __int128 signed_value = negative ? -rounded : rounded;
    if (signed_value > INT32_MAX || signed_value < INT32_MIN) {
        fail_message("fixed-point result does not fit int32");
    }
    return (int32_t)signed_value;
}

static int64_t saturate_accumulator(int64_t value) {
    const int64_t minimum = -(INT64_C(1) << (ACCUMULATOR_BITS - 1U));
    const int64_t maximum = (INT64_C(1) << (ACCUMULATOR_BITS - 1U)) - 1;
    if (value < minimum) return minimum;
    if (value > maximum) return maximum;
    return value;
}

static void run_h2(const H2Model *model, const uint8_t features[REFINER_FEATURE_DIM],
                   int32_t logits_q[4], uint64_t effective_macs[3]) {
    uint8_t current[64];
    uint8_t next[64];
    memcpy(current, features, REFINER_FEATURE_DIM);
    for (uint32_t layer_index = 0U; layer_index < model->header.layer_count; ++layer_index) {
        const H2Layer *layer = &model->layers[layer_index];
        for (uint32_t output = 0U; output < layer->output_dim; ++output) {
            int64_t raw = 0;
            for (uint32_t input = 0U; input < layer->input_dim; ++input) {
                const int8_t weight = layer->weights[input * layer->output_dim + output];
                if (current[input] != 0U && weight != 0) ++effective_macs[layer_index];
                raw += (int64_t)current[input] * (int64_t)weight;
            }
            raw = saturate_accumulator(raw);
            const __int128 scaled = (__int128)raw * layer->multipliers[output];
            if (layer_index + 1U < model->header.layer_count) {
                const int32_t rounded = signed_round_divide(scaled, INT64_C(1) << SCALE_FRACTION_BITS);
                next[output] = (uint8_t)(rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded));
            } else {
                logits_q[output] = signed_round_divide(
                    scaled * (INT64_C(1) << LOGIT_FRACTION_BITS),
                    (INT64_C(1) << SCALE_FRACTION_BITS) * 255);
            }
        }
        if (layer_index + 1U < model->header.layer_count) {
            memcpy(current, next, layer->output_dim);
        }
    }
}

static double sigmoid_q12(int32_t value) {
    double source = (double)value / 4096.0;
    if (source > 60.0) source = 60.0;
    if (source < -60.0) source = -60.0;
    return 1.0 / (1.0 + exp(-source));
}

static uint8_t round_u8(double value) {
    long rounded = lrint(value);
    if (rounded < 0L) rounded = 0L;
    if (rounded > 255L) rounded = 255L;
    return (uint8_t)rounded;
}

static double neighbour_score(const float *heatmaps, size_t window, uint32_t y, uint32_t x) {
    double total = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int yy = (int)y + dy;
            int xx = (int)x + dx;
            if (yy < 0) yy = 0;
            if (xx < 0) xx = 0;
            if (yy >= (int)HEATMAP_HEIGHT) yy = (int)HEATMAP_HEIGHT - 1;
            if (xx >= (int)HEATMAP_WIDTH) xx = (int)HEATMAP_WIDTH - 1;
            total += heatmaps[(window * HEATMAP_HEIGHT + (size_t)yy) * HEATMAP_WIDTH + (size_t)xx];
        }
    }
    return total / 9.0;
}

static void build_component_sizes(const uint8_t *mask, uint16_t *sizes) {
    const size_t cells = HEATMAP_HEIGHT * HEATMAP_WIDTH;
    int *queue = checked_alloc(cells, sizeof(*queue));
    int *component = checked_alloc(cells, sizeof(*component));
    for (size_t window = 0U; window < EVENT_WINDOWS; ++window) {
        const size_t base = window * cells;
        memset(sizes + base, 0, cells * sizeof(*sizes));
        for (uint32_t y = 0U; y < HEATMAP_HEIGHT; ++y) {
            for (uint32_t x = 0U; x < HEATMAP_WIDTH; ++x) {
                const size_t start = base + (size_t)y * HEATMAP_WIDTH + x;
                if (!mask[start] || sizes[start] != 0U) continue;
                size_t head = 0U;
                size_t tail = 0U;
                size_t component_count = 0U;
                queue[tail++] = (int)((size_t)y * HEATMAP_WIDTH + x);
                sizes[start] = UINT16_MAX;
                while (head < tail) {
                    const int item = queue[head++];
                    const uint32_t cy = (uint32_t)item / HEATMAP_WIDTH;
                    const uint32_t cx = (uint32_t)item % HEATMAP_WIDTH;
                    component[component_count++] = item;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            const int ny = (int)cy + dy;
                            const int nx = (int)cx + dx;
                            if (ny < 0 || nx < 0 || ny >= (int)HEATMAP_HEIGHT || nx >= (int)HEATMAP_WIDTH) {
                                continue;
                            }
                            const size_t neighbour = base + (size_t)ny * HEATMAP_WIDTH + (size_t)nx;
                            if (mask[neighbour] && sizes[neighbour] == 0U) {
                                sizes[neighbour] = UINT16_MAX;
                                queue[tail++] = ny * (int)HEATMAP_WIDTH + nx;
                            }
                        }
                    }
                }
                const uint16_t value = (uint16_t)(component_count > UINT16_MAX ? UINT16_MAX : component_count);
                for (size_t index = 0U; index < component_count; ++index) {
                    sizes[base + (size_t)component[index]] = value;
                }
            }
        }
    }
    free(queue);
    free(component);
}

static void build_heatmaps(const int32_t *logits, float *heatmaps, double threshold) {
    const size_t cells = HEATMAP_HEIGHT * HEATMAP_WIDTH;
    memset(heatmaps, 0, EVENT_WINDOWS * cells * sizeof(*heatmaps));
    for (uint32_t window = 0U; window < EVENT_WINDOWS; ++window) {
        for (uint32_t patch = 0U; patch < PATCHES_PER_WINDOW; ++patch) {
            const uint32_t patch_y = patch / 21U;
            const uint32_t patch_x = patch % 21U;
            const uint32_t y0 = patch_y * 2U;
            const uint32_t x0 = patch_x * 2U;
            for (uint32_t output = 0U; output < PATCH_OUTPUT_DIM; ++output) {
                const double score = sigmoid_q12(logits[((size_t)window * PATCHES_PER_WINDOW + patch) * PATCH_OUTPUT_DIM + output]);
                const uint32_t y = y0 + output / 4U;
                const uint32_t x = x0 + output % 4U;
                float *destination = &heatmaps[(window * cells) + (size_t)y * HEATMAP_WIDTH + x];
                if ((float)score > *destination) *destination = (float)score;
            }
        }
    }
    (void)threshold;
}

static void build_dilated_mask(const float *heatmaps, uint8_t *mask, double threshold) {
    const size_t cells = HEATMAP_HEIGHT * HEATMAP_WIDTH;
    memset(mask, 0, EVENT_WINDOWS * cells * sizeof(*mask));
    for (size_t window = 0U; window < EVENT_WINDOWS; ++window) {
        const size_t base = window * cells;
        for (uint32_t y = 0U; y < HEATMAP_HEIGHT; ++y) {
            for (uint32_t x = 0U; x < HEATMAP_WIDTH; ++x) {
                uint8_t selected = 0U;
                for (int dy = -1; dy <= 1 && !selected; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int ny = (int)y + dy;
                        const int nx = (int)x + dx;
                        if (ny >= 0 && nx >= 0 && ny < (int)HEATMAP_HEIGHT && nx < (int)HEATMAP_WIDTH &&
                            heatmaps[base + (size_t)ny * HEATMAP_WIDTH + (size_t)nx] >= (float)threshold) {
                            selected = 1U;
                            break;
                        }
                    }
                }
                mask[base + (size_t)y * HEATMAP_WIDTH + x] = selected;
            }
        }
    }
}

static int32_t read_capture(const char *path, int32_t **logits_out) {
    const size_t words = (size_t)EVENT_WINDOWS * PATCHES_PER_WINDOW * PATCH_OUTPUT_DIM;
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) fail_errno("cannot open capture file", path);
    if (fseek(stream, 0L, SEEK_END) != 0) fail_errno("cannot seek capture file", path);
    const long size = ftell(stream);
    if (size < 0L || (size_t)size != words * sizeof(int32_t)) {
        fclose(stream);
        fail_message("capture size does not contain 80 windows of Proposal-SNN logits");
    }
    rewind(stream);
    int32_t *logits = checked_alloc(words, sizeof(*logits));
    read_exact(stream, logits, words * sizeof(*logits), "capture logits");
    fclose(stream);
    *logits_out = logits;
    return (int32_t)words;
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s <events.bin> <capture.bin> <h2_weights.bin> <predictions.bin> [runtime.json]\n", program);
    fprintf(stderr, "predictions.bin contains two bytes per event: candidate, prediction\n");
    fprintf(stderr, "runtime.json is an optional PS-side segmented timing report\n");
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    const char *runtime_name = argc == 6 ? argv[5] : NULL;
    const uint64_t total_started = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
    uint32_t event_count = 0U;
    EventRecord *events = read_events(argv[1], &event_count);
    H2Model model = read_h2(argv[3]);
    int32_t *logits = NULL;
    read_capture(argv[2], &logits);
    const uint64_t input_finished = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
    const uint64_t compute_started = input_finished;

    const size_t cells = HEATMAP_HEIGHT * HEATMAP_WIDTH;
    float *heatmaps = checked_alloc(EVENT_WINDOWS * cells, sizeof(*heatmaps));
    uint8_t *mask = checked_alloc(EVENT_WINDOWS * cells, sizeof(*mask));
    uint16_t *sizes = checked_alloc(EVENT_WINDOWS * cells, sizeof(*sizes));
    build_heatmaps(logits, heatmaps, model.header.proposal_threshold);
    build_dilated_mask(heatmaps, mask, model.header.proposal_threshold);
    build_component_sizes(mask, sizes);

    PredictionRecord *predictions = checked_alloc(event_count, sizeof(*predictions));
    uint64_t candidates = 0U;
    uint64_t positives = 0U;
    uint64_t target_events = 0U;
    uint64_t candidate_targets = 0U;
    uint64_t effective_macs[3] = {0U, 0U, 0U};
    uint8_t features[REFINER_FEATURE_DIM];
    int32_t logits_q[4];
    for (uint32_t index = 0U; index < event_count; ++index) {
        const EventRecord *event = &events[index];
        PredictionRecord result = {0U, 0U};
        if (event->label != 0U) ++target_events;
        const uint32_t x = event->x / 8U;
        const uint32_t y = event->y / 8U;
        if (event->window < EVENT_WINDOWS && x < HEATMAP_WIDTH && y < HEATMAP_HEIGHT &&
            mask[((size_t)event->window * cells) + (size_t)y * HEATMAP_WIDTH + x]) {
            result.candidate = 1U;
            ++candidates;
            if (event->label != 0U) ++candidate_targets;
            memcpy(features, event->features, EVENT_FEATURE_DIM);
            const size_t cell = ((size_t)event->window * cells) + (size_t)y * HEATMAP_WIDTH + x;
            const double score = heatmaps[cell];
            const double neighbour = neighbour_score(heatmaps, event->window, y, x);
            const double component = fmin((double)sizes[cell] / 64.0, 1.0);
            features[28] = round_u8(255.0 * score);
            features[29] = round_u8(255.0 * neighbour);
            features[30] = round_u8(255.0 * component);
            features[31] = event->within_q8;
            run_h2(&model, features, logits_q, effective_macs);
            double max0 = -INFINITY;
            double max1 = -INFINITY;
            for (uint32_t prototype = 0U; prototype < PROTOTYPES; ++prototype) {
                const double class0 = (double)logits_q[prototype] / 65536.0;
                const double class1 = (double)logits_q[PROTOTYPES + prototype] / 65536.0;
                if (class0 > max0) max0 = class0;
                if (class1 > max1) max1 = class1;
            }
            const double shift = max1 - max0;
            const double probability = shift >= 0.0
                ? 1.0 / (1.0 + exp(-shift))
                : exp(shift) / (1.0 + exp(shift));
            if (probability >= (double)model.header.event_threshold) {
                result.prediction = 1U;
                ++positives;
            }
        }
        predictions[index] = result;
    }
    const uint64_t compute_finished = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
    const uint64_t output_started = compute_finished;
    FILE *output = fopen(argv[4], "wb");
    if (output == NULL) fail_errno("cannot open predictions file", argv[4]);
    if (fwrite(predictions, sizeof(*predictions), event_count, output) != event_count) {
        fclose(output);
        fail_message("cannot write predictions file");
    }
    int persist_error = 0;
    if (fflush(output) != 0) persist_error = 1;
    if (runtime_name != NULL && fsync(fileno(output)) != 0) persist_error = 1;
    if (fclose(output) != 0) persist_error = 1;
    if (persist_error) fail_errno("cannot persist predictions file", argv[4]);
    const uint64_t output_finished = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
    printf("WANGDAO_H2_PASS events=%" PRIu32 " candidates=%" PRIu64
           " candidate_targets=%" PRIu64 " targets=%" PRIu64 " positives=%" PRIu64 "\n",
           event_count, candidates, candidate_targets, target_events, positives);
    if (runtime_name != NULL) {
        struct rusage usage_value;
        const long max_rss_kib = getrusage(RUSAGE_SELF, &usage_value) == 0
            ? usage_value.ru_maxrss : -1L;
        write_runtime_json(runtime_name, &model, event_count, candidates,
                           effective_macs, max_rss_kib,
                           input_finished - total_started,
                           compute_finished - compute_started,
                           output_finished - output_started,
                           output_finished - total_started);
        printf("WANGDAO_H2_RUNTIME_JSON %s\n", runtime_name);
    }
    free(predictions);
    free(heatmaps);
    free(mask);
    free(sizes);
    free(logits);
    free(events);
    free_h2(&model);
    return EXIT_SUCCESS;
}
