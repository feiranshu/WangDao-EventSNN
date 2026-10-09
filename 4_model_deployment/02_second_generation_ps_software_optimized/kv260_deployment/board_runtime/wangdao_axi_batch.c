/*
 * Fast KV260 AXI-Lite batch runner for the exact Proposal-SNN bitstream.
 *
 * The input file is raw records: 512 bytes per patch, in the order emitted by
 * prepare_kv260_batch.py.  The output file is raw little-endian signed int32
 * words: sixteen logits per input patch.  Run as root because /dev/mem is
 * required by the current AXI-Lite MVP; the program only maps the requested
 * 4 KiB peripheral page.
 */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    CONTROL_OFFSET = 0x000,
    STATUS_OFFSET = 0x004,
    CONFIG_OFFSET = 0x008,
    INPUT_OFFSET = 0x100,
    OUTPUT_OFFSET = 0x900,
    MAP_LENGTH = 0x1000,
    INPUT_BYTES = 512,
    INPUT_WORDS = 128,
    OUTPUT_WORDS = 16,
    POLL_LIMIT = 200000,
    FILE_BUFFER_BYTES = 1024 * 1024,
};

typedef struct {
    uint64_t minimum;
    uint64_t maximum;
    uint64_t total;
    uint64_t median;
    uint64_t p95;
    double mean;
} TimingSummary;

static void usage(const char *program) {
    fprintf(stderr, "usage: %s <base-address> <input.bin> <output.bin> [runtime.json]\n", program);
    fprintf(stderr, "       input: 512 raw bytes per patch\n");
    fprintf(stderr, "       output: sixteen little-endian int32 logits per patch\n");
    fprintf(stderr, "       runtime.json: optional PS-observed segmented timing report\n");
}

static uint64_t monotonic_nanoseconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        fprintf(stderr, "clock_gettime failed: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static int compare_u64(const void *left, const void *right) {
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static TimingSummary summarize_timings(const uint64_t *values, uint64_t count) {
    TimingSummary summary = {0U, 0U, 0U, 0U, 0U, 0.0};
    if (count == 0U) return summary;
    uint64_t *ordered = calloc((size_t)count, sizeof(*ordered));
    if (ordered == NULL) {
        fprintf(stderr, "cannot allocate timing summary buffer\n");
        exit(EXIT_FAILURE);
    }
    memcpy(ordered, values, (size_t)count * sizeof(*ordered));
    qsort(ordered, (size_t)count, sizeof(*ordered), compare_u64);
    summary.minimum = ordered[0];
    summary.maximum = ordered[count - 1U];
    summary.median = ordered[(count - 1U) / 2U];
    summary.p95 = ordered[((count * 95U + 99U) / 100U) - 1U];
    long double total = 0.0L;
    for (uint64_t index = 0U; index < count; ++index) total += (long double)values[index];
    summary.total = (uint64_t)total;
    summary.mean = (double)(total / (long double)count);
    free(ordered);
    return summary;
}

static void write_timing_summary(FILE *stream, const char *name,
                                 const uint64_t *values, uint64_t count,
                                 int trailing_comma) {
    const TimingSummary summary = summarize_timings(values, count);
    fprintf(stream,
            "    \"%s\": {\"count\": %" PRIu64
            ", \"total\": %" PRIu64 ", \"mean\": %.3f"
            ", \"median\": %" PRIu64 ", \"p95\": %" PRIu64
            ", \"min\": %" PRIu64 ", \"max\": %" PRIu64 "}%s\n",
            name, count, summary.total, summary.mean, summary.median, summary.p95,
            summary.minimum, summary.maximum, trailing_comma ? "," : "");
}

static void write_runtime_json(const char *path, uint64_t patches, uint64_t input_bytes,
                               const uint64_t *sd_read_ns, const uint64_t *ps_to_pl_ns,
                               const uint64_t *sd_to_pl_ns, const uint64_t *pl_wait_ns,
                               const uint64_t *pl_to_ps_ns, const uint64_t *page_cache_write_ns,
                               const uint64_t *patch_total_ns, uint64_t capture_fsync_ns,
                               uint64_t batch_total_ns, const char *cache_mode) {
    FILE *stream = fopen(path, "w");
    if (stream == NULL) {
        fprintf(stderr, "cannot open runtime report %s: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    const TimingSummary output_copy = summarize_timings(pl_to_ps_ns, patches);
    const TimingSummary page_cache = summarize_timings(page_cache_write_ns, patches);
    fprintf(stream, "{\n");
    fprintf(stream, "  \"schema_version\": 1,\n");
    fprintf(stream, "  \"status\": \"ok\",\n");
    fprintf(stream, "  \"measurement_scope\": \"PS-observed AXI-Lite batch path\",\n");
    fprintf(stream, "  \"clock\": \"CLOCK_MONOTONIC_RAW\",\n");
    fprintf(stream, "  \"input_cache_mode\": \"%s\",\n", cache_mode);
    fprintf(stream, "  \"patches\": %" PRIu64 ",\n", patches);
    fprintf(stream, "  \"input_bytes\": %" PRIu64 ",\n", input_bytes);
    fprintf(stream, "  \"output_bytes\": %" PRIu64 ",\n", patches * OUTPUT_WORDS * 4U);
    fprintf(stream, "  \"timing_units\": \"nanoseconds\",\n");
    fprintf(stream, "  \"timing\": {\n");
    write_timing_summary(stream, "sd_or_page_cache_read_ns", sd_read_ns, patches, 1);
    write_timing_summary(stream, "ps_pack_and_axi_write_ns", ps_to_pl_ns, patches, 1);
    write_timing_summary(stream, "storage_to_pl_observed_ns", sd_to_pl_ns, patches, 1);
    write_timing_summary(stream, "pl_start_to_done_observed_ns", pl_wait_ns, patches, 1);
    write_timing_summary(stream, "axi_output_read_ns", pl_to_ps_ns, patches, 1);
    write_timing_summary(stream, "page_cache_write_ns", page_cache_write_ns, patches, 1);
    write_timing_summary(stream, "patch_total_to_page_cache_ns", patch_total_ns, patches, 1);
    fprintf(stream, "    \"capture_flush_fsync_ns\": %" PRIu64 ",\n", capture_fsync_ns);
    fprintf(stream, "    \"pl_to_storage_component_total_ns\": %" PRIu64 ",\n",
            output_copy.total + page_cache.total + capture_fsync_ns);
    fprintf(stream, "    \"batch_total_ns\": %" PRIu64 "\n", batch_total_ns);
    fprintf(stream, "  },\n");
    fprintf(stream, "  \"limitations\": [\n");
    fprintf(stream, "    \"storage read may be served by the Linux page cache; inspect the mount and cache protocol before calling it SD time\",\n");
    fprintf(stream, "    \"PL time is observed by PS polling and includes AXI access, scheduler and 20 us polling sleeps\",\n");
    fprintf(stream, "    \"pure PL cycles require a hardware cycle counter and a new bitstream\",\n");
    fprintf(stream, "    \"output persistence uses one final fflush and fsync per capture file, not one fsync per patch\"\n");
    fprintf(stream, "  ]\n");
    fprintf(stream, "}\n");
    int persist_error = 0;
    if (fflush(stream) != 0) persist_error = 1;
    if (fsync(fileno(stream)) != 0) persist_error = 1;
    if (fclose(stream) != 0) persist_error = 1;
    if (persist_error) {
        fprintf(stderr, "cannot persist runtime report %s: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

static uint64_t parse_address(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > UINT64_MAX) {
        fprintf(stderr, "invalid AXI base address: %s\n", text);
        exit(EXIT_FAILURE);
    }
    if ((value & 0xfffULL) != 0) {
        fprintf(stderr, "AXI base address must be 4 KiB aligned: %s\n", text);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value;
}

static int read_full(FILE *stream, void *buffer, size_t bytes) {
    unsigned char *cursor = (unsigned char *)buffer;
    size_t total = 0;
    while (total < bytes) {
        size_t count = fread(cursor + total, 1, bytes - total, stream);
        if (count == 0) {
            if (ferror(stream)) {
                return -1;
            }
            return (total == 0) ? 0 : -2;
        }
        total += count;
    }
    return 1;
}

static int write_full(FILE *stream, const void *buffer, size_t bytes) {
    const unsigned char *cursor = (const unsigned char *)buffer;
    size_t total = 0;
    while (total < bytes) {
        size_t count = fwrite(cursor + total, 1, bytes - total, stream);
        if (count == 0) {
            return -1;
        }
        total += count;
    }
    return 0;
}

static void sleep_microseconds(long microseconds) {
    struct timespec request;
    request.tv_sec = microseconds / 1000000L;
    request.tv_nsec = (microseconds % 1000000L) * 1000L;
    while (nanosleep(&request, &request) != 0 && errno == EINTR) {
        /* Retry with the remaining interval. */
    }
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const uint64_t base = parse_address(argv[1]);
    const char *input_name = argv[2];
    const char *output_name = argv[3];
    const char *runtime_name = argc == 5 ? argv[4] : NULL;
    const char *cold_cache_environment = getenv("WANGDAO_COLD_CACHE");
    const int cold_cache = cold_cache_environment != NULL &&
        strcmp(cold_cache_environment, "1") == 0;
    FILE *input = fopen(input_name, "rb");
    if (input == NULL) {
        fprintf(stderr, "cannot open input %s: %s\n", input_name, strerror(errno));
        return EXIT_FAILURE;
    }
    (void)setvbuf(input, NULL, _IOFBF, FILE_BUFFER_BYTES);
    if (fseek(input, 0, SEEK_END) != 0) {
        fprintf(stderr, "cannot seek input %s\n", input_name);
        fclose(input);
        return EXIT_FAILURE;
    }
    long input_length = ftell(input);
    if (input_length < 0 || (uint64_t)input_length % INPUT_BYTES != 0) {
        fprintf(stderr, "input length must be a multiple of %d bytes\n", INPUT_BYTES);
        fclose(input);
        return EXIT_FAILURE;
    }
    const uint64_t patches = (uint64_t)input_length / INPUT_BYTES;
    if (patches == 0) {
        fprintf(stderr, "input contains no patch records\n");
        fclose(input);
        return EXIT_FAILURE;
    }
    rewind(input);
#if defined(POSIX_FADV_SEQUENTIAL)
    (void)posix_fadvise(fileno(input), 0, input_length, POSIX_FADV_SEQUENTIAL);
    if (cold_cache) {
        (void)fsync(fileno(input));
        (void)posix_fadvise(fileno(input), 0, input_length, POSIX_FADV_DONTNEED);
    } else {
        (void)posix_fadvise(fileno(input), 0, input_length, POSIX_FADV_WILLNEED);
    }
#endif

    FILE *output = fopen(output_name, "wb");
    if (output == NULL) {
        fprintf(stderr, "cannot open output %s: %s\n", output_name, strerror(errno));
        fclose(input);
        return EXIT_FAILURE;
    }
    (void)setvbuf(output, NULL, _IOFBF, FILE_BUFFER_BYTES);

    uint64_t *sd_read_ns = NULL;
    uint64_t *ps_to_pl_ns = NULL;
    uint64_t *sd_to_pl_ns = NULL;
    uint64_t *pl_wait_ns = NULL;
    uint64_t *pl_to_ps_ns = NULL;
    uint64_t *page_cache_write_ns = NULL;
    uint64_t *patch_total_ns = NULL;
    if (runtime_name != NULL) {
        sd_read_ns = calloc((size_t)patches, sizeof(*sd_read_ns));
        ps_to_pl_ns = calloc((size_t)patches, sizeof(*ps_to_pl_ns));
        sd_to_pl_ns = calloc((size_t)patches, sizeof(*sd_to_pl_ns));
        pl_wait_ns = calloc((size_t)patches, sizeof(*pl_wait_ns));
        pl_to_ps_ns = calloc((size_t)patches, sizeof(*pl_to_ps_ns));
        page_cache_write_ns = calloc((size_t)patches, sizeof(*page_cache_write_ns));
        patch_total_ns = calloc((size_t)patches, sizeof(*patch_total_ns));
        if (sd_read_ns == NULL || ps_to_pl_ns == NULL || sd_to_pl_ns == NULL ||
            pl_wait_ns == NULL || pl_to_ps_ns == NULL || page_cache_write_ns == NULL ||
            patch_total_ns == NULL) {
            fprintf(stderr, "cannot allocate timing buffers for %" PRIu64 " patches\n", patches);
            fclose(input);
            fclose(output);
            return EXIT_FAILURE;
        }
    }

    int mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem < 0) {
        fprintf(stderr, "cannot open /dev/mem: %s (run this program with sudo)\n", strerror(errno));
        fclose(input);
        fclose(output);
        return EXIT_FAILURE;
    }
    void *mapping = mmap(NULL, MAP_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)base);
    if (mapping == MAP_FAILED) {
        fprintf(stderr, "cannot map AXI page at 0x%" PRIx64 ": %s\n", base, strerror(errno));
        close(mem);
        fclose(input);
        fclose(output);
        return EXIT_FAILURE;
    }
    volatile uint32_t *regs = (volatile uint32_t *)mapping;
    unsigned char input_bytes[INPUT_BYTES];
    uint32_t logits[OUTPUT_WORDS];

    printf("WANGDAO_CONFIG 0x%08" PRIX32 "\n", regs[CONFIG_OFFSET / 4]);
    fflush(stdout);
    const uint64_t batch_started = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
    for (uint64_t patch = 0; patch < patches; ++patch) {
        const uint64_t patch_started = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
        const uint64_t read_started = patch_started;
        int read_result = read_full(input, input_bytes, sizeof(input_bytes));
        if (read_result != 1) {
            fprintf(stderr, "short or failed input record at patch %" PRIu64 "\n", patch);
            munmap(mapping, MAP_LENGTH);
            close(mem);
            fclose(input);
            fclose(output);
            return EXIT_FAILURE;
        }
        const uint64_t read_finished = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
        const uint64_t input_write_started = read_finished;
        for (unsigned int word = 0; word < INPUT_WORDS; ++word) {
            const unsigned int offset = word * 4;
            uint32_t value = (uint32_t)input_bytes[offset]
                           | ((uint32_t)input_bytes[offset + 1] << 8)
                           | ((uint32_t)input_bytes[offset + 2] << 16)
                           | ((uint32_t)input_bytes[offset + 3] << 24);
            regs[(INPUT_OFFSET / 4) + word] = value;
        }
        const uint64_t input_write_finished = runtime_name != NULL ? monotonic_nanoseconds() : 0U;

        const uint64_t pl_started = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
        regs[CONTROL_OFFSET / 4] = 0x00000001U;
        uint32_t status = 0;
        unsigned int polls = 0;
        int complete = 0;
        for (polls = 0; polls < POLL_LIMIT; ++polls) {
            status = regs[STATUS_OFFSET / 4];
            if ((status & 0x2U) != 0U && (status & 0x1U) == 0U) {
                complete = 1;
                break;
            }
            if ((status & 0x4U) != 0U) {
                fprintf(stderr, "hardware error at patch %" PRIu64 ", status=0x%08" PRIX32 "\n",
                        patch, status);
                break;
            }
            sleep_microseconds(20);
        }
        if (!complete) {
            fprintf(stderr, "timeout at patch %" PRIu64 ", status=0x%08" PRIX32 ", polls=%u\n",
                    patch, status, polls);
            munmap(mapping, MAP_LENGTH);
            close(mem);
            fclose(input);
            fclose(output);
            return EXIT_FAILURE;
        }
        const uint64_t pl_finished = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
        const uint64_t output_read_started = pl_finished;
        for (unsigned int index = 0; index < OUTPUT_WORDS; ++index) {
            logits[index] = regs[(OUTPUT_OFFSET / 4) + index];
        }
        const uint64_t output_read_finished = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
        const uint64_t output_write_started = output_read_finished;
        if (write_full(output, logits, sizeof(logits)) != 0) {
            fprintf(stderr, "cannot write output at patch %" PRIu64 "\n", patch);
            munmap(mapping, MAP_LENGTH);
            close(mem);
            fclose(input);
            fclose(output);
            return EXIT_FAILURE;
        }
        const uint64_t output_write_finished = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
        if (runtime_name != NULL) {
            sd_read_ns[patch] = read_finished - read_started;
            ps_to_pl_ns[patch] = input_write_finished - input_write_started;
            sd_to_pl_ns[patch] = input_write_finished - patch_started;
            pl_wait_ns[patch] = pl_finished - pl_started;
            pl_to_ps_ns[patch] = output_read_finished - output_read_started;
            page_cache_write_ns[patch] = output_write_finished - output_write_started;
            patch_total_ns[patch] = output_write_finished - patch_started;
        }
        if ((patch + 1) % 1024 == 0 || patch + 1 == patches) {
            fprintf(stderr, "WANGDAO_PROGRESS patches=%" PRIu64 "/%" PRIu64 "\n",
                    patch + 1, patches);
        }
    }
    const uint64_t sync_started = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
    if (fflush(output) != 0 || (runtime_name != NULL && fsync(fileno(output)) != 0)) {
        fprintf(stderr, "cannot persist output %s: %s\n", output_name, strerror(errno));
        munmap(mapping, MAP_LENGTH);
        close(mem);
        fclose(input);
        fclose(output);
        return EXIT_FAILURE;
    }
    const uint64_t sync_finished = runtime_name != NULL ? monotonic_nanoseconds() : 0U;
    const uint64_t batch_finished = sync_finished;
    printf("WANGDAO_BATCH_PASS patches=%" PRIu64 " logits=%" PRIu64 "\n",
           patches, patches * OUTPUT_WORDS);
    fflush(stdout);
    munmap(mapping, MAP_LENGTH);
    close(mem);
    fclose(input);
    if (fclose(output) != 0) {
        fprintf(stderr, "cannot close output %s: %s\n", output_name, strerror(errno));
        return EXIT_FAILURE;
    }
    if (runtime_name != NULL) {
        write_runtime_json(runtime_name, patches, (uint64_t)input_length,
                           sd_read_ns, ps_to_pl_ns, sd_to_pl_ns, pl_wait_ns,
                           pl_to_ps_ns, page_cache_write_ns, patch_total_ns,
                           sync_finished - sync_started,
                           batch_finished - batch_started,
                           cold_cache ? "requested_cold_file_cache" : "warm_or_unspecified");
        free(sd_read_ns);
        free(ps_to_pl_ns);
        free(sd_to_pl_ns);
        free(pl_wait_ns);
        free(pl_to_ps_ns);
        free(page_cache_write_ns);
        free(patch_total_ns);
        printf("WANGDAO_RUNTIME_JSON %s\n", runtime_name);
    }
    return EXIT_SUCCESS;
}
