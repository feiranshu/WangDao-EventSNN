#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kRequestMagic = 0x56414444;
constexpr std::uint32_t kResponseMagic = 0x56415253;
constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::uint32_t kMaximumElements = 16U * 1024U * 1024U;

struct Timing {
    double host_to_device_ms{};
    double kernel_ms{};
    double device_to_host_ms{};
};

double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::vector<std::int32_t> read_i32_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("无法打开输入文件: " + path);
    }
    const auto size = stream.tellg();
    if (size <= 0 || size % static_cast<std::streamoff>(sizeof(std::int32_t)) != 0) {
        throw std::runtime_error("输入文件为空或长度不是4字节的整数倍: " + path);
    }
    stream.seekg(0);
    std::vector<std::int32_t> values(static_cast<std::size_t>(size) / sizeof(std::int32_t));
    stream.read(reinterpret_cast<char*>(values.data()), size);
    if (!stream) {
        throw std::runtime_error("读取输入文件失败: " + path);
    }
    return values;
}

void write_i32_file(const std::string& path, const std::vector<std::int32_t>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("无法创建输出文件: " + path);
    }
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(std::int32_t)));
    if (!stream) {
        throw std::runtime_error("写入输出文件失败: " + path);
    }
    stream.flush();
}

std::size_t verify(const std::vector<std::int32_t>& a,
                   const std::vector<std::int32_t>& b,
                   const std::vector<std::int32_t>& c) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const auto expected = static_cast<std::int64_t>(a[i]) + static_cast<std::int64_t>(b[i]);
        if (expected != c[i]) {
            if (mismatches < 8) {
                std::cerr << "不匹配 i=" << i << " a=" << a[i] << " b=" << b[i]
                          << " expected=" << expected << " actual=" << c[i] << '\n';
            }
            ++mismatches;
        }
    }
    return mismatches;
}

std::vector<std::int32_t> run_kernel(xrt::device& device,
                                     xrt::kernel& kernel,
                                     const std::vector<std::int32_t>& a,
                                     const std::vector<std::int32_t>& b,
                                     Timing& timing) {
    if (a.empty() || a.size() != b.size() || a.size() > kMaximumElements) {
        throw std::runtime_error("A/B长度必须相同，并且元素数在1到16777216之间");
    }
    const std::size_t bytes = a.size() * sizeof(std::int32_t);
    xrt::bo bo_a(device, bytes, kernel.group_id(0));
    xrt::bo bo_b(device, bytes, kernel.group_id(1));
    xrt::bo bo_c(device, bytes, kernel.group_id(2));

    auto* mapped_a = bo_a.map<std::int32_t*>();
    auto* mapped_b = bo_b.map<std::int32_t*>();
    auto* mapped_c = bo_c.map<std::int32_t*>();
    std::copy(a.begin(), a.end(), mapped_a);
    std::copy(b.begin(), b.end(), mapped_b);
    std::fill(mapped_c, mapped_c + a.size(), 0);

    auto begin = std::chrono::steady_clock::now();
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto end = std::chrono::steady_clock::now();
    timing.host_to_device_ms = elapsed_ms(begin, end);

    begin = std::chrono::steady_clock::now();
    auto run = kernel(bo_a, bo_b, bo_c, static_cast<std::uint32_t>(a.size()));
    run.wait();
    end = std::chrono::steady_clock::now();
    timing.kernel_ms = elapsed_ms(begin, end);

    begin = std::chrono::steady_clock::now();
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    end = std::chrono::steady_clock::now();
    timing.device_to_host_ms = elapsed_ms(begin, end);

    return std::vector<std::int32_t>(mapped_c, mapped_c + a.size());
}

bool receive_all(int fd, void* buffer, std::size_t bytes) {
    auto* cursor = static_cast<std::uint8_t*>(buffer);
    while (bytes != 0) {
        const auto received = ::recv(fd, cursor, bytes, 0);
        if (received == 0) {
            return false;
        }
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("TCP接收失败");
        }
        cursor += received;
        bytes -= static_cast<std::size_t>(received);
    }
    return true;
}

void send_all(int fd, const void* buffer, std::size_t bytes) {
    const auto* cursor = static_cast<const std::uint8_t*>(buffer);
    while (bytes != 0) {
        const auto sent = ::send(fd, cursor, bytes, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("TCP发送失败");
        }
        cursor += sent;
        bytes -= static_cast<std::size_t>(sent);
    }
}

void print_timing(const Timing& timing) {
    std::cout << "Host->Device: " << timing.host_to_device_ms << " ms\n"
              << "PL kernel:     " << timing.kernel_ms << " ms\n"
              << "Device->Host: " << timing.device_to_host_ms << " ms\n";
}

int run_sd_mode(xrt::device& device,
                xrt::kernel& kernel,
                const std::string& input_a,
                const std::string& input_b,
                const std::string& output) {
    const auto a = read_i32_file(input_a);
    const auto b = read_i32_file(input_b);
    if (a.size() != b.size()) {
        throw std::runtime_error("input_a.bin和input_b.bin元素数量不同");
    }
    Timing timing;
    const auto c = run_kernel(device, kernel, a, b, timing);
    const auto mismatches = verify(a, b, c);
    write_i32_file(output, c);
    std::cout << "元素数: " << c.size() << "，错误元素: " << mismatches << '\n';
    print_timing(timing);
    std::cout << (mismatches == 0 ? "TEST PASSED\n" : "TEST FAILED\n");
    return mismatches == 0 ? 0 : 2;
}

void serve_one_client(int client, xrt::device& device, xrt::kernel& kernel) {
    std::uint32_t request[4]{};
    if (!receive_all(client, request, sizeof(request))) {
        return;
    }
    const auto magic = ntohl(request[0]);
    const auto version = ntohl(request[1]);
    const auto count = ntohl(request[2]);
    const auto request_id = ntohl(request[3]);
    if (magic != kRequestMagic || version != kProtocolVersion || count == 0 ||
        count > kMaximumElements) {
        throw std::runtime_error("TCP请求头非法");
    }

    std::vector<std::int32_t> a(count);
    std::vector<std::int32_t> b(count);
    if (!receive_all(client, a.data(), a.size() * sizeof(std::int32_t)) ||
        !receive_all(client, b.data(), b.size() * sizeof(std::int32_t))) {
        throw std::runtime_error("TCP输入数据不完整");
    }

    Timing timing;
    const auto c = run_kernel(device, kernel, a, b, timing);
    const auto mismatches = verify(a, b, c);
    std::uint32_t response[5]{htonl(kResponseMagic),
                              htonl(kProtocolVersion),
                              htonl(count),
                              htonl(request_id),
                              htonl(mismatches == 0 ? 0U : 1U)};
    send_all(client, response, sizeof(response));
    send_all(client, c.data(), c.size() * sizeof(std::int32_t));
    std::cout << "request=" << request_id << " count=" << count
              << " mismatches=" << mismatches << '\n';
    print_timing(timing);
}

int run_server_mode(xrt::device& device, xrt::kernel& kernel, std::uint16_t port) {
    const int server = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) {
        throw std::runtime_error("创建TCP socket失败");
    }
    int reuse = 1;
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(server, 4) != 0) {
        ::close(server);
        throw std::runtime_error("监听TCP端口失败");
    }
    std::cout << "KV260 vadd服务器正在监听0.0.0.0:" << port << "，Ctrl+C停止\n";
    while (true) {
        const int client = ::accept(server, nullptr, nullptr);
        if (client < 0) {
            if (errno == EINTR) {
                continue;
            }
            ::close(server);
            throw std::runtime_error("接受TCP连接失败");
        }
        try {
            serve_one_client(client, device, kernel);
        } catch (const std::exception& error) {
            std::cerr << "本次请求失败: " << error.what() << '\n';
        }
        ::close(client);
    }
}

void usage(const char* program) {
    std::cerr << "用法:\n"
              << "  " << program << " <xclbin> sd <input_a.bin> <input_b.bin> <output.bin>\n"
              << "  " << program << " <xclbin> server [port]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            usage(argv[0]);
            return 1;
        }
        xrt::device device(0);
        const auto uuid = device.load_xclbin(argv[1]);
        xrt::kernel kernel(device, uuid, "krnl_vadd");
        const std::string mode = argv[2];
        if (mode == "sd" && argc == 6) {
            return run_sd_mode(device, kernel, argv[3], argv[4], argv[5]);
        }
        if (mode == "server" && (argc == 3 || argc == 4)) {
            const int port = argc == 4 ? std::stoi(argv[3]) : 5000;
            if (port <= 0 || port > 65535) {
                throw std::runtime_error("端口必须在1到65535之间");
            }
            return run_server_mode(device, kernel, static_cast<std::uint16_t>(port));
        }
        usage(argv[0]);
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << '\n';
        return 1;
    }
}
