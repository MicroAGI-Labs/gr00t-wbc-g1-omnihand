// Persistent M4010 torque controller, adapted from tools/dex1_timed.cpp.
// stdin: C <sequence> <mode: 0=hold, 1=track, 2=stop> <output radians>\n
// stdout: versioned JSON feedback. The SDK's diagnostics go to stderr.
#include "serialPort/SerialPort.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
constexpr float kTorque = 1.0f, kSpeed = 9.0f, kP = 8.0f, kD = .35f;
constexpr double kWatchdog = .25;
void on_signal(int) { interrupted = 1; }
void check(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
struct CommunicationError : std::runtime_error { using std::runtime_error::runtime_error; };
double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

struct Reference { float q, dq; };
Reference reference(float start, float end, float t, float duration) {
    const float u = std::clamp(t / duration, 0.0f, 1.0f);
    return {start + (end-start)*u*u*u*(10+u*(-15+6*u)),
            (end-start)*30*u*u*(1-u)*(1-u)/duration};
}
float torque(Reference r, float q, float dq, float dt, float& integral) {
    const float error = r.q-q;
    const float candidate = std::clamp(integral+4*error*dt, -.30f, .30f);
    const float pd = kP*error + kD*(r.dq-dq);
    if (std::abs(pd+candidate) <= kTorque || error*(pd+candidate) < 0) integral = candidate;
    return std::clamp(pd+integral, -kTorque, kTorque);
}

MotorCmd command(int id, int mode, float tau = 0) {
    // Upstream constructors leave fields uninitialized. Initialize every input
    // explicitly; no dependency on the locally patched SDK header.
    MotorCmd c;
    c.motorType = MotorType::M4010; c.hex_len = 0; c.id = id; c.mode = mode;
    c.tau = tau/25; c.q = c.dq = c.kp = c.kd = 0;
    c.timeout = 0; c.Res = {}; c.res = 0;
    return c;
}

struct Request { unsigned long long sequence = 0; int mode = 0; float q = 0; };
Request parse(const std::string& line, float lower, float upper) {
    Request r; std::string tag, extra;
    std::istringstream input(line);
    check(bool(input >> tag >> r.sequence >> r.mode >> r.q) && !(input >> extra)
          && tag == "C" && r.sequence > 0 && r.mode >= 0 && r.mode <= 2
          && std::isfinite(r.q) && r.q >= lower && r.q <= upper, "invalid control request");
    return r;
}

struct Motor {
    int id;
    std::string port_name;
    float lower, upper;
    std::unique_ptr<SerialPort> serial;
    MotorData data;
    int lock_fd = -1;
    Clock::time_point last_reply = Clock::now();
    float q() const { return data.q/25; }
    float dq() const { return data.dq/25; }
    float tau() const { return data.tau*25; }
    float voltage() { return data.get_motor_recv_data()[5]/2.0f; }

    Motor(const std::string& port, int motor_id, float lo, float hi)
        : id(motor_id), port_name(port), lower(lo), upper(hi) {
        // Kernel exclusivity prevents new opens by other services/test tools.
        lock_fd = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        check(lock_fd >= 0, "cannot open " + port);
        if (flock(lock_fd, LOCK_EX | LOCK_NB)) {
            ::close(lock_fd); lock_fd = -1;
            throw std::runtime_error("serial device is already owned: " + port);
        }
        // Open the SDK handle before taking TIOCEXCL, retaining our flock
        // against another worker while the SDK opens its own descriptor.
        try {
            serial = std::make_unique<SerialPort>(port, 26, 6000000, 20000);
            check(ioctl(lock_fd, TIOCEXCL) == 0, "cannot claim serial device");
        } catch (...) { ::close(lock_fd); lock_fd = -1; throw; }
    }
    void exchange(int mode, float tau_command = 0, bool enabling = false) {
        check(!interrupted, "interrupted");
        check(std::isfinite(tau_command) && std::abs(tau_command) <= kTorque, "torque command limit");
        if (mode == 1) check(seconds(Clock::now()-last_reply) < .060, "feedback watchdog");
        MotorData next;
        next.motorType = MotorType::M4010; next.correct = false;
        auto c = command(id, mode, tau_command);
        if (!serial->sendRecv(&c, &next) || !next.correct || next.motor_id != id)
            throw CommunicationError("invalid motor response: port=" + port_name + " motor_id=" + std::to_string(id));
        data = next; last_reply = Clock::now();
        check(std::isfinite(q()) && std::isfinite(dq()) && std::isfinite(tau()), "nonfinite feedback");
        check(data.merror == 0, "motor fault " + std::to_string(data.merror));
        check(voltage() >= 24 && voltage() <= 64, "supply outside 24-64 V");
        check(data.temp < 55 && data.get_motor_recv_data()[4] < 80, "temperature limit");
        check(q() >= lower && q() <= upper, "position limit");
        check(std::abs(dq()) < kSpeed && std::abs(tau()) < 1.30f, "speed/torque limit");
        if (mode == 1 && !enabling) check(data.mode == 1 && data.timeout == 0, "drive not enabled");
    }
    void establish_feedback() {
        // USB enumeration does not mean the motor is ready to reply yet.
        // Retry only this initial, disabled handshake. Never retry a health
        // violation or a communication fault after this handshake succeeds.
        const auto deadline = Clock::now() + std::chrono::seconds(1);
        int retries = 0;
        while (true) {
            try {
                exchange(0);
                if (retries)
                    std::cerr << "DEX1 startup recovered: motor_id=" << id << " retries=" << retries << '\n';
                return;
            } catch (const CommunicationError& e) {
                if (Clock::now() >= deadline)
                    throw CommunicationError(std::string(e.what()) +
                        "; startup timed out with drive disabled; check motor power and data cable, then reconnect");
                ++retries;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
    }
    void stop() noexcept {
        if (!serial) return;
        for (int n = 0; n < 3; ++n) {
            try { auto c = command(id, 0); MotorData d; d.motorType = MotorType::M4010;
                  serial->sendRecv(&c, &d); } catch (...) {}
        }
    }
    ~Motor() {
        stop(); serial.reset();
        if (lock_fd >= 0) { ioctl(lock_fd, TIOCNXCL); ::close(lock_fd); }
    }
};

void feedback(Motor& m, Reference r, int mode, unsigned long long sequence) {
    char line[512];
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    const int size = std::snprintf(line, sizeof(line),
        "{\"version\":1,\"monotonic_ns\":%lld,\"sequence\":%llu,\"mode\":%d,"
        "\"q\":%.7g,\"dq\":%.7g,\"tau\":%.7g,\"applied_q\":%.7g,"
        "\"voltage\":%.7g,\"temperature\":%d,\"motor_error\":%d}\n",
        static_cast<long long>(ns), sequence, mode, m.q(), m.dq(), m.tau(), r.q,
        m.voltage(), m.data.temp, m.data.merror);
    // Less than PIPE_BUF: nonblocking atomic write, drop a snapshot if full.
    const auto written = ::write(STDOUT_FILENO, line, size);
    check(written == size || (written < 0 && (errno == EAGAIN || errno == EINTR)), "feedback pipe closed");
}

void self_test() {
    check(queryGearRatio(MotorType::M4010) == 25, "gear ratio");
    for (const auto& ends : {std::pair<float,float>{.12f,5.30f}, {-2.36f,2.84f}, {5.30f,.12f}}) {
        check(std::abs(reference(ends.first, ends.second, 0, 1.5).q-ends.first) < 1e-5, "ramp start");
        check(std::abs(reference(ends.first, ends.second, 1.5, 1.5).q-ends.second) < 1e-5, "ramp end");
        for (int tick = 0; tick <= 300; ++tick) {
            const auto r = reference(ends.first, ends.second, tick*.005f, 1.5);
            check(r.q >= std::min(ends.first,ends.second)-1e-5
                  && r.q <= std::max(ends.first,ends.second)+1e-5 && std::abs(r.dq) < kSpeed, "ramp limits");
        }
    }
    float integral = 0;
    for (int tick = 0; tick < 10000; ++tick)
        check(torque({.12f,0}, 3, 0, .005f, integral) == -kTorque && integral == 0, "grasp cap/antiwindup");
    check(torque({5.3f,0}, 3, 0, .005f, integral) > 0, "release after contact");
    for (int id : {0,1}) for (float tau : {-kTorque,0.0f,kTorque}) {
        auto c = command(id, 1, tau); c.modify_data(&c);
        auto* p = c.get_motor_send_data(); int16_t wire; std::memcpy(&wire,p+4,2);
        check(c.hex_len == 20 && (p[2]&15) == id && ((p[2]>>4)&7) == 1, "packet address");
        check(std::abs(wire*25.0f/2560) <= kTorque+.01f && (tau == 0 ? wire == 0 : tau*wire > 0), "wire torque");
        for (int i = 6; i < 16; ++i) check(p[i] == 0, "wire position/gains must be zero");
    }
    check(parse("C 1 1 2.5", -.1, 5.75).mode == 1, "control parser");
    for (const std::string bad : {"C 1 1 nan", "C 1 3 1", "C 1 1 6", "C 1 1 2 extra"}) {
        bool rejected = false; try { parse(bad,-.1,5.75); } catch (...) { rejected = true; }
        check(rejected, "invalid request admitted");
    }
    std::cerr << "DEX1_SELF_TEST_PASS: ramps, grasp/release, packets, command validation\n";
}

int main(int argc, char** argv) {
    std::cout.rdbuf(std::cerr.rdbuf());
    std::signal(SIGINT,on_signal); std::signal(SIGTERM,on_signal); std::signal(SIGHUP,on_signal);
    std::signal(SIGPIPE,SIG_IGN);
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        check(argc == 7, "usage: dex1_worker PORT ID LOWER UPPER DURATION ENABLE");
        const int id = std::stoi(argv[2]);
        const float lower = std::stof(argv[3]), upper = std::stof(argv[4]), duration = std::stof(argv[5]);
        const bool enabled = std::string(argv[6]) == "1";
        check((id == 0 || id == 1) && std::isfinite(lower) && std::isfinite(upper)
              && lower < upper && upper-lower < 6 && std::isfinite(duration)
              && duration >= 1.35 && duration <= 30, "invalid motor configuration");
        check(fcntl(STDIN_FILENO,F_SETFL,O_NONBLOCK) == 0
              && fcntl(STDOUT_FILENO,F_SETFL,O_NONBLOCK) == 0, "pipe flags");
        Motor m(argv[1],id,lower,upper); m.establish_feedback();
        Request request; request.q = m.q();
        auto last_command = Clock::now(), previous = last_command, next = last_command, ramp_start = last_command;
        float start = m.q(), target = start, integral = 0;
        Reference ref{start,0};
        int mode = -1, enabling_ticks = 0, tick = 0;
        std::string pending;
        while (!interrupted) {
            const auto now = Clock::now();
            check(seconds(now-previous) < .060, "control loop delayed");
            const float dt = std::clamp(float(seconds(now-previous)),0.0f,.020f); previous = now;
            char bytes[4096]; const auto count = ::read(STDIN_FILENO,bytes,sizeof(bytes));
            if (count == 0) break;
            check(count >= 0 || errno == EAGAIN || errno == EINTR, "control pipe read");
            if (count > 0) pending.append(bytes,count);
            check(pending.size() <= 4096, "control backlog");
            auto latest = request;
            for (size_t newline; (newline = pending.find('\n')) != std::string::npos;) {
                auto incoming = parse(pending.substr(0,newline),lower,upper); pending.erase(0,newline+1);
                check(incoming.sequence > latest.sequence, "replayed control request");
                latest = incoming;
            }
            if (latest.sequence > request.sequence) {
                last_command = now;
                if (latest.mode == 2) break;
                check(enabled, "motor commands disabled");
                if (mode != latest.mode || (latest.mode == 1 && latest.q != target)) {
                    // Latest target replaces the old ramp immediately. Holds
                    // freeze measured position once, without completing a close.
                    start = m.q(); target = latest.mode == 0 ? start : latest.q;
                    integral = 0; ramp_start = now;
                }
                if (mode < 0) enabling_ticks = 10;
                mode = latest.mode; request = latest;
            }
            check(seconds(now-last_command) < (mode < 0 ? 5.0 : kWatchdog), "controller watchdog");
            ref = mode == 1 ? reference(start,target,seconds(now-ramp_start),duration) : Reference{target,0};
            if (mode < 0) m.exchange(0);
            else if (enabling_ticks > 0) {
                m.exchange(1,0,true); --enabling_ticks;
                if (m.data.mode == 1 && m.data.timeout == 0) { enabling_ticks = 0; ramp_start = now; }
                else check(enabling_ticks > 0, "cannot enable drive");
            } else {
                // Closing resistance is expected when grasping. Opening must
                // still follow its bounded trajectory instead of forcing a jam.
                if (mode == 1 && target > start)
                    check(std::abs(ref.q-m.q()) < .60f, "opening stalled");
                m.exchange(1,torque(ref,m.q(),m.dq(),dt,integral));
            }
            if (tick++ % 4 == 0) feedback(m,ref,mode,request.sequence);
            next += std::chrono::milliseconds(5);
            if (next < Clock::now()) next = Clock::now();
            std::this_thread::sleep_until(next);
        }
        return 0; // Motor destructor sends stop packets, including on exceptions.
    } catch (const std::exception& e) {
        std::cerr << "DEX1 FAULT: " << e.what() << '\n';
        return 2; // Requires explicit UI reconnect; never silently re-enable.
    }
}
