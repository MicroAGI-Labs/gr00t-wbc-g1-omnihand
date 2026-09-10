// Test-only stationary motor: contact at 2.5 output radians. No motor I/O.
#pragma once
#ifndef DEX1_FAKE_MOTOR_FOR_TEST
#error "The fake motor header is only for explicitly compiled offline tests"
#endif
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

enum class MotorType { M4010 };
inline float queryGearRatio(MotorType) { return 25; }
struct MotorCmd {
    MotorType motorType;
    int hex_len = 0, id = 0, mode = 0, timeout = 0, Res = 0, res = 0;
    float tau = 0, q = 0, dq = 0, kp = 0, kd = 0;
    uint8_t bytes[20]{};
    void modify_data(MotorCmd*) {
        hex_len = 20; bytes[2] = id | (mode<<4);
        int16_t wire = tau*2560; std::memcpy(bytes+4,&wire,2);
    }
    uint8_t* get_motor_send_data() { return bytes; }
};
struct MotorData {
    MotorType motorType;
    bool correct = false;
    int motor_id = 0, mode = 0, timeout = 0, merror = 0, temp = 30;
    float q = 62.5f, dq = 0, tau = 0;
    uint8_t bytes[26]{};
    uint8_t* get_motor_recv_data() { return bytes; }
};
struct SerialPort {
    int calls = 0;
    SerialPort(const std::string&, int, int, int) {}
    bool sendRecv(MotorCmd* c, MotorData* d) {
        ++calls;
        if (const auto* log = std::getenv("DEX1_TEST_EXCHANGE_LOG"))
            std::ofstream(log,std::ios::app) << c->mode << '\n';
        if (const auto* failures = std::getenv("DEX1_TEST_STARTUP_FAILURES"))
            if (calls <= std::stoi(failures)) return false;
        if (std::getenv("DEX1_TEST_ACTIVE_FAILURE") && c->mode == 1) return false;
        d->correct = true; d->motor_id = c->id; d->mode = c->mode;
        d->tau = c->tau; d->bytes[5] = 50; d->bytes[4] = 30;
        if (const auto* speed = std::getenv("DEX1_TEST_FEEDBACK_SPEED")) d->dq = std::stof(speed)*25;
        if (const auto* torque = std::getenv("DEX1_TEST_FEEDBACK_TORQUE")) d->tau = std::stof(torque)/25;
        if (std::getenv("DEX1_TEST_LOW_VOLTAGE")) d->bytes[5] = 20;
        if (const auto* log = std::getenv("DEX1_TEST_MOTOR_LOG"))
            std::ofstream(log,std::ios::app) << c->mode << ' ' << c->tau*25 << '\n';
        return true;
    }
};
