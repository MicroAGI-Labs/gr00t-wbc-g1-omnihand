#include "audio_thread.hpp"
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>

static const std::string PLANNER_MODE = "Planner mode";
static const std::string POSE_MODE = "Pose mode";
static const std::string WARNING_STREAMING_DATA_ABSENT = "Streaming data absent";
static const std::string WARNING_MOTOR_ERROR = "Motor error detected";
static const std::string WARNING_LOW_STATE_LATE = "ROBOT DATA LATE";
static const std::string RECORDING_STARTED = "Recording started";
static const std::string RECORDING_DISCARDED = "Recording discarded";
static const std::string RECORDING_SAVED = "Recording saved";
static const std::string RECORDING_VALIDATION_FAILED = "Recording failed validation";
static const std::string RECORDING_SAVE_FAILED = "Recording save failed";

AudioThread::AudioThread():
  client_(),
  recording_status_context_(1) {
  client_.Init();
  client_.SetTimeout(10.0f);
  client_.SetVolume(100);

  try {
    recording_status_socket_ =
      std::make_unique<zmq::socket_t>(recording_status_context_, zmq::socket_type::sub);
    recording_status_socket_->set(zmq::sockopt::linger, 0);
    recording_status_socket_->set(zmq::sockopt::rcvhwm, 100);
    recording_status_socket_->set(zmq::sockopt::subscribe, "");
    recording_status_socket_->connect("tcp://127.0.0.1:5581");
    std::cout << "[AudioThread] Listening for recording events on port 5581" << std::endl;
  } catch (const zmq::error_t& error) {
    std::cerr << "[AudioThread] Recording status connection failed: "
              << error.what() << std::endl;
    recording_status_socket_.reset();
  }

  thread_ = std::jthread([this](std::stop_token st) { loop(st); });
}

void AudioThread::SetCommand(const AudioCommand& command) {
  std::lock_guard<std::mutex> lock(command_mutex_);
  std::string prev_tts = std::move(command_.tts_message);
  command_ = command;

  if (command_.tts_message.empty()) {
    // No new tts, keep pending
    command_.tts_message = std::move(prev_tts);
  } else if (!prev_tts.empty()) {
    // Both have tts, concatenate
    command_.tts_message = prev_tts + ". " + command_.tts_message;
  }
}

void AudioThread::PollRecordingStatus() {
  if (!recording_status_socket_) {
    return;
  }

  try {
    while (true) {
      zmq::message_t message;
      const auto received =
        recording_status_socket_->recv(message, zmq::recv_flags::dontwait);
      if (!received) {
        return;
      }

      const std::string payload(
        static_cast<const char*>(message.data()), message.size());
      const auto status = nlohmann::json::parse(payload, nullptr, false);
      if (status.is_discarded()) {
        continue;
      }

      const auto sequence =
        status.value("recording_audio_sequence", std::uint64_t{0});
      const auto event =
        status.value("recording_audio_event", std::string{});
      if (event.empty() || sequence <= last_recording_audio_sequence_) {
        continue;
      }
      last_recording_audio_sequence_ = sequence;

      if (event == "start") {
        client_.TtsMaker(RECORDING_STARTED, 1);
      } else if (event == "saved") {
        client_.TtsMaker(RECORDING_SAVED, 1);
      } else if (event == "discard") {
        client_.TtsMaker(RECORDING_DISCARDED, 1);
      } else if (event == "validation_failed") {
        client_.TtsMaker(RECORDING_VALIDATION_FAILED, 1);
      } else if (event == "save_failed") {
        client_.TtsMaker(RECORDING_SAVE_FAILED, 1);
      }
    }
  } catch (const zmq::error_t& error) {
    std::cerr << "[AudioThread] Recording status receive failed: "
              << error.what() << std::endl;
    recording_status_socket_.reset();
  } catch (const nlohmann::json::exception& error) {
    std::cerr << "[AudioThread] Invalid recording status: "
              << error.what() << std::endl;
  }
}

void AudioThread::loop(std::stop_token st) {
  while (!st.stop_requested()) {
    PollRecordingStatus();
    AudioCommand command;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      command = command_;
      // Clear one-shot TTS so it's only spoken once
      command_.tts_message.clear();
    }
    if (command.streaming_data_absent && !command_last_.streaming_data_absent) {
      client_.TtsMaker(WARNING_STREAMING_DATA_ABSENT, 1);
    }
    if (command.motor_error && !command_last_.motor_error) {
      client_.TtsMaker(WARNING_MOTOR_ERROR, 1);
    }
    if (!command.tts_message.empty()) {
      client_.TtsMaker(command.tts_message, 1);
    }
    if (command.high_temperature && !command.high_temperature_message.empty()) {
      auto now = std::chrono::steady_clock::now();
      if (now - last_high_temp_tts_ >= HIGH_TEMP_TTS_INTERVAL) {
        client_.TtsMaker(command.high_temperature_message, 1);
        last_high_temp_tts_ = now;
      }
    }
    if (command.low_state_late && !command_last_.low_state_late) {
      client_.TtsMaker(WARNING_LOW_STATE_LATE, 1);
    }

    command_last_ = command;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}
