#include <gtest/gtest.h>

#include "input_interface/zmq_manager.hpp"

class ZMQManagerTestPeer {
public:
    static void receive(ZMQManager& manager,
                        const ZMQPackedMessageSubscriber::DecodedHeader& header,
                        const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers) {
        manager.OnPlannerReceived("test_planner", header, buffers);
    }
    static void expire(ZMQManager& manager) {
        manager.latest_planner_message_.timestamp =
            std::chrono::steady_clock::now() - std::chrono::seconds(20);
    }
    static void heldFor(ZMQManager& manager, double seconds) {
        ASSERT_TRUE(manager.pico_lost_at_.has_value());
        manager.advancePicoDisconnectReturn(*manager.pico_lost_at_ +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds)));
    }
};

// Exercise the actual decoder and timeout path without a robot or publisher.
// Subscribers use test-only topics on localhost; input is injected in-process.
TEST(PlannerDisconnect, KeepsVrTargetsAndEncoderAcrossTimeoutAndReconnect) {
    ZMQManager manager("127.0.0.1", 1, "test_pose", "test_command", "test_planner");
    MotionDataReader reader;
    auto motion = std::make_shared<MotionSequence>();
    motion->ReserveCapacity(1);
    motion->timesteps = 1;
    motion->SetEncodeMode(0);
    // A deliberately unrelated planner reference must never replace VR control.
    std::fill_n(motion->JointPositions(0), 29, 2.0);
    std::shared_ptr<const MotionSequence> current_motion = motion;
    int frame = 0;
    OperatorState operator_state;
    PlannerState planner_state{true, true};
    bool reinitialize_heading = false;
    DataBuffer<HeadingState> heading;
    DataBuffer<MovementState> movement;
    movement.SetData(MovementState{});
    std::mutex motion_mutex;
    bool report_temperature = false;
    auto tick = [&] {
        manager.handle_input(reader, current_motion, frame, operator_state,
                             reinitialize_heading, heading, true, planner_state,
                             movement, motion_mutex, report_temperature);
    };

    std::array<double, 9> position{0.3, 0.2, 0.1, 0.3, -0.2, 0.1, 0, 0, 0.4};
    std::array<double, 12> orientation{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    std::array<double, 14> base{0.1, 0.15, -0.1, 1, 0, 0, 0, 0.1, -0.15, -0.1, 1, 0, 0, 0};
    int32_t mode = static_cast<int32_t>(LocomotionMode::WALK);
    std::array<double, 3> direction{1, 0, 0};
    ZMQPackedMessageSubscriber::DecodedHeader header;
    header.fields = {{"mode", "i32", {1}}, {"movement", "f64", {3}},
                     {"facing", "f64", {3}}, {"vr_position", "f64", {9}},
                     {"vr_orientation", "f64", {12}}, {"vr_base_pose", "f64", {14}}};
    auto view = [](const auto& value) {
        return ZMQPackedMessageSubscriber::BufferView{&value, sizeof(value)};
    };
    auto send_vr = [&] {
        ZMQManagerTestPeer::receive(manager, header,
            {view(mode), view(direction), view(direction), view(position), view(orientation), view(base)});
        tick();
    };
    auto expect_vr_hold = [&] {
        EXPECT_TRUE(manager.HasVR3PointControl());
        EXPECT_EQ(manager.GetVR3PointPosition().second, position);
        EXPECT_EQ(manager.GetVR3PointOrientation().second, orientation);
        EXPECT_FALSE(manager.GetUpperBodyJointPositions().first);
        EXPECT_EQ(motion->GetEncodeMode(), 1);
        EXPECT_FALSE(operator_state.stop);
    };

    // An earlier explicit joint hold must release when valid VR control arrives.
    auto joint_header = header;
    joint_header.fields.resize(3);
    joint_header.fields.push_back({"upper_body_position", "f64", {17}});
    std::array<double, 17> joint_position{};
    joint_position.fill(0.4);
    ZMQManagerTestPeer::receive(manager, joint_header,
        {view(mode), view(direction), view(direction), view(joint_position)});
    tick();
    ZMQManagerTestPeer::expire(manager);
    tick();
    ASSERT_TRUE(manager.GetUpperBodyJointPositions().first);
    EXPECT_EQ(manager.GetUpperBodyJointPositions().second, joint_position);
    // Older publishers without a base target keep holding indefinitely.
    auto base_field = header.fields.back();
    header.fields.pop_back();
    send_vr();
    ZMQManagerTestPeer::expire(manager);
    tick();
    ZMQManagerTestPeer::heldFor(manager, 1000.0);
    expect_vr_hold();
    header.fields.push_back(base_field);
    send_vr();
    const auto valid_base = base;
    base[0] = std::numeric_limits<double>::quiet_NaN();
    send_vr();  // A malformed update must not replace the cached base target.
    base = valid_base;
    expect_vr_hold();
    ASSERT_EQ(movement.GetDataWithTime().data->locomotion_mode,
              static_cast<int>(LocomotionMode::WALK));
    ZMQManagerTestPeer::expire(manager);
    tick();
    expect_vr_hold();
    EXPECT_EQ(movement.GetDataWithTime().data->locomotion_mode,
              static_cast<int>(LocomotionMode::IDLE));
    EXPECT_EQ(movement.GetDataWithTime().data->movement_direction,
              (std::array<double, 3>{0, 0, 0}));
    // Repeated ticks after the timeout remain latched, even without new input.
    for (int i = 0; i < 2000; ++i) tick();
    expect_vr_hold();
    EXPECT_EQ(manager.TakeStatusAnnouncement(), "Pico connection lost. Holding position.");
    EXPECT_TRUE(manager.TakeStatusAnnouncement().empty());
    for (double elapsed : {1.0, 14.99, 15.0}) {
        ZMQManagerTestPeer::heldFor(manager, elapsed);
        expect_vr_hold();
        EXPECT_TRUE(manager.TakeStatusAnnouncement().empty());
    }
    ZMQManagerTestPeer::heldFor(manager, 17.5);
    auto halfway = manager.GetVR3PointPosition().second;
    for (size_t side = 0; side < 2; ++side) {
        for (size_t axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(halfway[3 * side + axis],
                        (position[3 * side + axis] + base[7 * side + axis]) / 2, 1e-10);
        }
    }
    EXPECT_EQ(manager.TakeStatusAnnouncement(), "Pico still disconnected. Returning arms to rest on legs.");
    ZMQManagerTestPeer::heldFor(manager, 20.0);
    for (size_t side = 0; side < 2; ++side) {
        for (size_t axis = 0; axis < 3; ++axis) position[3 * side + axis] = base[7 * side + axis];
    }
    expect_vr_hold();
    ZMQManagerTestPeer::heldFor(manager, 1000.0);
    expect_vr_hold();
    EXPECT_TRUE(manager.TakeStatusAnnouncement().empty());
    // Reconnecting with the held target must not install a stale joint override.
    mode = static_cast<int32_t>(LocomotionMode::IDLE);
    send_vr();
    expect_vr_hold();
    EXPECT_EQ(manager.TakeStatusAnnouncement(), "Pico connection restored. Arms held.");
    position[0] += 0.01;
    send_vr();
    expect_vr_hold();
    // An explicit joint command can still deliberately leave VR control.
    ZMQManagerTestPeer::receive(manager, joint_header,
        {view(mode), view(direction), view(direction), view(joint_position)});
    tick();
    EXPECT_FALSE(manager.HasVR3PointControl());
    EXPECT_TRUE(manager.GetUpperBodyJointPositions().first);
    EXPECT_EQ(motion->GetEncodeMode(), 0);

    // Custom limits are cached on the robot and constrain a large autonomous
    // return even when there are no more publisher messages.
    header.fields.push_back({"vr_motion_limits", "f64", {4}});
    for (bool translation_dominates : {true, false}) {
        std::array<double, 4> limits{0.2, 0.5, 0.2, 0.4};
        position[0] = base[0] + (translation_dominates ? 1.2 : 0.0);
        orientation[0] = std::sqrt(0.5);
        orientation[3] = std::sqrt(0.5);
        base[3] = std::cos(M_PI / 12);
        base[6] = std::sin(M_PI / 12);
        ZMQManagerTestPeer::receive(manager, header,
            {view(mode), view(direction), view(direction), view(position), view(orientation), view(base), view(limits)});
        tick();
        ZMQManagerTestPeer::expire(manager);
        tick();
        auto previous_position = manager.GetVR3PointPosition().second;
        auto previous_orientation = manager.GetVR3PointOrientation().second;
        Eigen::Vector3d previous_v = Eigen::Vector3d::Zero();
        Eigen::Vector3d previous_w = Eigen::Vector3d::Zero();
        for (int i = 1; i <= 2000; ++i) {
            ZMQManagerTestPeer::heldFor(manager, 15.0 + i * 0.01);
            auto p = manager.GetVR3PointPosition().second;
            auto q = manager.GetVR3PointOrientation().second;
            Eigen::Vector3d v((p[0] - previous_position[0]) / 0.01,
                              (p[1] - previous_position[1]) / 0.01,
                              (p[2] - previous_position[2]) / 0.01);
            // This trajectory rotates only around Z. Measure its signed angle
            // independently of Eigen quaternion conjugation and interpolation.
            Eigen::Vector3d w(0, 0, 2 * (std::atan2(q[3], q[0]) -
                std::atan2(previous_orientation[3], previous_orientation[0])) / 0.01);
            EXPECT_LE(v.norm(), limits[0] + 1e-6);
            EXPECT_LE((v - previous_v).norm() / 0.01, limits[1] + 1e-5);
            ASSERT_LE(w.norm(), limits[2] + 1e-6);
            EXPECT_LE((w - previous_w).norm() / 0.01, limits[3] + 1e-5);
            previous_position = p; previous_orientation = q;
            previous_v = v; previous_w = w;
        }
        EXPECT_DOUBLE_EQ(previous_position[0], base[0]);
        EXPECT_NEAR(previous_orientation[0], base[3], 1e-12);
        EXPECT_NEAR(previous_orientation[3], base[6], 1e-12);
        EXPECT_EQ(motion->GetEncodeMode(), 1);
    }
}
