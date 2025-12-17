// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/robot_state_rt__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_RobotStateRt_reserved
{
public:
  explicit Init_RobotStateRt_reserved(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::RobotStateRt reserved(::dsr_msgs2::msg::RobotStateRt::_reserved_type arg)
  {
    msg_.reserved = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_control_mode
{
public:
  explicit Init_RobotStateRt_control_mode(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_reserved control_mode(::dsr_msgs2::msg::RobotStateRt::_control_mode_type arg)
  {
    msg_.control_mode = std::move(arg);
    return Init_RobotStateRt_reserved(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_robot_state
{
public:
  explicit Init_RobotStateRt_robot_state(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_control_mode robot_state(::dsr_msgs2::msg::RobotStateRt::_robot_state_type arg)
  {
    msg_.robot_state = std::move(arg);
    return Init_RobotStateRt_control_mode(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_robot_mode
{
public:
  explicit Init_RobotStateRt_robot_mode(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_robot_state robot_mode(::dsr_msgs2::msg::RobotStateRt::_robot_mode_type arg)
  {
    msg_.robot_mode = std::move(arg);
    return Init_RobotStateRt_robot_state(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_goal_tcp_position
{
public:
  explicit Init_RobotStateRt_goal_tcp_position(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_robot_mode goal_tcp_position(::dsr_msgs2::msg::RobotStateRt::_goal_tcp_position_type arg)
  {
    msg_.goal_tcp_position = std::move(arg);
    return Init_RobotStateRt_robot_mode(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_goal_joint_position
{
public:
  explicit Init_RobotStateRt_goal_joint_position(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_goal_tcp_position goal_joint_position(::dsr_msgs2::msg::RobotStateRt::_goal_joint_position_type arg)
  {
    msg_.goal_joint_position = std::move(arg);
    return Init_RobotStateRt_goal_tcp_position(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_external_encoder_count
{
public:
  explicit Init_RobotStateRt_external_encoder_count(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_goal_joint_position external_encoder_count(::dsr_msgs2::msg::RobotStateRt::_external_encoder_count_type arg)
  {
    msg_.external_encoder_count = std::move(arg);
    return Init_RobotStateRt_goal_joint_position(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_external_encoder_strobe_count
{
public:
  explicit Init_RobotStateRt_external_encoder_strobe_count(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_external_encoder_count external_encoder_strobe_count(::dsr_msgs2::msg::RobotStateRt::_external_encoder_strobe_count_type arg)
  {
    msg_.external_encoder_strobe_count = std::move(arg);
    return Init_RobotStateRt_external_encoder_count(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_flange_analog_input
{
public:
  explicit Init_RobotStateRt_flange_analog_input(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_external_encoder_strobe_count flange_analog_input(::dsr_msgs2::msg::RobotStateRt::_flange_analog_input_type arg)
  {
    msg_.flange_analog_input = std::move(arg);
    return Init_RobotStateRt_external_encoder_strobe_count(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_flange_digital_output
{
public:
  explicit Init_RobotStateRt_flange_digital_output(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_flange_analog_input flange_digital_output(::dsr_msgs2::msg::RobotStateRt::_flange_digital_output_type arg)
  {
    msg_.flange_digital_output = std::move(arg);
    return Init_RobotStateRt_flange_analog_input(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_flange_digital_input
{
public:
  explicit Init_RobotStateRt_flange_digital_input(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_flange_digital_output flange_digital_input(::dsr_msgs2::msg::RobotStateRt::_flange_digital_input_type arg)
  {
    msg_.flange_digital_input = std::move(arg);
    return Init_RobotStateRt_flange_digital_output(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_controller_analog_output
{
public:
  explicit Init_RobotStateRt_controller_analog_output(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_flange_digital_input controller_analog_output(::dsr_msgs2::msg::RobotStateRt::_controller_analog_output_type arg)
  {
    msg_.controller_analog_output = std::move(arg);
    return Init_RobotStateRt_flange_digital_input(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_controller_analog_output_type
{
public:
  explicit Init_RobotStateRt_controller_analog_output_type(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_controller_analog_output controller_analog_output_type(::dsr_msgs2::msg::RobotStateRt::_controller_analog_output_type_type arg)
  {
    msg_.controller_analog_output_type = std::move(arg);
    return Init_RobotStateRt_controller_analog_output(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_controller_analog_input
{
public:
  explicit Init_RobotStateRt_controller_analog_input(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_controller_analog_output_type controller_analog_input(::dsr_msgs2::msg::RobotStateRt::_controller_analog_input_type arg)
  {
    msg_.controller_analog_input = std::move(arg);
    return Init_RobotStateRt_controller_analog_output_type(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_controller_analog_input_type
{
public:
  explicit Init_RobotStateRt_controller_analog_input_type(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_controller_analog_input controller_analog_input_type(::dsr_msgs2::msg::RobotStateRt::_controller_analog_input_type_type arg)
  {
    msg_.controller_analog_input_type = std::move(arg);
    return Init_RobotStateRt_controller_analog_input(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_controller_digital_output
{
public:
  explicit Init_RobotStateRt_controller_digital_output(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_controller_analog_input_type controller_digital_output(::dsr_msgs2::msg::RobotStateRt::_controller_digital_output_type arg)
  {
    msg_.controller_digital_output = std::move(arg);
    return Init_RobotStateRt_controller_analog_input_type(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_controller_digital_input
{
public:
  explicit Init_RobotStateRt_controller_digital_input(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_controller_digital_output controller_digital_input(::dsr_msgs2::msg::RobotStateRt::_controller_digital_input_type arg)
  {
    msg_.controller_digital_input = std::move(arg);
    return Init_RobotStateRt_controller_digital_output(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_joint_temperature
{
public:
  explicit Init_RobotStateRt_joint_temperature(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_controller_digital_input joint_temperature(::dsr_msgs2::msg::RobotStateRt::_joint_temperature_type arg)
  {
    msg_.joint_temperature = std::move(arg);
    return Init_RobotStateRt_controller_digital_input(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_operation_speed_rate
{
public:
  explicit Init_RobotStateRt_operation_speed_rate(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_joint_temperature operation_speed_rate(::dsr_msgs2::msg::RobotStateRt::_operation_speed_rate_type arg)
  {
    msg_.operation_speed_rate = std::move(arg);
    return Init_RobotStateRt_joint_temperature(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_singularity
{
public:
  explicit Init_RobotStateRt_singularity(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_operation_speed_rate singularity(::dsr_msgs2::msg::RobotStateRt::_singularity_type arg)
  {
    msg_.singularity = std::move(arg);
    return Init_RobotStateRt_operation_speed_rate(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_solution_space
{
public:
  explicit Init_RobotStateRt_solution_space(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_singularity solution_space(::dsr_msgs2::msg::RobotStateRt::_solution_space_type arg)
  {
    msg_.solution_space = std::move(arg);
    return Init_RobotStateRt_singularity(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_mass_matrix
{
public:
  explicit Init_RobotStateRt_mass_matrix(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_solution_space mass_matrix(::dsr_msgs2::msg::RobotStateRt::_mass_matrix_type arg)
  {
    msg_.mass_matrix = std::move(arg);
    return Init_RobotStateRt_solution_space(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_coriolis_matrix
{
public:
  explicit Init_RobotStateRt_coriolis_matrix(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_mass_matrix coriolis_matrix(::dsr_msgs2::msg::RobotStateRt::_coriolis_matrix_type arg)
  {
    msg_.coriolis_matrix = std::move(arg);
    return Init_RobotStateRt_mass_matrix(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_gravity_torque
{
public:
  explicit Init_RobotStateRt_gravity_torque(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_coriolis_matrix gravity_torque(::dsr_msgs2::msg::RobotStateRt::_gravity_torque_type arg)
  {
    msg_.gravity_torque = std::move(arg);
    return Init_RobotStateRt_coriolis_matrix(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_jacobian_matrix
{
public:
  explicit Init_RobotStateRt_jacobian_matrix(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_gravity_torque jacobian_matrix(::dsr_msgs2::msg::RobotStateRt::_jacobian_matrix_type arg)
  {
    msg_.jacobian_matrix = std::move(arg);
    return Init_RobotStateRt_gravity_torque(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_target_tcp_velocity
{
public:
  explicit Init_RobotStateRt_target_tcp_velocity(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_jacobian_matrix target_tcp_velocity(::dsr_msgs2::msg::RobotStateRt::_target_tcp_velocity_type arg)
  {
    msg_.target_tcp_velocity = std::move(arg);
    return Init_RobotStateRt_jacobian_matrix(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_target_tcp_position
{
public:
  explicit Init_RobotStateRt_target_tcp_position(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_target_tcp_velocity target_tcp_position(::dsr_msgs2::msg::RobotStateRt::_target_tcp_position_type arg)
  {
    msg_.target_tcp_position = std::move(arg);
    return Init_RobotStateRt_target_tcp_velocity(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_target_motor_torque
{
public:
  explicit Init_RobotStateRt_target_motor_torque(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_target_tcp_position target_motor_torque(::dsr_msgs2::msg::RobotStateRt::_target_motor_torque_type arg)
  {
    msg_.target_motor_torque = std::move(arg);
    return Init_RobotStateRt_target_tcp_position(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_target_joint_acceleration
{
public:
  explicit Init_RobotStateRt_target_joint_acceleration(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_target_motor_torque target_joint_acceleration(::dsr_msgs2::msg::RobotStateRt::_target_joint_acceleration_type arg)
  {
    msg_.target_joint_acceleration = std::move(arg);
    return Init_RobotStateRt_target_motor_torque(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_target_joint_velocity
{
public:
  explicit Init_RobotStateRt_target_joint_velocity(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_target_joint_acceleration target_joint_velocity(::dsr_msgs2::msg::RobotStateRt::_target_joint_velocity_type arg)
  {
    msg_.target_joint_velocity = std::move(arg);
    return Init_RobotStateRt_target_joint_acceleration(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_target_joint_position
{
public:
  explicit Init_RobotStateRt_target_joint_position(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_target_joint_velocity target_joint_position(::dsr_msgs2::msg::RobotStateRt::_target_joint_position_type arg)
  {
    msg_.target_joint_position = std::move(arg);
    return Init_RobotStateRt_target_joint_velocity(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_external_tcp_force
{
public:
  explicit Init_RobotStateRt_external_tcp_force(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_target_joint_position external_tcp_force(::dsr_msgs2::msg::RobotStateRt::_external_tcp_force_type arg)
  {
    msg_.external_tcp_force = std::move(arg);
    return Init_RobotStateRt_target_joint_position(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_external_joint_torque
{
public:
  explicit Init_RobotStateRt_external_joint_torque(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_external_tcp_force external_joint_torque(::dsr_msgs2::msg::RobotStateRt::_external_joint_torque_type arg)
  {
    msg_.external_joint_torque = std::move(arg);
    return Init_RobotStateRt_external_tcp_force(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_raw_force_torque
{
public:
  explicit Init_RobotStateRt_raw_force_torque(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_external_joint_torque raw_force_torque(::dsr_msgs2::msg::RobotStateRt::_raw_force_torque_type arg)
  {
    msg_.raw_force_torque = std::move(arg);
    return Init_RobotStateRt_external_joint_torque(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_raw_joint_torque
{
public:
  explicit Init_RobotStateRt_raw_joint_torque(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_raw_force_torque raw_joint_torque(::dsr_msgs2::msg::RobotStateRt::_raw_joint_torque_type arg)
  {
    msg_.raw_joint_torque = std::move(arg);
    return Init_RobotStateRt_raw_force_torque(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_joint_torque
{
public:
  explicit Init_RobotStateRt_actual_joint_torque(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_raw_joint_torque actual_joint_torque(::dsr_msgs2::msg::RobotStateRt::_actual_joint_torque_type arg)
  {
    msg_.actual_joint_torque = std::move(arg);
    return Init_RobotStateRt_raw_joint_torque(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_motor_torque
{
public:
  explicit Init_RobotStateRt_actual_motor_torque(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_actual_joint_torque actual_motor_torque(::dsr_msgs2::msg::RobotStateRt::_actual_motor_torque_type arg)
  {
    msg_.actual_motor_torque = std::move(arg);
    return Init_RobotStateRt_actual_joint_torque(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_flange_velocity
{
public:
  explicit Init_RobotStateRt_actual_flange_velocity(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_actual_motor_torque actual_flange_velocity(::dsr_msgs2::msg::RobotStateRt::_actual_flange_velocity_type arg)
  {
    msg_.actual_flange_velocity = std::move(arg);
    return Init_RobotStateRt_actual_motor_torque(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_flange_position
{
public:
  explicit Init_RobotStateRt_actual_flange_position(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_actual_flange_velocity actual_flange_position(::dsr_msgs2::msg::RobotStateRt::_actual_flange_position_type arg)
  {
    msg_.actual_flange_position = std::move(arg);
    return Init_RobotStateRt_actual_flange_velocity(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_tcp_velocity
{
public:
  explicit Init_RobotStateRt_actual_tcp_velocity(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_actual_flange_position actual_tcp_velocity(::dsr_msgs2::msg::RobotStateRt::_actual_tcp_velocity_type arg)
  {
    msg_.actual_tcp_velocity = std::move(arg);
    return Init_RobotStateRt_actual_flange_position(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_tcp_position
{
public:
  explicit Init_RobotStateRt_actual_tcp_position(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_actual_tcp_velocity actual_tcp_position(::dsr_msgs2::msg::RobotStateRt::_actual_tcp_position_type arg)
  {
    msg_.actual_tcp_position = std::move(arg);
    return Init_RobotStateRt_actual_tcp_velocity(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_joint_velocity_abs
{
public:
  explicit Init_RobotStateRt_actual_joint_velocity_abs(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_actual_tcp_position actual_joint_velocity_abs(::dsr_msgs2::msg::RobotStateRt::_actual_joint_velocity_abs_type arg)
  {
    msg_.actual_joint_velocity_abs = std::move(arg);
    return Init_RobotStateRt_actual_tcp_position(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_joint_velocity
{
public:
  explicit Init_RobotStateRt_actual_joint_velocity(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_actual_joint_velocity_abs actual_joint_velocity(::dsr_msgs2::msg::RobotStateRt::_actual_joint_velocity_type arg)
  {
    msg_.actual_joint_velocity = std::move(arg);
    return Init_RobotStateRt_actual_joint_velocity_abs(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_joint_position_abs
{
public:
  explicit Init_RobotStateRt_actual_joint_position_abs(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_actual_joint_velocity actual_joint_position_abs(::dsr_msgs2::msg::RobotStateRt::_actual_joint_position_abs_type arg)
  {
    msg_.actual_joint_position_abs = std::move(arg);
    return Init_RobotStateRt_actual_joint_velocity(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_actual_joint_position
{
public:
  explicit Init_RobotStateRt_actual_joint_position(::dsr_msgs2::msg::RobotStateRt & msg)
  : msg_(msg)
  {}
  Init_RobotStateRt_actual_joint_position_abs actual_joint_position(::dsr_msgs2::msg::RobotStateRt::_actual_joint_position_type arg)
  {
    msg_.actual_joint_position = std::move(arg);
    return Init_RobotStateRt_actual_joint_position_abs(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

class Init_RobotStateRt_time_stamp
{
public:
  Init_RobotStateRt_time_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotStateRt_actual_joint_position time_stamp(::dsr_msgs2::msg::RobotStateRt::_time_stamp_type arg)
  {
    msg_.time_stamp = std::move(arg);
    return Init_RobotStateRt_actual_joint_position(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStateRt msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::RobotStateRt>()
{
  return dsr_msgs2::msg::builder::Init_RobotStateRt_time_stamp();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__BUILDER_HPP_
