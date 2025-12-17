// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/RobotState.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/robot_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_RobotState_b_actual_encorder_reset_signal
{
public:
  explicit Init_RobotState_b_actual_encorder_reset_signal(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::RobotState b_actual_encorder_reset_signal(::dsr_msgs2::msg::RobotState::_b_actual_encorder_reset_signal_type arg)
  {
    msg_.b_actual_encorder_reset_signal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_i_actual_encorder_raw_data
{
public:
  explicit Init_RobotState_i_actual_encorder_raw_data(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_b_actual_encorder_reset_signal i_actual_encorder_raw_data(::dsr_msgs2::msg::RobotState::_i_actual_encorder_raw_data_type arg)
  {
    msg_.i_actual_encorder_raw_data = std::move(arg);
    return Init_RobotState_b_actual_encorder_reset_signal(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_b_actual_encorder_strove_signal
{
public:
  explicit Init_RobotState_b_actual_encorder_strove_signal(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_i_actual_encorder_raw_data b_actual_encorder_strove_signal(::dsr_msgs2::msg::RobotState::_b_actual_encorder_strove_signal_type arg)
  {
    msg_.b_actual_encorder_strove_signal = std::move(arg);
    return Init_RobotState_i_actual_encorder_raw_data(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_i_target_analog_output_type
{
public:
  explicit Init_RobotState_i_target_analog_output_type(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_b_actual_encorder_strove_signal i_target_analog_output_type(::dsr_msgs2::msg::RobotState::_i_target_analog_output_type_type arg)
  {
    msg_.i_target_analog_output_type = std::move(arg);
    return Init_RobotState_b_actual_encorder_strove_signal(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_target_analog_output
{
public:
  explicit Init_RobotState_f_target_analog_output(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_i_target_analog_output_type f_target_analog_output(::dsr_msgs2::msg::RobotState::_f_target_analog_output_type arg)
  {
    msg_.f_target_analog_output = std::move(arg);
    return Init_RobotState_i_target_analog_output_type(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_i_actual_analog_input_type
{
public:
  explicit Init_RobotState_i_actual_analog_input_type(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_target_analog_output i_actual_analog_input_type(::dsr_msgs2::msg::RobotState::_i_actual_analog_input_type_type arg)
  {
    msg_.i_actual_analog_input_type = std::move(arg);
    return Init_RobotState_f_target_analog_output(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_b_actual_safety_input
{
public:
  explicit Init_RobotState_b_actual_safety_input(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_i_actual_analog_input_type b_actual_safety_input(::dsr_msgs2::msg::RobotState::_b_actual_safety_input_type arg)
  {
    msg_.b_actual_safety_input = std::move(arg);
    return Init_RobotState_i_actual_analog_input_type(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_b_actual_switch_input
{
public:
  explicit Init_RobotState_b_actual_switch_input(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_b_actual_safety_input b_actual_switch_input(::dsr_msgs2::msg::RobotState::_b_actual_switch_input_type arg)
  {
    msg_.b_actual_switch_input = std::move(arg);
    return Init_RobotState_b_actual_safety_input(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_actual_analog_input
{
public:
  explicit Init_RobotState_f_actual_analog_input(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_b_actual_switch_input f_actual_analog_input(::dsr_msgs2::msg::RobotState::_f_actual_analog_input_type arg)
  {
    msg_.f_actual_analog_input = std::move(arg);
    return Init_RobotState_b_actual_switch_input(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_rotation_matrix_user
{
public:
  explicit Init_RobotState_f_rotation_matrix_user(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_actual_analog_input f_rotation_matrix_user(::dsr_msgs2::msg::RobotState::_f_rotation_matrix_user_type arg)
  {
    msg_.f_rotation_matrix_user = std::move(arg);
    return Init_RobotState_f_actual_analog_input(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_target_vel_user
{
public:
  explicit Init_RobotState_f_target_vel_user(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_rotation_matrix_user f_target_vel_user(::dsr_msgs2::msg::RobotState::_f_target_vel_user_type arg)
  {
    msg_.f_target_vel_user = std::move(arg);
    return Init_RobotState_f_rotation_matrix_user(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_target_pos_user
{
public:
  explicit Init_RobotState_f_target_pos_user(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_target_vel_user f_target_pos_user(::dsr_msgs2::msg::RobotState::_f_target_pos_user_type arg)
  {
    msg_.f_target_pos_user = std::move(arg);
    return Init_RobotState_f_target_vel_user(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_user_ext_task_torque
{
public:
  explicit Init_RobotState_f_user_ext_task_torque(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_target_pos_user f_user_ext_task_torque(::dsr_msgs2::msg::RobotState::_f_user_ext_task_torque_type arg)
  {
    msg_.f_user_ext_task_torque = std::move(arg);
    return Init_RobotState_f_target_pos_user(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_current_vel_user
{
public:
  explicit Init_RobotState_f_current_vel_user(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_user_ext_task_torque f_current_vel_user(::dsr_msgs2::msg::RobotState::_f_current_vel_user_type arg)
  {
    msg_.f_current_vel_user = std::move(arg);
    return Init_RobotState_f_user_ext_task_torque(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_current_pos_user
{
public:
  explicit Init_RobotState_f_current_pos_user(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_current_vel_user f_current_pos_user(::dsr_msgs2::msg::RobotState::_f_current_pos_user_type arg)
  {
    msg_.f_current_pos_user = std::move(arg);
    return Init_RobotState_f_current_vel_user(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_i_coord_ref
{
public:
  explicit Init_RobotState_i_coord_ref(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_current_pos_user i_coord_ref(::dsr_msgs2::msg::RobotState::_i_coord_ref_type arg)
  {
    msg_.i_coord_ref = std::move(arg);
    return Init_RobotState_f_current_pos_user(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_i_actual_user_coord_num
{
public:
  explicit Init_RobotState_i_actual_user_coord_num(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_i_coord_ref i_actual_user_coord_num(::dsr_msgs2::msg::RobotState::_i_actual_user_coord_num_type arg)
  {
    msg_.i_actual_user_coord_num = std::move(arg);
    return Init_RobotState_i_coord_ref(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_rotation_matrix_world
{
public:
  explicit Init_RobotState_f_rotation_matrix_world(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_i_actual_user_coord_num f_rotation_matrix_world(::dsr_msgs2::msg::RobotState::_f_rotation_matrix_world_type arg)
  {
    msg_.f_rotation_matrix_world = std::move(arg);
    return Init_RobotState_i_actual_user_coord_num(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_target_vel_world
{
public:
  explicit Init_RobotState_f_target_vel_world(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_rotation_matrix_world f_target_vel_world(::dsr_msgs2::msg::RobotState::_f_target_vel_world_type arg)
  {
    msg_.f_target_vel_world = std::move(arg);
    return Init_RobotState_f_rotation_matrix_world(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_target_pos_world
{
public:
  explicit Init_RobotState_f_target_pos_world(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_target_vel_world f_target_pos_world(::dsr_msgs2::msg::RobotState::_f_target_pos_world_type arg)
  {
    msg_.f_target_pos_world = std::move(arg);
    return Init_RobotState_f_target_vel_world(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_world_ext_target_torque
{
public:
  explicit Init_RobotState_f_world_ext_target_torque(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_target_pos_world f_world_ext_target_torque(::dsr_msgs2::msg::RobotState::_f_world_ext_target_torque_type arg)
  {
    msg_.f_world_ext_target_torque = std::move(arg);
    return Init_RobotState_f_target_pos_world(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_current_vel_world
{
public:
  explicit Init_RobotState_f_current_vel_world(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_world_ext_target_torque f_current_vel_world(::dsr_msgs2::msg::RobotState::_f_current_vel_world_type arg)
  {
    msg_.f_current_vel_world = std::move(arg);
    return Init_RobotState_f_world_ext_target_torque(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_current_pos_world
{
public:
  explicit Init_RobotState_f_current_pos_world(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_current_vel_world f_current_pos_world(::dsr_msgs2::msg::RobotState::_f_current_pos_world_type arg)
  {
    msg_.f_current_pos_world = std::move(arg);
    return Init_RobotState_f_current_vel_world(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_f_actual_w2b
{
public:
  explicit Init_RobotState_f_actual_w2b(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_current_pos_world f_actual_w2b(::dsr_msgs2::msg::RobotState::_f_actual_w2b_type arg)
  {
    msg_.f_actual_w2b = std::move(arg);
    return Init_RobotState_f_current_pos_world(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_disconnected
{
public:
  explicit Init_RobotState_disconnected(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_f_actual_w2b disconnected(::dsr_msgs2::msg::RobotState::_disconnected_type arg)
  {
    msg_.disconnected = std::move(arg);
    return Init_RobotState_f_actual_w2b(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_drl_stopped
{
public:
  explicit Init_RobotState_drl_stopped(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_disconnected drl_stopped(::dsr_msgs2::msg::RobotState::_drl_stopped_type arg)
  {
    msg_.drl_stopped = std::move(arg);
    return Init_RobotState_disconnected(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_mastering_need
{
public:
  explicit Init_RobotState_mastering_need(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_drl_stopped mastering_need(::dsr_msgs2::msg::RobotState::_mastering_need_type arg)
  {
    msg_.mastering_need = std::move(arg);
    return Init_RobotState_drl_stopped(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_tp_initialized
{
public:
  explicit Init_RobotState_tp_initialized(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_mastering_need tp_initialized(::dsr_msgs2::msg::RobotState::_tp_initialized_type arg)
  {
    msg_.tp_initialized = std::move(arg);
    return Init_RobotState_mastering_need(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_homming_completed
{
public:
  explicit Init_RobotState_homming_completed(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_tp_initialized homming_completed(::dsr_msgs2::msg::RobotState::_homming_completed_type arg)
  {
    msg_.homming_completed = std::move(arg);
    return Init_RobotState_tp_initialized(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_access_control
{
public:
  explicit Init_RobotState_access_control(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_homming_completed access_control(::dsr_msgs2::msg::RobotState::_access_control_type arg)
  {
    msg_.access_control = std::move(arg);
    return Init_RobotState_homming_completed(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_modbus_state
{
public:
  explicit Init_RobotState_modbus_state(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_access_control modbus_state(::dsr_msgs2::msg::RobotState::_modbus_state_type arg)
  {
    msg_.modbus_state = std::move(arg);
    return Init_RobotState_access_control(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_flange_digital_output
{
public:
  explicit Init_RobotState_flange_digital_output(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_modbus_state flange_digital_output(::dsr_msgs2::msg::RobotState::_flange_digital_output_type arg)
  {
    msg_.flange_digital_output = std::move(arg);
    return Init_RobotState_modbus_state(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_flange_digital_input
{
public:
  explicit Init_RobotState_flange_digital_input(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_flange_digital_output flange_digital_input(::dsr_msgs2::msg::RobotState::_flange_digital_input_type arg)
  {
    msg_.flange_digital_input = std::move(arg);
    return Init_RobotState_flange_digital_output(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_ctrlbox_digital_output
{
public:
  explicit Init_RobotState_ctrlbox_digital_output(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_flange_digital_input ctrlbox_digital_output(::dsr_msgs2::msg::RobotState::_ctrlbox_digital_output_type arg)
  {
    msg_.ctrlbox_digital_output = std::move(arg);
    return Init_RobotState_flange_digital_input(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_ctrlbox_digital_input
{
public:
  explicit Init_RobotState_ctrlbox_digital_input(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_ctrlbox_digital_output ctrlbox_digital_input(::dsr_msgs2::msg::RobotState::_ctrlbox_digital_input_type arg)
  {
    msg_.ctrlbox_digital_input = std::move(arg);
    return Init_RobotState_ctrlbox_digital_output(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_rotation_matrix
{
public:
  explicit Init_RobotState_rotation_matrix(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_ctrlbox_digital_input rotation_matrix(::dsr_msgs2::msg::RobotState::_rotation_matrix_type arg)
  {
    msg_.rotation_matrix = std::move(arg);
    return Init_RobotState_ctrlbox_digital_input(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_actual_bt
{
public:
  explicit Init_RobotState_actual_bt(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_rotation_matrix actual_bt(::dsr_msgs2::msg::RobotState::_actual_bt_type arg)
  {
    msg_.actual_bt = std::move(arg);
    return Init_RobotState_rotation_matrix(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_sync_time
{
public:
  explicit Init_RobotState_sync_time(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_actual_bt sync_time(::dsr_msgs2::msg::RobotState::_sync_time_type arg)
  {
    msg_.sync_time = std::move(arg);
    return Init_RobotState_actual_bt(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_solution_space
{
public:
  explicit Init_RobotState_solution_space(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_sync_time solution_space(::dsr_msgs2::msg::RobotState::_solution_space_type arg)
  {
    msg_.solution_space = std::move(arg);
    return Init_RobotState_sync_time(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_actual_mt
{
public:
  explicit Init_RobotState_actual_mt(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_solution_space actual_mt(::dsr_msgs2::msg::RobotState::_actual_mt_type arg)
  {
    msg_.actual_mt = std::move(arg);
    return Init_RobotState_solution_space(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_actual_mc
{
public:
  explicit Init_RobotState_actual_mc(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_actual_mt actual_mc(::dsr_msgs2::msg::RobotState::_actual_mc_type arg)
  {
    msg_.actual_mc = std::move(arg);
    return Init_RobotState_actual_mt(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_actual_bk
{
public:
  explicit Init_RobotState_actual_bk(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_actual_mc actual_bk(::dsr_msgs2::msg::RobotState::_actual_bk_type arg)
  {
    msg_.actual_bk = std::move(arg);
    return Init_RobotState_actual_mc(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_actual_ett
{
public:
  explicit Init_RobotState_actual_ett(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_actual_bk actual_ett(::dsr_msgs2::msg::RobotState::_actual_ett_type arg)
  {
    msg_.actual_ett = std::move(arg);
    return Init_RobotState_actual_bk(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_actual_ejt
{
public:
  explicit Init_RobotState_actual_ejt(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_actual_ett actual_ejt(::dsr_msgs2::msg::RobotState::_actual_ejt_type arg)
  {
    msg_.actual_ejt = std::move(arg);
    return Init_RobotState_actual_ett(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_actual_jts
{
public:
  explicit Init_RobotState_actual_jts(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_actual_ejt actual_jts(::dsr_msgs2::msg::RobotState::_actual_jts_type arg)
  {
    msg_.actual_jts = std::move(arg);
    return Init_RobotState_actual_ejt(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_dynamic_tor
{
public:
  explicit Init_RobotState_dynamic_tor(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_actual_jts dynamic_tor(::dsr_msgs2::msg::RobotState::_dynamic_tor_type arg)
  {
    msg_.dynamic_tor = std::move(arg);
    return Init_RobotState_actual_jts(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_target_posx
{
public:
  explicit Init_RobotState_target_posx(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_dynamic_tor target_posx(::dsr_msgs2::msg::RobotState::_target_posx_type arg)
  {
    msg_.target_posx = std::move(arg);
    return Init_RobotState_dynamic_tor(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_target_velx
{
public:
  explicit Init_RobotState_target_velx(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_target_posx target_velx(::dsr_msgs2::msg::RobotState::_target_velx_type arg)
  {
    msg_.target_velx = std::move(arg);
    return Init_RobotState_target_posx(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_task_err
{
public:
  explicit Init_RobotState_task_err(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_target_velx task_err(::dsr_msgs2::msg::RobotState::_task_err_type arg)
  {
    msg_.task_err = std::move(arg);
    return Init_RobotState_target_velx(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_current_velx
{
public:
  explicit Init_RobotState_current_velx(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_task_err current_velx(::dsr_msgs2::msg::RobotState::_current_velx_type arg)
  {
    msg_.current_velx = std::move(arg);
    return Init_RobotState_task_err(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_current_tool_posx
{
public:
  explicit Init_RobotState_current_tool_posx(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_current_velx current_tool_posx(::dsr_msgs2::msg::RobotState::_current_tool_posx_type arg)
  {
    msg_.current_tool_posx = std::move(arg);
    return Init_RobotState_current_velx(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_current_posx
{
public:
  explicit Init_RobotState_current_posx(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_current_tool_posx current_posx(::dsr_msgs2::msg::RobotState::_current_posx_type arg)
  {
    msg_.current_posx = std::move(arg);
    return Init_RobotState_current_tool_posx(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_target_velj
{
public:
  explicit Init_RobotState_target_velj(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_current_posx target_velj(::dsr_msgs2::msg::RobotState::_target_velj_type arg)
  {
    msg_.target_velj = std::move(arg);
    return Init_RobotState_current_posx(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_target_posj
{
public:
  explicit Init_RobotState_target_posj(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_target_velj target_posj(::dsr_msgs2::msg::RobotState::_target_posj_type arg)
  {
    msg_.target_posj = std::move(arg);
    return Init_RobotState_target_velj(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_joint_err
{
public:
  explicit Init_RobotState_joint_err(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_target_posj joint_err(::dsr_msgs2::msg::RobotState::_joint_err_type arg)
  {
    msg_.joint_err = std::move(arg);
    return Init_RobotState_target_posj(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_joint_abs
{
public:
  explicit Init_RobotState_joint_abs(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_joint_err joint_abs(::dsr_msgs2::msg::RobotState::_joint_abs_type arg)
  {
    msg_.joint_abs = std::move(arg);
    return Init_RobotState_joint_err(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_current_velj
{
public:
  explicit Init_RobotState_current_velj(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_joint_abs current_velj(::dsr_msgs2::msg::RobotState::_current_velj_type arg)
  {
    msg_.current_velj = std::move(arg);
    return Init_RobotState_joint_abs(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_current_posj
{
public:
  explicit Init_RobotState_current_posj(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_current_velj current_posj(::dsr_msgs2::msg::RobotState::_current_posj_type arg)
  {
    msg_.current_posj = std::move(arg);
    return Init_RobotState_current_velj(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_actual_space
{
public:
  explicit Init_RobotState_actual_space(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_current_posj actual_space(::dsr_msgs2::msg::RobotState::_actual_space_type arg)
  {
    msg_.actual_space = std::move(arg);
    return Init_RobotState_current_posj(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_actual_mode
{
public:
  explicit Init_RobotState_actual_mode(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_actual_space actual_mode(::dsr_msgs2::msg::RobotState::_actual_mode_type arg)
  {
    msg_.actual_mode = std::move(arg);
    return Init_RobotState_actual_space(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_robot_state_str
{
public:
  explicit Init_RobotState_robot_state_str(::dsr_msgs2::msg::RobotState & msg)
  : msg_(msg)
  {}
  Init_RobotState_actual_mode robot_state_str(::dsr_msgs2::msg::RobotState::_robot_state_str_type arg)
  {
    msg_.robot_state_str = std::move(arg);
    return Init_RobotState_actual_mode(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

class Init_RobotState_robot_state
{
public:
  Init_RobotState_robot_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RobotState_robot_state_str robot_state(::dsr_msgs2::msg::RobotState::_robot_state_type arg)
  {
    msg_.robot_state = std::move(arg);
    return Init_RobotState_robot_state_str(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::RobotState>()
{
  return dsr_msgs2::msg::builder::Init_RobotState_robot_state();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STATE__BUILDER_HPP_
