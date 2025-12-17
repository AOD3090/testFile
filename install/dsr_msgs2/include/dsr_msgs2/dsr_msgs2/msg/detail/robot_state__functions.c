// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dsr_msgs2:msg/RobotState.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/robot_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `robot_state_str`
#include "rosidl_runtime_c/string_functions.h"
// Member `rotation_matrix`
// Member `f_current_pos_world`
// Member `f_rotation_matrix_world`
// Member `f_current_pos_user`
// Member `f_rotation_matrix_user`
#include "std_msgs/msg/detail/float64_multi_array__functions.h"
// Member `modbus_state`
#include "dsr_msgs2/msg/detail/modbus_state__functions.h"

bool
dsr_msgs2__msg__RobotState__init(dsr_msgs2__msg__RobotState * msg)
{
  if (!msg) {
    return false;
  }
  // robot_state
  // robot_state_str
  if (!rosidl_runtime_c__String__init(&msg->robot_state_str)) {
    dsr_msgs2__msg__RobotState__fini(msg);
    return false;
  }
  // actual_mode
  // actual_space
  // current_posj
  // current_velj
  // joint_abs
  // joint_err
  // target_posj
  // target_velj
  // current_posx
  // current_tool_posx
  // current_velx
  // task_err
  // target_velx
  // target_posx
  // dynamic_tor
  // actual_jts
  // actual_ejt
  // actual_ett
  // actual_bk
  // actual_mc
  // actual_mt
  // solution_space
  // sync_time
  // actual_bt
  // rotation_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__init(&msg->rotation_matrix, 0)) {
    dsr_msgs2__msg__RobotState__fini(msg);
    return false;
  }
  // ctrlbox_digital_input
  // ctrlbox_digital_output
  // flange_digital_input
  // flange_digital_output
  // modbus_state
  if (!dsr_msgs2__msg__ModbusState__Sequence__init(&msg->modbus_state, 0)) {
    dsr_msgs2__msg__RobotState__fini(msg);
    return false;
  }
  // access_control
  // homming_completed
  // tp_initialized
  // mastering_need
  // drl_stopped
  // disconnected
  // f_actual_w2b
  // f_current_pos_world
  if (!std_msgs__msg__Float64MultiArray__Sequence__init(&msg->f_current_pos_world, 0)) {
    dsr_msgs2__msg__RobotState__fini(msg);
    return false;
  }
  // f_current_vel_world
  // f_world_ext_target_torque
  // f_target_pos_world
  // f_target_vel_world
  // f_rotation_matrix_world
  if (!std_msgs__msg__Float64MultiArray__Sequence__init(&msg->f_rotation_matrix_world, 0)) {
    dsr_msgs2__msg__RobotState__fini(msg);
    return false;
  }
  // i_actual_user_coord_num
  // i_coord_ref
  // f_current_pos_user
  if (!std_msgs__msg__Float64MultiArray__Sequence__init(&msg->f_current_pos_user, 0)) {
    dsr_msgs2__msg__RobotState__fini(msg);
    return false;
  }
  // f_current_vel_user
  // f_user_ext_task_torque
  // f_target_pos_user
  // f_target_vel_user
  // f_rotation_matrix_user
  if (!std_msgs__msg__Float64MultiArray__Sequence__init(&msg->f_rotation_matrix_user, 0)) {
    dsr_msgs2__msg__RobotState__fini(msg);
    return false;
  }
  // f_actual_analog_input
  // b_actual_switch_input
  // b_actual_safety_input
  // i_actual_analog_input_type
  // f_target_analog_output
  // i_target_analog_output_type
  // b_actual_encorder_strove_signal
  // i_actual_encorder_raw_data
  // b_actual_encorder_reset_signal
  return true;
}

void
dsr_msgs2__msg__RobotState__fini(dsr_msgs2__msg__RobotState * msg)
{
  if (!msg) {
    return;
  }
  // robot_state
  // robot_state_str
  rosidl_runtime_c__String__fini(&msg->robot_state_str);
  // actual_mode
  // actual_space
  // current_posj
  // current_velj
  // joint_abs
  // joint_err
  // target_posj
  // target_velj
  // current_posx
  // current_tool_posx
  // current_velx
  // task_err
  // target_velx
  // target_posx
  // dynamic_tor
  // actual_jts
  // actual_ejt
  // actual_ett
  // actual_bk
  // actual_mc
  // actual_mt
  // solution_space
  // sync_time
  // actual_bt
  // rotation_matrix
  std_msgs__msg__Float64MultiArray__Sequence__fini(&msg->rotation_matrix);
  // ctrlbox_digital_input
  // ctrlbox_digital_output
  // flange_digital_input
  // flange_digital_output
  // modbus_state
  dsr_msgs2__msg__ModbusState__Sequence__fini(&msg->modbus_state);
  // access_control
  // homming_completed
  // tp_initialized
  // mastering_need
  // drl_stopped
  // disconnected
  // f_actual_w2b
  // f_current_pos_world
  std_msgs__msg__Float64MultiArray__Sequence__fini(&msg->f_current_pos_world);
  // f_current_vel_world
  // f_world_ext_target_torque
  // f_target_pos_world
  // f_target_vel_world
  // f_rotation_matrix_world
  std_msgs__msg__Float64MultiArray__Sequence__fini(&msg->f_rotation_matrix_world);
  // i_actual_user_coord_num
  // i_coord_ref
  // f_current_pos_user
  std_msgs__msg__Float64MultiArray__Sequence__fini(&msg->f_current_pos_user);
  // f_current_vel_user
  // f_user_ext_task_torque
  // f_target_pos_user
  // f_target_vel_user
  // f_rotation_matrix_user
  std_msgs__msg__Float64MultiArray__Sequence__fini(&msg->f_rotation_matrix_user);
  // f_actual_analog_input
  // b_actual_switch_input
  // b_actual_safety_input
  // i_actual_analog_input_type
  // f_target_analog_output
  // i_target_analog_output_type
  // b_actual_encorder_strove_signal
  // i_actual_encorder_raw_data
  // b_actual_encorder_reset_signal
}

bool
dsr_msgs2__msg__RobotState__are_equal(const dsr_msgs2__msg__RobotState * lhs, const dsr_msgs2__msg__RobotState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // robot_state
  if (lhs->robot_state != rhs->robot_state) {
    return false;
  }
  // robot_state_str
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->robot_state_str), &(rhs->robot_state_str)))
  {
    return false;
  }
  // actual_mode
  if (lhs->actual_mode != rhs->actual_mode) {
    return false;
  }
  // actual_space
  if (lhs->actual_space != rhs->actual_space) {
    return false;
  }
  // current_posj
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->current_posj[i] != rhs->current_posj[i]) {
      return false;
    }
  }
  // current_velj
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->current_velj[i] != rhs->current_velj[i]) {
      return false;
    }
  }
  // joint_abs
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->joint_abs[i] != rhs->joint_abs[i]) {
      return false;
    }
  }
  // joint_err
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->joint_err[i] != rhs->joint_err[i]) {
      return false;
    }
  }
  // target_posj
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_posj[i] != rhs->target_posj[i]) {
      return false;
    }
  }
  // target_velj
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_velj[i] != rhs->target_velj[i]) {
      return false;
    }
  }
  // current_posx
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->current_posx[i] != rhs->current_posx[i]) {
      return false;
    }
  }
  // current_tool_posx
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->current_tool_posx[i] != rhs->current_tool_posx[i]) {
      return false;
    }
  }
  // current_velx
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->current_velx[i] != rhs->current_velx[i]) {
      return false;
    }
  }
  // task_err
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->task_err[i] != rhs->task_err[i]) {
      return false;
    }
  }
  // target_velx
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_velx[i] != rhs->target_velx[i]) {
      return false;
    }
  }
  // target_posx
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_posx[i] != rhs->target_posx[i]) {
      return false;
    }
  }
  // dynamic_tor
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->dynamic_tor[i] != rhs->dynamic_tor[i]) {
      return false;
    }
  }
  // actual_jts
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_jts[i] != rhs->actual_jts[i]) {
      return false;
    }
  }
  // actual_ejt
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_ejt[i] != rhs->actual_ejt[i]) {
      return false;
    }
  }
  // actual_ett
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_ett[i] != rhs->actual_ett[i]) {
      return false;
    }
  }
  // actual_bk
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_bk[i] != rhs->actual_bk[i]) {
      return false;
    }
  }
  // actual_mc
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_mc[i] != rhs->actual_mc[i]) {
      return false;
    }
  }
  // actual_mt
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_mt[i] != rhs->actual_mt[i]) {
      return false;
    }
  }
  // solution_space
  if (lhs->solution_space != rhs->solution_space) {
    return false;
  }
  // sync_time
  if (lhs->sync_time != rhs->sync_time) {
    return false;
  }
  // actual_bt
  for (size_t i = 0; i < 5; ++i) {
    if (lhs->actual_bt[i] != rhs->actual_bt[i]) {
      return false;
    }
  }
  // rotation_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__are_equal(
      &(lhs->rotation_matrix), &(rhs->rotation_matrix)))
  {
    return false;
  }
  // ctrlbox_digital_input
  for (size_t i = 0; i < 16; ++i) {
    if (lhs->ctrlbox_digital_input[i] != rhs->ctrlbox_digital_input[i]) {
      return false;
    }
  }
  // ctrlbox_digital_output
  for (size_t i = 0; i < 16; ++i) {
    if (lhs->ctrlbox_digital_output[i] != rhs->ctrlbox_digital_output[i]) {
      return false;
    }
  }
  // flange_digital_input
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->flange_digital_input[i] != rhs->flange_digital_input[i]) {
      return false;
    }
  }
  // flange_digital_output
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->flange_digital_output[i] != rhs->flange_digital_output[i]) {
      return false;
    }
  }
  // modbus_state
  if (!dsr_msgs2__msg__ModbusState__Sequence__are_equal(
      &(lhs->modbus_state), &(rhs->modbus_state)))
  {
    return false;
  }
  // access_control
  if (lhs->access_control != rhs->access_control) {
    return false;
  }
  // homming_completed
  if (lhs->homming_completed != rhs->homming_completed) {
    return false;
  }
  // tp_initialized
  if (lhs->tp_initialized != rhs->tp_initialized) {
    return false;
  }
  // mastering_need
  if (lhs->mastering_need != rhs->mastering_need) {
    return false;
  }
  // drl_stopped
  if (lhs->drl_stopped != rhs->drl_stopped) {
    return false;
  }
  // disconnected
  if (lhs->disconnected != rhs->disconnected) {
    return false;
  }
  // f_actual_w2b
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_actual_w2b[i] != rhs->f_actual_w2b[i]) {
      return false;
    }
  }
  // f_current_pos_world
  if (!std_msgs__msg__Float64MultiArray__Sequence__are_equal(
      &(lhs->f_current_pos_world), &(rhs->f_current_pos_world)))
  {
    return false;
  }
  // f_current_vel_world
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_current_vel_world[i] != rhs->f_current_vel_world[i]) {
      return false;
    }
  }
  // f_world_ext_target_torque
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_world_ext_target_torque[i] != rhs->f_world_ext_target_torque[i]) {
      return false;
    }
  }
  // f_target_pos_world
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_target_pos_world[i] != rhs->f_target_pos_world[i]) {
      return false;
    }
  }
  // f_target_vel_world
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_target_vel_world[i] != rhs->f_target_vel_world[i]) {
      return false;
    }
  }
  // f_rotation_matrix_world
  if (!std_msgs__msg__Float64MultiArray__Sequence__are_equal(
      &(lhs->f_rotation_matrix_world), &(rhs->f_rotation_matrix_world)))
  {
    return false;
  }
  // i_actual_user_coord_num
  if (lhs->i_actual_user_coord_num != rhs->i_actual_user_coord_num) {
    return false;
  }
  // i_coord_ref
  if (lhs->i_coord_ref != rhs->i_coord_ref) {
    return false;
  }
  // f_current_pos_user
  if (!std_msgs__msg__Float64MultiArray__Sequence__are_equal(
      &(lhs->f_current_pos_user), &(rhs->f_current_pos_user)))
  {
    return false;
  }
  // f_current_vel_user
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_current_vel_user[i] != rhs->f_current_vel_user[i]) {
      return false;
    }
  }
  // f_user_ext_task_torque
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_user_ext_task_torque[i] != rhs->f_user_ext_task_torque[i]) {
      return false;
    }
  }
  // f_target_pos_user
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_target_pos_user[i] != rhs->f_target_pos_user[i]) {
      return false;
    }
  }
  // f_target_vel_user
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_target_vel_user[i] != rhs->f_target_vel_user[i]) {
      return false;
    }
  }
  // f_rotation_matrix_user
  if (!std_msgs__msg__Float64MultiArray__Sequence__are_equal(
      &(lhs->f_rotation_matrix_user), &(rhs->f_rotation_matrix_user)))
  {
    return false;
  }
  // f_actual_analog_input
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->f_actual_analog_input[i] != rhs->f_actual_analog_input[i]) {
      return false;
    }
  }
  // b_actual_switch_input
  for (size_t i = 0; i < 3; ++i) {
    if (lhs->b_actual_switch_input[i] != rhs->b_actual_switch_input[i]) {
      return false;
    }
  }
  // b_actual_safety_input
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->b_actual_safety_input[i] != rhs->b_actual_safety_input[i]) {
      return false;
    }
  }
  // i_actual_analog_input_type
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->i_actual_analog_input_type[i] != rhs->i_actual_analog_input_type[i]) {
      return false;
    }
  }
  // f_target_analog_output
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->f_target_analog_output[i] != rhs->f_target_analog_output[i]) {
      return false;
    }
  }
  // i_target_analog_output_type
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->i_target_analog_output_type[i] != rhs->i_target_analog_output_type[i]) {
      return false;
    }
  }
  // b_actual_encorder_strove_signal
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->b_actual_encorder_strove_signal[i] != rhs->b_actual_encorder_strove_signal[i]) {
      return false;
    }
  }
  // i_actual_encorder_raw_data
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->i_actual_encorder_raw_data[i] != rhs->i_actual_encorder_raw_data[i]) {
      return false;
    }
  }
  // b_actual_encorder_reset_signal
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->b_actual_encorder_reset_signal[i] != rhs->b_actual_encorder_reset_signal[i]) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__msg__RobotState__copy(
  const dsr_msgs2__msg__RobotState * input,
  dsr_msgs2__msg__RobotState * output)
{
  if (!input || !output) {
    return false;
  }
  // robot_state
  output->robot_state = input->robot_state;
  // robot_state_str
  if (!rosidl_runtime_c__String__copy(
      &(input->robot_state_str), &(output->robot_state_str)))
  {
    return false;
  }
  // actual_mode
  output->actual_mode = input->actual_mode;
  // actual_space
  output->actual_space = input->actual_space;
  // current_posj
  for (size_t i = 0; i < 6; ++i) {
    output->current_posj[i] = input->current_posj[i];
  }
  // current_velj
  for (size_t i = 0; i < 6; ++i) {
    output->current_velj[i] = input->current_velj[i];
  }
  // joint_abs
  for (size_t i = 0; i < 6; ++i) {
    output->joint_abs[i] = input->joint_abs[i];
  }
  // joint_err
  for (size_t i = 0; i < 6; ++i) {
    output->joint_err[i] = input->joint_err[i];
  }
  // target_posj
  for (size_t i = 0; i < 6; ++i) {
    output->target_posj[i] = input->target_posj[i];
  }
  // target_velj
  for (size_t i = 0; i < 6; ++i) {
    output->target_velj[i] = input->target_velj[i];
  }
  // current_posx
  for (size_t i = 0; i < 6; ++i) {
    output->current_posx[i] = input->current_posx[i];
  }
  // current_tool_posx
  for (size_t i = 0; i < 6; ++i) {
    output->current_tool_posx[i] = input->current_tool_posx[i];
  }
  // current_velx
  for (size_t i = 0; i < 6; ++i) {
    output->current_velx[i] = input->current_velx[i];
  }
  // task_err
  for (size_t i = 0; i < 6; ++i) {
    output->task_err[i] = input->task_err[i];
  }
  // target_velx
  for (size_t i = 0; i < 6; ++i) {
    output->target_velx[i] = input->target_velx[i];
  }
  // target_posx
  for (size_t i = 0; i < 6; ++i) {
    output->target_posx[i] = input->target_posx[i];
  }
  // dynamic_tor
  for (size_t i = 0; i < 6; ++i) {
    output->dynamic_tor[i] = input->dynamic_tor[i];
  }
  // actual_jts
  for (size_t i = 0; i < 6; ++i) {
    output->actual_jts[i] = input->actual_jts[i];
  }
  // actual_ejt
  for (size_t i = 0; i < 6; ++i) {
    output->actual_ejt[i] = input->actual_ejt[i];
  }
  // actual_ett
  for (size_t i = 0; i < 6; ++i) {
    output->actual_ett[i] = input->actual_ett[i];
  }
  // actual_bk
  for (size_t i = 0; i < 6; ++i) {
    output->actual_bk[i] = input->actual_bk[i];
  }
  // actual_mc
  for (size_t i = 0; i < 6; ++i) {
    output->actual_mc[i] = input->actual_mc[i];
  }
  // actual_mt
  for (size_t i = 0; i < 6; ++i) {
    output->actual_mt[i] = input->actual_mt[i];
  }
  // solution_space
  output->solution_space = input->solution_space;
  // sync_time
  output->sync_time = input->sync_time;
  // actual_bt
  for (size_t i = 0; i < 5; ++i) {
    output->actual_bt[i] = input->actual_bt[i];
  }
  // rotation_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__copy(
      &(input->rotation_matrix), &(output->rotation_matrix)))
  {
    return false;
  }
  // ctrlbox_digital_input
  for (size_t i = 0; i < 16; ++i) {
    output->ctrlbox_digital_input[i] = input->ctrlbox_digital_input[i];
  }
  // ctrlbox_digital_output
  for (size_t i = 0; i < 16; ++i) {
    output->ctrlbox_digital_output[i] = input->ctrlbox_digital_output[i];
  }
  // flange_digital_input
  for (size_t i = 0; i < 6; ++i) {
    output->flange_digital_input[i] = input->flange_digital_input[i];
  }
  // flange_digital_output
  for (size_t i = 0; i < 6; ++i) {
    output->flange_digital_output[i] = input->flange_digital_output[i];
  }
  // modbus_state
  if (!dsr_msgs2__msg__ModbusState__Sequence__copy(
      &(input->modbus_state), &(output->modbus_state)))
  {
    return false;
  }
  // access_control
  output->access_control = input->access_control;
  // homming_completed
  output->homming_completed = input->homming_completed;
  // tp_initialized
  output->tp_initialized = input->tp_initialized;
  // mastering_need
  output->mastering_need = input->mastering_need;
  // drl_stopped
  output->drl_stopped = input->drl_stopped;
  // disconnected
  output->disconnected = input->disconnected;
  // f_actual_w2b
  for (size_t i = 0; i < 6; ++i) {
    output->f_actual_w2b[i] = input->f_actual_w2b[i];
  }
  // f_current_pos_world
  if (!std_msgs__msg__Float64MultiArray__Sequence__copy(
      &(input->f_current_pos_world), &(output->f_current_pos_world)))
  {
    return false;
  }
  // f_current_vel_world
  for (size_t i = 0; i < 6; ++i) {
    output->f_current_vel_world[i] = input->f_current_vel_world[i];
  }
  // f_world_ext_target_torque
  for (size_t i = 0; i < 6; ++i) {
    output->f_world_ext_target_torque[i] = input->f_world_ext_target_torque[i];
  }
  // f_target_pos_world
  for (size_t i = 0; i < 6; ++i) {
    output->f_target_pos_world[i] = input->f_target_pos_world[i];
  }
  // f_target_vel_world
  for (size_t i = 0; i < 6; ++i) {
    output->f_target_vel_world[i] = input->f_target_vel_world[i];
  }
  // f_rotation_matrix_world
  if (!std_msgs__msg__Float64MultiArray__Sequence__copy(
      &(input->f_rotation_matrix_world), &(output->f_rotation_matrix_world)))
  {
    return false;
  }
  // i_actual_user_coord_num
  output->i_actual_user_coord_num = input->i_actual_user_coord_num;
  // i_coord_ref
  output->i_coord_ref = input->i_coord_ref;
  // f_current_pos_user
  if (!std_msgs__msg__Float64MultiArray__Sequence__copy(
      &(input->f_current_pos_user), &(output->f_current_pos_user)))
  {
    return false;
  }
  // f_current_vel_user
  for (size_t i = 0; i < 6; ++i) {
    output->f_current_vel_user[i] = input->f_current_vel_user[i];
  }
  // f_user_ext_task_torque
  for (size_t i = 0; i < 6; ++i) {
    output->f_user_ext_task_torque[i] = input->f_user_ext_task_torque[i];
  }
  // f_target_pos_user
  for (size_t i = 0; i < 6; ++i) {
    output->f_target_pos_user[i] = input->f_target_pos_user[i];
  }
  // f_target_vel_user
  for (size_t i = 0; i < 6; ++i) {
    output->f_target_vel_user[i] = input->f_target_vel_user[i];
  }
  // f_rotation_matrix_user
  if (!std_msgs__msg__Float64MultiArray__Sequence__copy(
      &(input->f_rotation_matrix_user), &(output->f_rotation_matrix_user)))
  {
    return false;
  }
  // f_actual_analog_input
  for (size_t i = 0; i < 6; ++i) {
    output->f_actual_analog_input[i] = input->f_actual_analog_input[i];
  }
  // b_actual_switch_input
  for (size_t i = 0; i < 3; ++i) {
    output->b_actual_switch_input[i] = input->b_actual_switch_input[i];
  }
  // b_actual_safety_input
  for (size_t i = 0; i < 2; ++i) {
    output->b_actual_safety_input[i] = input->b_actual_safety_input[i];
  }
  // i_actual_analog_input_type
  for (size_t i = 0; i < 2; ++i) {
    output->i_actual_analog_input_type[i] = input->i_actual_analog_input_type[i];
  }
  // f_target_analog_output
  for (size_t i = 0; i < 2; ++i) {
    output->f_target_analog_output[i] = input->f_target_analog_output[i];
  }
  // i_target_analog_output_type
  for (size_t i = 0; i < 2; ++i) {
    output->i_target_analog_output_type[i] = input->i_target_analog_output_type[i];
  }
  // b_actual_encorder_strove_signal
  for (size_t i = 0; i < 2; ++i) {
    output->b_actual_encorder_strove_signal[i] = input->b_actual_encorder_strove_signal[i];
  }
  // i_actual_encorder_raw_data
  for (size_t i = 0; i < 2; ++i) {
    output->i_actual_encorder_raw_data[i] = input->i_actual_encorder_raw_data[i];
  }
  // b_actual_encorder_reset_signal
  for (size_t i = 0; i < 2; ++i) {
    output->b_actual_encorder_reset_signal[i] = input->b_actual_encorder_reset_signal[i];
  }
  return true;
}

dsr_msgs2__msg__RobotState *
dsr_msgs2__msg__RobotState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__RobotState * msg = (dsr_msgs2__msg__RobotState *)allocator.allocate(sizeof(dsr_msgs2__msg__RobotState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dsr_msgs2__msg__RobotState));
  bool success = dsr_msgs2__msg__RobotState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dsr_msgs2__msg__RobotState__destroy(dsr_msgs2__msg__RobotState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dsr_msgs2__msg__RobotState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dsr_msgs2__msg__RobotState__Sequence__init(dsr_msgs2__msg__RobotState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__RobotState * data = NULL;

  if (size) {
    data = (dsr_msgs2__msg__RobotState *)allocator.zero_allocate(size, sizeof(dsr_msgs2__msg__RobotState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dsr_msgs2__msg__RobotState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dsr_msgs2__msg__RobotState__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
dsr_msgs2__msg__RobotState__Sequence__fini(dsr_msgs2__msg__RobotState__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      dsr_msgs2__msg__RobotState__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

dsr_msgs2__msg__RobotState__Sequence *
dsr_msgs2__msg__RobotState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__RobotState__Sequence * array = (dsr_msgs2__msg__RobotState__Sequence *)allocator.allocate(sizeof(dsr_msgs2__msg__RobotState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dsr_msgs2__msg__RobotState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dsr_msgs2__msg__RobotState__Sequence__destroy(dsr_msgs2__msg__RobotState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dsr_msgs2__msg__RobotState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dsr_msgs2__msg__RobotState__Sequence__are_equal(const dsr_msgs2__msg__RobotState__Sequence * lhs, const dsr_msgs2__msg__RobotState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dsr_msgs2__msg__RobotState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__msg__RobotState__Sequence__copy(
  const dsr_msgs2__msg__RobotState__Sequence * input,
  dsr_msgs2__msg__RobotState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dsr_msgs2__msg__RobotState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dsr_msgs2__msg__RobotState * data =
      (dsr_msgs2__msg__RobotState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dsr_msgs2__msg__RobotState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dsr_msgs2__msg__RobotState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dsr_msgs2__msg__RobotState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
