// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/robot_state_rt__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `jacobian_matrix`
// Member `coriolis_matrix`
// Member `mass_matrix`
#include "std_msgs/msg/detail/float64_multi_array__functions.h"

bool
dsr_msgs2__msg__RobotStateRt__init(dsr_msgs2__msg__RobotStateRt * msg)
{
  if (!msg) {
    return false;
  }
  // time_stamp
  // actual_joint_position
  // actual_joint_position_abs
  // actual_joint_velocity
  // actual_joint_velocity_abs
  // actual_tcp_position
  // actual_tcp_velocity
  // actual_flange_position
  // actual_flange_velocity
  // actual_motor_torque
  // actual_joint_torque
  // raw_joint_torque
  // raw_force_torque
  // external_joint_torque
  // external_tcp_force
  // target_joint_position
  // target_joint_velocity
  // target_joint_acceleration
  // target_motor_torque
  // target_tcp_position
  // target_tcp_velocity
  // jacobian_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__init(&msg->jacobian_matrix, 0)) {
    dsr_msgs2__msg__RobotStateRt__fini(msg);
    return false;
  }
  // gravity_torque
  // coriolis_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__init(&msg->coriolis_matrix, 0)) {
    dsr_msgs2__msg__RobotStateRt__fini(msg);
    return false;
  }
  // mass_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__init(&msg->mass_matrix, 0)) {
    dsr_msgs2__msg__RobotStateRt__fini(msg);
    return false;
  }
  // solution_space
  // singularity
  // operation_speed_rate
  // joint_temperature
  // controller_digital_input
  // controller_digital_output
  // controller_analog_input_type
  // controller_analog_input
  // controller_analog_output_type
  // controller_analog_output
  // flange_digital_input
  // flange_digital_output
  // flange_analog_input
  // external_encoder_strobe_count
  // external_encoder_count
  // goal_joint_position
  // goal_tcp_position
  // robot_mode
  // robot_state
  // control_mode
  // reserved
  return true;
}

void
dsr_msgs2__msg__RobotStateRt__fini(dsr_msgs2__msg__RobotStateRt * msg)
{
  if (!msg) {
    return;
  }
  // time_stamp
  // actual_joint_position
  // actual_joint_position_abs
  // actual_joint_velocity
  // actual_joint_velocity_abs
  // actual_tcp_position
  // actual_tcp_velocity
  // actual_flange_position
  // actual_flange_velocity
  // actual_motor_torque
  // actual_joint_torque
  // raw_joint_torque
  // raw_force_torque
  // external_joint_torque
  // external_tcp_force
  // target_joint_position
  // target_joint_velocity
  // target_joint_acceleration
  // target_motor_torque
  // target_tcp_position
  // target_tcp_velocity
  // jacobian_matrix
  std_msgs__msg__Float64MultiArray__Sequence__fini(&msg->jacobian_matrix);
  // gravity_torque
  // coriolis_matrix
  std_msgs__msg__Float64MultiArray__Sequence__fini(&msg->coriolis_matrix);
  // mass_matrix
  std_msgs__msg__Float64MultiArray__Sequence__fini(&msg->mass_matrix);
  // solution_space
  // singularity
  // operation_speed_rate
  // joint_temperature
  // controller_digital_input
  // controller_digital_output
  // controller_analog_input_type
  // controller_analog_input
  // controller_analog_output_type
  // controller_analog_output
  // flange_digital_input
  // flange_digital_output
  // flange_analog_input
  // external_encoder_strobe_count
  // external_encoder_count
  // goal_joint_position
  // goal_tcp_position
  // robot_mode
  // robot_state
  // control_mode
  // reserved
}

bool
dsr_msgs2__msg__RobotStateRt__are_equal(const dsr_msgs2__msg__RobotStateRt * lhs, const dsr_msgs2__msg__RobotStateRt * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // time_stamp
  if (lhs->time_stamp != rhs->time_stamp) {
    return false;
  }
  // actual_joint_position
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_joint_position[i] != rhs->actual_joint_position[i]) {
      return false;
    }
  }
  // actual_joint_position_abs
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_joint_position_abs[i] != rhs->actual_joint_position_abs[i]) {
      return false;
    }
  }
  // actual_joint_velocity
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_joint_velocity[i] != rhs->actual_joint_velocity[i]) {
      return false;
    }
  }
  // actual_joint_velocity_abs
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_joint_velocity_abs[i] != rhs->actual_joint_velocity_abs[i]) {
      return false;
    }
  }
  // actual_tcp_position
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_tcp_position[i] != rhs->actual_tcp_position[i]) {
      return false;
    }
  }
  // actual_tcp_velocity
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_tcp_velocity[i] != rhs->actual_tcp_velocity[i]) {
      return false;
    }
  }
  // actual_flange_position
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_flange_position[i] != rhs->actual_flange_position[i]) {
      return false;
    }
  }
  // actual_flange_velocity
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_flange_velocity[i] != rhs->actual_flange_velocity[i]) {
      return false;
    }
  }
  // actual_motor_torque
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_motor_torque[i] != rhs->actual_motor_torque[i]) {
      return false;
    }
  }
  // actual_joint_torque
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->actual_joint_torque[i] != rhs->actual_joint_torque[i]) {
      return false;
    }
  }
  // raw_joint_torque
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->raw_joint_torque[i] != rhs->raw_joint_torque[i]) {
      return false;
    }
  }
  // raw_force_torque
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->raw_force_torque[i] != rhs->raw_force_torque[i]) {
      return false;
    }
  }
  // external_joint_torque
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->external_joint_torque[i] != rhs->external_joint_torque[i]) {
      return false;
    }
  }
  // external_tcp_force
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->external_tcp_force[i] != rhs->external_tcp_force[i]) {
      return false;
    }
  }
  // target_joint_position
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_joint_position[i] != rhs->target_joint_position[i]) {
      return false;
    }
  }
  // target_joint_velocity
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_joint_velocity[i] != rhs->target_joint_velocity[i]) {
      return false;
    }
  }
  // target_joint_acceleration
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_joint_acceleration[i] != rhs->target_joint_acceleration[i]) {
      return false;
    }
  }
  // target_motor_torque
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_motor_torque[i] != rhs->target_motor_torque[i]) {
      return false;
    }
  }
  // target_tcp_position
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_tcp_position[i] != rhs->target_tcp_position[i]) {
      return false;
    }
  }
  // target_tcp_velocity
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->target_tcp_velocity[i] != rhs->target_tcp_velocity[i]) {
      return false;
    }
  }
  // jacobian_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__are_equal(
      &(lhs->jacobian_matrix), &(rhs->jacobian_matrix)))
  {
    return false;
  }
  // gravity_torque
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->gravity_torque[i] != rhs->gravity_torque[i]) {
      return false;
    }
  }
  // coriolis_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__are_equal(
      &(lhs->coriolis_matrix), &(rhs->coriolis_matrix)))
  {
    return false;
  }
  // mass_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__are_equal(
      &(lhs->mass_matrix), &(rhs->mass_matrix)))
  {
    return false;
  }
  // solution_space
  if (lhs->solution_space != rhs->solution_space) {
    return false;
  }
  // singularity
  if (lhs->singularity != rhs->singularity) {
    return false;
  }
  // operation_speed_rate
  if (lhs->operation_speed_rate != rhs->operation_speed_rate) {
    return false;
  }
  // joint_temperature
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->joint_temperature[i] != rhs->joint_temperature[i]) {
      return false;
    }
  }
  // controller_digital_input
  if (lhs->controller_digital_input != rhs->controller_digital_input) {
    return false;
  }
  // controller_digital_output
  if (lhs->controller_digital_output != rhs->controller_digital_output) {
    return false;
  }
  // controller_analog_input_type
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->controller_analog_input_type[i] != rhs->controller_analog_input_type[i]) {
      return false;
    }
  }
  // controller_analog_input
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->controller_analog_input[i] != rhs->controller_analog_input[i]) {
      return false;
    }
  }
  // controller_analog_output_type
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->controller_analog_output_type[i] != rhs->controller_analog_output_type[i]) {
      return false;
    }
  }
  // controller_analog_output
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->controller_analog_output[i] != rhs->controller_analog_output[i]) {
      return false;
    }
  }
  // flange_digital_input
  if (lhs->flange_digital_input != rhs->flange_digital_input) {
    return false;
  }
  // flange_digital_output
  if (lhs->flange_digital_output != rhs->flange_digital_output) {
    return false;
  }
  // flange_analog_input
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->flange_analog_input[i] != rhs->flange_analog_input[i]) {
      return false;
    }
  }
  // external_encoder_strobe_count
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->external_encoder_strobe_count[i] != rhs->external_encoder_strobe_count[i]) {
      return false;
    }
  }
  // external_encoder_count
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->external_encoder_count[i] != rhs->external_encoder_count[i]) {
      return false;
    }
  }
  // goal_joint_position
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->goal_joint_position[i] != rhs->goal_joint_position[i]) {
      return false;
    }
  }
  // goal_tcp_position
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->goal_tcp_position[i] != rhs->goal_tcp_position[i]) {
      return false;
    }
  }
  // robot_mode
  if (lhs->robot_mode != rhs->robot_mode) {
    return false;
  }
  // robot_state
  if (lhs->robot_state != rhs->robot_state) {
    return false;
  }
  // control_mode
  if (lhs->control_mode != rhs->control_mode) {
    return false;
  }
  // reserved
  for (size_t i = 0; i < 256; ++i) {
    if (lhs->reserved[i] != rhs->reserved[i]) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__msg__RobotStateRt__copy(
  const dsr_msgs2__msg__RobotStateRt * input,
  dsr_msgs2__msg__RobotStateRt * output)
{
  if (!input || !output) {
    return false;
  }
  // time_stamp
  output->time_stamp = input->time_stamp;
  // actual_joint_position
  for (size_t i = 0; i < 6; ++i) {
    output->actual_joint_position[i] = input->actual_joint_position[i];
  }
  // actual_joint_position_abs
  for (size_t i = 0; i < 6; ++i) {
    output->actual_joint_position_abs[i] = input->actual_joint_position_abs[i];
  }
  // actual_joint_velocity
  for (size_t i = 0; i < 6; ++i) {
    output->actual_joint_velocity[i] = input->actual_joint_velocity[i];
  }
  // actual_joint_velocity_abs
  for (size_t i = 0; i < 6; ++i) {
    output->actual_joint_velocity_abs[i] = input->actual_joint_velocity_abs[i];
  }
  // actual_tcp_position
  for (size_t i = 0; i < 6; ++i) {
    output->actual_tcp_position[i] = input->actual_tcp_position[i];
  }
  // actual_tcp_velocity
  for (size_t i = 0; i < 6; ++i) {
    output->actual_tcp_velocity[i] = input->actual_tcp_velocity[i];
  }
  // actual_flange_position
  for (size_t i = 0; i < 6; ++i) {
    output->actual_flange_position[i] = input->actual_flange_position[i];
  }
  // actual_flange_velocity
  for (size_t i = 0; i < 6; ++i) {
    output->actual_flange_velocity[i] = input->actual_flange_velocity[i];
  }
  // actual_motor_torque
  for (size_t i = 0; i < 6; ++i) {
    output->actual_motor_torque[i] = input->actual_motor_torque[i];
  }
  // actual_joint_torque
  for (size_t i = 0; i < 6; ++i) {
    output->actual_joint_torque[i] = input->actual_joint_torque[i];
  }
  // raw_joint_torque
  for (size_t i = 0; i < 6; ++i) {
    output->raw_joint_torque[i] = input->raw_joint_torque[i];
  }
  // raw_force_torque
  for (size_t i = 0; i < 6; ++i) {
    output->raw_force_torque[i] = input->raw_force_torque[i];
  }
  // external_joint_torque
  for (size_t i = 0; i < 6; ++i) {
    output->external_joint_torque[i] = input->external_joint_torque[i];
  }
  // external_tcp_force
  for (size_t i = 0; i < 6; ++i) {
    output->external_tcp_force[i] = input->external_tcp_force[i];
  }
  // target_joint_position
  for (size_t i = 0; i < 6; ++i) {
    output->target_joint_position[i] = input->target_joint_position[i];
  }
  // target_joint_velocity
  for (size_t i = 0; i < 6; ++i) {
    output->target_joint_velocity[i] = input->target_joint_velocity[i];
  }
  // target_joint_acceleration
  for (size_t i = 0; i < 6; ++i) {
    output->target_joint_acceleration[i] = input->target_joint_acceleration[i];
  }
  // target_motor_torque
  for (size_t i = 0; i < 6; ++i) {
    output->target_motor_torque[i] = input->target_motor_torque[i];
  }
  // target_tcp_position
  for (size_t i = 0; i < 6; ++i) {
    output->target_tcp_position[i] = input->target_tcp_position[i];
  }
  // target_tcp_velocity
  for (size_t i = 0; i < 6; ++i) {
    output->target_tcp_velocity[i] = input->target_tcp_velocity[i];
  }
  // jacobian_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__copy(
      &(input->jacobian_matrix), &(output->jacobian_matrix)))
  {
    return false;
  }
  // gravity_torque
  for (size_t i = 0; i < 6; ++i) {
    output->gravity_torque[i] = input->gravity_torque[i];
  }
  // coriolis_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__copy(
      &(input->coriolis_matrix), &(output->coriolis_matrix)))
  {
    return false;
  }
  // mass_matrix
  if (!std_msgs__msg__Float64MultiArray__Sequence__copy(
      &(input->mass_matrix), &(output->mass_matrix)))
  {
    return false;
  }
  // solution_space
  output->solution_space = input->solution_space;
  // singularity
  output->singularity = input->singularity;
  // operation_speed_rate
  output->operation_speed_rate = input->operation_speed_rate;
  // joint_temperature
  for (size_t i = 0; i < 6; ++i) {
    output->joint_temperature[i] = input->joint_temperature[i];
  }
  // controller_digital_input
  output->controller_digital_input = input->controller_digital_input;
  // controller_digital_output
  output->controller_digital_output = input->controller_digital_output;
  // controller_analog_input_type
  for (size_t i = 0; i < 2; ++i) {
    output->controller_analog_input_type[i] = input->controller_analog_input_type[i];
  }
  // controller_analog_input
  for (size_t i = 0; i < 2; ++i) {
    output->controller_analog_input[i] = input->controller_analog_input[i];
  }
  // controller_analog_output_type
  for (size_t i = 0; i < 2; ++i) {
    output->controller_analog_output_type[i] = input->controller_analog_output_type[i];
  }
  // controller_analog_output
  for (size_t i = 0; i < 2; ++i) {
    output->controller_analog_output[i] = input->controller_analog_output[i];
  }
  // flange_digital_input
  output->flange_digital_input = input->flange_digital_input;
  // flange_digital_output
  output->flange_digital_output = input->flange_digital_output;
  // flange_analog_input
  for (size_t i = 0; i < 4; ++i) {
    output->flange_analog_input[i] = input->flange_analog_input[i];
  }
  // external_encoder_strobe_count
  for (size_t i = 0; i < 2; ++i) {
    output->external_encoder_strobe_count[i] = input->external_encoder_strobe_count[i];
  }
  // external_encoder_count
  for (size_t i = 0; i < 2; ++i) {
    output->external_encoder_count[i] = input->external_encoder_count[i];
  }
  // goal_joint_position
  for (size_t i = 0; i < 6; ++i) {
    output->goal_joint_position[i] = input->goal_joint_position[i];
  }
  // goal_tcp_position
  for (size_t i = 0; i < 6; ++i) {
    output->goal_tcp_position[i] = input->goal_tcp_position[i];
  }
  // robot_mode
  output->robot_mode = input->robot_mode;
  // robot_state
  output->robot_state = input->robot_state;
  // control_mode
  output->control_mode = input->control_mode;
  // reserved
  for (size_t i = 0; i < 256; ++i) {
    output->reserved[i] = input->reserved[i];
  }
  return true;
}

dsr_msgs2__msg__RobotStateRt *
dsr_msgs2__msg__RobotStateRt__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__RobotStateRt * msg = (dsr_msgs2__msg__RobotStateRt *)allocator.allocate(sizeof(dsr_msgs2__msg__RobotStateRt), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dsr_msgs2__msg__RobotStateRt));
  bool success = dsr_msgs2__msg__RobotStateRt__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dsr_msgs2__msg__RobotStateRt__destroy(dsr_msgs2__msg__RobotStateRt * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dsr_msgs2__msg__RobotStateRt__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dsr_msgs2__msg__RobotStateRt__Sequence__init(dsr_msgs2__msg__RobotStateRt__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__RobotStateRt * data = NULL;

  if (size) {
    data = (dsr_msgs2__msg__RobotStateRt *)allocator.zero_allocate(size, sizeof(dsr_msgs2__msg__RobotStateRt), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dsr_msgs2__msg__RobotStateRt__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dsr_msgs2__msg__RobotStateRt__fini(&data[i - 1]);
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
dsr_msgs2__msg__RobotStateRt__Sequence__fini(dsr_msgs2__msg__RobotStateRt__Sequence * array)
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
      dsr_msgs2__msg__RobotStateRt__fini(&array->data[i]);
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

dsr_msgs2__msg__RobotStateRt__Sequence *
dsr_msgs2__msg__RobotStateRt__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__RobotStateRt__Sequence * array = (dsr_msgs2__msg__RobotStateRt__Sequence *)allocator.allocate(sizeof(dsr_msgs2__msg__RobotStateRt__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dsr_msgs2__msg__RobotStateRt__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dsr_msgs2__msg__RobotStateRt__Sequence__destroy(dsr_msgs2__msg__RobotStateRt__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dsr_msgs2__msg__RobotStateRt__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dsr_msgs2__msg__RobotStateRt__Sequence__are_equal(const dsr_msgs2__msg__RobotStateRt__Sequence * lhs, const dsr_msgs2__msg__RobotStateRt__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dsr_msgs2__msg__RobotStateRt__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__msg__RobotStateRt__Sequence__copy(
  const dsr_msgs2__msg__RobotStateRt__Sequence * input,
  dsr_msgs2__msg__RobotStateRt__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dsr_msgs2__msg__RobotStateRt);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dsr_msgs2__msg__RobotStateRt * data =
      (dsr_msgs2__msg__RobotStateRt *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dsr_msgs2__msg__RobotStateRt__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dsr_msgs2__msg__RobotStateRt__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dsr_msgs2__msg__RobotStateRt__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
