// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/robot_state_rt__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "dsr_msgs2/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "dsr_msgs2/msg/detail/robot_state_rt__struct.h"
#include "dsr_msgs2/msg/detail/robot_state_rt__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "std_msgs/msg/detail/float64_multi_array__functions.h"  // coriolis_matrix, jacobian_matrix, mass_matrix

// forward declare type support functions
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_dsr_msgs2
size_t get_serialized_size_std_msgs__msg__Float64MultiArray(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_dsr_msgs2
size_t max_serialized_size_std_msgs__msg__Float64MultiArray(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_dsr_msgs2
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray)();


using _RobotStateRt__ros_msg_type = dsr_msgs2__msg__RobotStateRt;

static bool _RobotStateRt__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _RobotStateRt__ros_msg_type * ros_message = static_cast<const _RobotStateRt__ros_msg_type *>(untyped_ros_message);
  // Field name: time_stamp
  {
    cdr << ros_message->time_stamp;
  }

  // Field name: actual_joint_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_position;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_joint_position_abs
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_position_abs;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_joint_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_velocity;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_joint_velocity_abs
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_velocity_abs;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_tcp_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_tcp_position;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_tcp_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_tcp_velocity;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_flange_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_flange_position;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_flange_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_flange_velocity;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_motor_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_motor_torque;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_joint_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_torque;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: raw_joint_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->raw_joint_torque;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: raw_force_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->raw_force_torque;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: external_joint_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->external_joint_torque;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: external_tcp_force
  {
    size_t size = 6;
    auto array_ptr = ros_message->external_tcp_force;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_joint_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_joint_position;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_joint_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_joint_velocity;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_joint_acceleration
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_joint_acceleration;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_motor_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_motor_torque;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_tcp_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_tcp_position;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_tcp_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_tcp_velocity;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: jacobian_matrix
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    size_t size = ros_message->jacobian_matrix.size;
    auto array_ptr = ros_message->jacobian_matrix.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: gravity_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->gravity_torque;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: coriolis_matrix
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    size_t size = ros_message->coriolis_matrix.size;
    auto array_ptr = ros_message->coriolis_matrix.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: mass_matrix
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    size_t size = ros_message->mass_matrix.size;
    auto array_ptr = ros_message->mass_matrix.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: solution_space
  {
    cdr << ros_message->solution_space;
  }

  // Field name: singularity
  {
    cdr << ros_message->singularity;
  }

  // Field name: operation_speed_rate
  {
    cdr << ros_message->operation_speed_rate;
  }

  // Field name: joint_temperature
  {
    size_t size = 6;
    auto array_ptr = ros_message->joint_temperature;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: controller_digital_input
  {
    cdr << ros_message->controller_digital_input;
  }

  // Field name: controller_digital_output
  {
    cdr << ros_message->controller_digital_output;
  }

  // Field name: controller_analog_input_type
  {
    size_t size = 2;
    auto array_ptr = ros_message->controller_analog_input_type;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: controller_analog_input
  {
    size_t size = 2;
    auto array_ptr = ros_message->controller_analog_input;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: controller_analog_output_type
  {
    size_t size = 2;
    auto array_ptr = ros_message->controller_analog_output_type;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: controller_analog_output
  {
    size_t size = 2;
    auto array_ptr = ros_message->controller_analog_output;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: flange_digital_input
  {
    cdr << ros_message->flange_digital_input;
  }

  // Field name: flange_digital_output
  {
    cdr << ros_message->flange_digital_output;
  }

  // Field name: flange_analog_input
  {
    size_t size = 4;
    auto array_ptr = ros_message->flange_analog_input;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: external_encoder_strobe_count
  {
    size_t size = 2;
    auto array_ptr = ros_message->external_encoder_strobe_count;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: external_encoder_count
  {
    size_t size = 2;
    auto array_ptr = ros_message->external_encoder_count;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: goal_joint_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->goal_joint_position;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: goal_tcp_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->goal_tcp_position;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: robot_mode
  {
    cdr << ros_message->robot_mode;
  }

  // Field name: robot_state
  {
    cdr << ros_message->robot_state;
  }

  // Field name: control_mode
  {
    cdr << ros_message->control_mode;
  }

  // Field name: reserved
  {
    size_t size = 256;
    auto array_ptr = ros_message->reserved;
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _RobotStateRt__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _RobotStateRt__ros_msg_type * ros_message = static_cast<_RobotStateRt__ros_msg_type *>(untyped_ros_message);
  // Field name: time_stamp
  {
    cdr >> ros_message->time_stamp;
  }

  // Field name: actual_joint_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_position;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_joint_position_abs
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_position_abs;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_joint_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_velocity;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_joint_velocity_abs
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_velocity_abs;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_tcp_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_tcp_position;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_tcp_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_tcp_velocity;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_flange_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_flange_position;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_flange_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_flange_velocity;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_motor_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_motor_torque;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_joint_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_joint_torque;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: raw_joint_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->raw_joint_torque;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: raw_force_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->raw_force_torque;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: external_joint_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->external_joint_torque;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: external_tcp_force
  {
    size_t size = 6;
    auto array_ptr = ros_message->external_tcp_force;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_joint_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_joint_position;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_joint_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_joint_velocity;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_joint_acceleration
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_joint_acceleration;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_motor_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_motor_torque;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_tcp_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_tcp_position;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_tcp_velocity
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_tcp_velocity;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: jacobian_matrix
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.getState();
    bool correct_size = cdr.jump(size);
    cdr.setState(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    if (ros_message->jacobian_matrix.data) {
      std_msgs__msg__Float64MultiArray__Sequence__fini(&ros_message->jacobian_matrix);
    }
    if (!std_msgs__msg__Float64MultiArray__Sequence__init(&ros_message->jacobian_matrix, size)) {
      fprintf(stderr, "failed to create array for field 'jacobian_matrix'");
      return false;
    }
    auto array_ptr = ros_message->jacobian_matrix.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: gravity_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->gravity_torque;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: coriolis_matrix
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.getState();
    bool correct_size = cdr.jump(size);
    cdr.setState(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    if (ros_message->coriolis_matrix.data) {
      std_msgs__msg__Float64MultiArray__Sequence__fini(&ros_message->coriolis_matrix);
    }
    if (!std_msgs__msg__Float64MultiArray__Sequence__init(&ros_message->coriolis_matrix, size)) {
      fprintf(stderr, "failed to create array for field 'coriolis_matrix'");
      return false;
    }
    auto array_ptr = ros_message->coriolis_matrix.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: mass_matrix
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);

    // Check there are at least 'size' remaining bytes in the CDR stream before resizing
    auto old_state = cdr.getState();
    bool correct_size = cdr.jump(size);
    cdr.setState(old_state);
    if (!correct_size) {
      fprintf(stderr, "sequence size exceeds remaining buffer\n");
      return false;
    }

    if (ros_message->mass_matrix.data) {
      std_msgs__msg__Float64MultiArray__Sequence__fini(&ros_message->mass_matrix);
    }
    if (!std_msgs__msg__Float64MultiArray__Sequence__init(&ros_message->mass_matrix, size)) {
      fprintf(stderr, "failed to create array for field 'mass_matrix'");
      return false;
    }
    auto array_ptr = ros_message->mass_matrix.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: solution_space
  {
    cdr >> ros_message->solution_space;
  }

  // Field name: singularity
  {
    cdr >> ros_message->singularity;
  }

  // Field name: operation_speed_rate
  {
    cdr >> ros_message->operation_speed_rate;
  }

  // Field name: joint_temperature
  {
    size_t size = 6;
    auto array_ptr = ros_message->joint_temperature;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: controller_digital_input
  {
    cdr >> ros_message->controller_digital_input;
  }

  // Field name: controller_digital_output
  {
    cdr >> ros_message->controller_digital_output;
  }

  // Field name: controller_analog_input_type
  {
    size_t size = 2;
    auto array_ptr = ros_message->controller_analog_input_type;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: controller_analog_input
  {
    size_t size = 2;
    auto array_ptr = ros_message->controller_analog_input;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: controller_analog_output_type
  {
    size_t size = 2;
    auto array_ptr = ros_message->controller_analog_output_type;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: controller_analog_output
  {
    size_t size = 2;
    auto array_ptr = ros_message->controller_analog_output;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: flange_digital_input
  {
    cdr >> ros_message->flange_digital_input;
  }

  // Field name: flange_digital_output
  {
    cdr >> ros_message->flange_digital_output;
  }

  // Field name: flange_analog_input
  {
    size_t size = 4;
    auto array_ptr = ros_message->flange_analog_input;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: external_encoder_strobe_count
  {
    size_t size = 2;
    auto array_ptr = ros_message->external_encoder_strobe_count;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: external_encoder_count
  {
    size_t size = 2;
    auto array_ptr = ros_message->external_encoder_count;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: goal_joint_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->goal_joint_position;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: goal_tcp_position
  {
    size_t size = 6;
    auto array_ptr = ros_message->goal_tcp_position;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: robot_mode
  {
    cdr >> ros_message->robot_mode;
  }

  // Field name: robot_state
  {
    cdr >> ros_message->robot_state;
  }

  // Field name: control_mode
  {
    cdr >> ros_message->control_mode;
  }

  // Field name: reserved
  {
    size_t size = 256;
    auto array_ptr = ros_message->reserved;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t get_serialized_size_dsr_msgs2__msg__RobotStateRt(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _RobotStateRt__ros_msg_type * ros_message = static_cast<const _RobotStateRt__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name time_stamp
  {
    size_t item_size = sizeof(ros_message->time_stamp);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_joint_position
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_joint_position;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_joint_position_abs
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_joint_position_abs;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_joint_velocity
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_joint_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_joint_velocity_abs
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_joint_velocity_abs;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_tcp_position
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_tcp_position;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_tcp_velocity
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_tcp_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_flange_position
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_flange_position;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_flange_velocity
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_flange_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_motor_torque
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_motor_torque;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_joint_torque
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_joint_torque;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name raw_joint_torque
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->raw_joint_torque;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name raw_force_torque
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->raw_force_torque;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name external_joint_torque
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->external_joint_torque;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name external_tcp_force
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->external_tcp_force;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_joint_position
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_joint_position;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_joint_velocity
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_joint_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_joint_acceleration
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_joint_acceleration;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_motor_torque
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_motor_torque;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_tcp_position
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_tcp_position;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_tcp_velocity
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_tcp_velocity;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name jacobian_matrix
  {
    size_t array_size = ros_message->jacobian_matrix.size;
    auto array_ptr = ros_message->jacobian_matrix.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_std_msgs__msg__Float64MultiArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name gravity_torque
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->gravity_torque;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name coriolis_matrix
  {
    size_t array_size = ros_message->coriolis_matrix.size;
    auto array_ptr = ros_message->coriolis_matrix.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_std_msgs__msg__Float64MultiArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name mass_matrix
  {
    size_t array_size = ros_message->mass_matrix.size;
    auto array_ptr = ros_message->mass_matrix.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_std_msgs__msg__Float64MultiArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name solution_space
  {
    size_t item_size = sizeof(ros_message->solution_space);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name singularity
  {
    size_t item_size = sizeof(ros_message->singularity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name operation_speed_rate
  {
    size_t item_size = sizeof(ros_message->operation_speed_rate);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name joint_temperature
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->joint_temperature;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name controller_digital_input
  {
    size_t item_size = sizeof(ros_message->controller_digital_input);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name controller_digital_output
  {
    size_t item_size = sizeof(ros_message->controller_digital_output);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name controller_analog_input_type
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->controller_analog_input_type;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name controller_analog_input
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->controller_analog_input;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name controller_analog_output_type
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->controller_analog_output_type;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name controller_analog_output
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->controller_analog_output;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name flange_digital_input
  {
    size_t item_size = sizeof(ros_message->flange_digital_input);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name flange_digital_output
  {
    size_t item_size = sizeof(ros_message->flange_digital_output);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name flange_analog_input
  {
    size_t array_size = 4;
    auto array_ptr = ros_message->flange_analog_input;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name external_encoder_strobe_count
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->external_encoder_strobe_count;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name external_encoder_count
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->external_encoder_count;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name goal_joint_position
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->goal_joint_position;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name goal_tcp_position
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->goal_tcp_position;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name robot_mode
  {
    size_t item_size = sizeof(ros_message->robot_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name robot_state
  {
    size_t item_size = sizeof(ros_message->robot_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name control_mode
  {
    size_t item_size = sizeof(ros_message->control_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name reserved
  {
    size_t array_size = 256;
    auto array_ptr = ros_message->reserved;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _RobotStateRt__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_dsr_msgs2__msg__RobotStateRt(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t max_serialized_size_dsr_msgs2__msg__RobotStateRt(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // member: time_stamp
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_joint_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_joint_position_abs
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_joint_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_joint_velocity_abs
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_tcp_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_tcp_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_flange_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_flange_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_motor_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_joint_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: raw_joint_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: raw_force_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: external_joint_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: external_tcp_force
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_joint_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_joint_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_joint_acceleration
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_motor_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_tcp_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_tcp_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: jacobian_matrix
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_std_msgs__msg__Float64MultiArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: gravity_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: coriolis_matrix
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_std_msgs__msg__Float64MultiArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: mass_matrix
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    last_member_size = 0;
    for (size_t index = 0; index < array_size; ++index) {
      bool inner_full_bounded;
      bool inner_is_plain;
      size_t inner_size;
      inner_size =
        max_serialized_size_std_msgs__msg__Float64MultiArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: solution_space
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }
  // member: singularity
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: operation_speed_rate
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: joint_temperature
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: controller_digital_input
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }
  // member: controller_digital_output
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }
  // member: controller_analog_input_type
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: controller_analog_input
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: controller_analog_output_type
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: controller_analog_output
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: flange_digital_input
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: flange_digital_output
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: flange_analog_input
  {
    size_t array_size = 4;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: external_encoder_strobe_count
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: external_encoder_count
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }
  // member: goal_joint_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: goal_tcp_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: robot_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: robot_state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: control_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }
  // member: reserved
  {
    size_t array_size = 256;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = dsr_msgs2__msg__RobotStateRt;
    is_plain =
      (
      offsetof(DataType, reserved) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _RobotStateRt__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_dsr_msgs2__msg__RobotStateRt(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_RobotStateRt = {
  "dsr_msgs2::msg",
  "RobotStateRt",
  _RobotStateRt__cdr_serialize,
  _RobotStateRt__cdr_deserialize,
  _RobotStateRt__get_serialized_size,
  _RobotStateRt__max_serialized_size
};

static rosidl_message_type_support_t _RobotStateRt__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_RobotStateRt,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, msg, RobotStateRt)() {
  return &_RobotStateRt__type_support;
}

#if defined(__cplusplus)
}
#endif
