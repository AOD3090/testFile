// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/robot_state_rt__rosidl_typesupport_fastrtps_cpp.hpp"
#include "dsr_msgs2/msg/detail/robot_state_rt__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions
namespace std_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const std_msgs::msg::Float64MultiArray &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  std_msgs::msg::Float64MultiArray &);
size_t get_serialized_size(
  const std_msgs::msg::Float64MultiArray &,
  size_t current_alignment);
size_t
max_serialized_size_Float64MultiArray(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace std_msgs

// functions for std_msgs::msg::Float64MultiArray already declared above

// functions for std_msgs::msg::Float64MultiArray already declared above


namespace dsr_msgs2
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
cdr_serialize(
  const dsr_msgs2::msg::RobotStateRt & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: time_stamp
  cdr << ros_message.time_stamp;
  // Member: actual_joint_position
  {
    cdr << ros_message.actual_joint_position;
  }
  // Member: actual_joint_position_abs
  {
    cdr << ros_message.actual_joint_position_abs;
  }
  // Member: actual_joint_velocity
  {
    cdr << ros_message.actual_joint_velocity;
  }
  // Member: actual_joint_velocity_abs
  {
    cdr << ros_message.actual_joint_velocity_abs;
  }
  // Member: actual_tcp_position
  {
    cdr << ros_message.actual_tcp_position;
  }
  // Member: actual_tcp_velocity
  {
    cdr << ros_message.actual_tcp_velocity;
  }
  // Member: actual_flange_position
  {
    cdr << ros_message.actual_flange_position;
  }
  // Member: actual_flange_velocity
  {
    cdr << ros_message.actual_flange_velocity;
  }
  // Member: actual_motor_torque
  {
    cdr << ros_message.actual_motor_torque;
  }
  // Member: actual_joint_torque
  {
    cdr << ros_message.actual_joint_torque;
  }
  // Member: raw_joint_torque
  {
    cdr << ros_message.raw_joint_torque;
  }
  // Member: raw_force_torque
  {
    cdr << ros_message.raw_force_torque;
  }
  // Member: external_joint_torque
  {
    cdr << ros_message.external_joint_torque;
  }
  // Member: external_tcp_force
  {
    cdr << ros_message.external_tcp_force;
  }
  // Member: target_joint_position
  {
    cdr << ros_message.target_joint_position;
  }
  // Member: target_joint_velocity
  {
    cdr << ros_message.target_joint_velocity;
  }
  // Member: target_joint_acceleration
  {
    cdr << ros_message.target_joint_acceleration;
  }
  // Member: target_motor_torque
  {
    cdr << ros_message.target_motor_torque;
  }
  // Member: target_tcp_position
  {
    cdr << ros_message.target_tcp_position;
  }
  // Member: target_tcp_velocity
  {
    cdr << ros_message.target_tcp_velocity;
  }
  // Member: jacobian_matrix
  {
    size_t size = ros_message.jacobian_matrix.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.jacobian_matrix[i],
        cdr);
    }
  }
  // Member: gravity_torque
  {
    cdr << ros_message.gravity_torque;
  }
  // Member: coriolis_matrix
  {
    size_t size = ros_message.coriolis_matrix.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.coriolis_matrix[i],
        cdr);
    }
  }
  // Member: mass_matrix
  {
    size_t size = ros_message.mass_matrix.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.mass_matrix[i],
        cdr);
    }
  }
  // Member: solution_space
  cdr << ros_message.solution_space;
  // Member: singularity
  cdr << ros_message.singularity;
  // Member: operation_speed_rate
  cdr << ros_message.operation_speed_rate;
  // Member: joint_temperature
  {
    cdr << ros_message.joint_temperature;
  }
  // Member: controller_digital_input
  cdr << ros_message.controller_digital_input;
  // Member: controller_digital_output
  cdr << ros_message.controller_digital_output;
  // Member: controller_analog_input_type
  {
    cdr << ros_message.controller_analog_input_type;
  }
  // Member: controller_analog_input
  {
    cdr << ros_message.controller_analog_input;
  }
  // Member: controller_analog_output_type
  {
    cdr << ros_message.controller_analog_output_type;
  }
  // Member: controller_analog_output
  {
    cdr << ros_message.controller_analog_output;
  }
  // Member: flange_digital_input
  cdr << ros_message.flange_digital_input;
  // Member: flange_digital_output
  cdr << ros_message.flange_digital_output;
  // Member: flange_analog_input
  {
    cdr << ros_message.flange_analog_input;
  }
  // Member: external_encoder_strobe_count
  {
    cdr << ros_message.external_encoder_strobe_count;
  }
  // Member: external_encoder_count
  {
    cdr << ros_message.external_encoder_count;
  }
  // Member: goal_joint_position
  {
    cdr << ros_message.goal_joint_position;
  }
  // Member: goal_tcp_position
  {
    cdr << ros_message.goal_tcp_position;
  }
  // Member: robot_mode
  cdr << ros_message.robot_mode;
  // Member: robot_state
  cdr << ros_message.robot_state;
  // Member: control_mode
  cdr << ros_message.control_mode;
  // Member: reserved
  {
    cdr << ros_message.reserved;
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  dsr_msgs2::msg::RobotStateRt & ros_message)
{
  // Member: time_stamp
  cdr >> ros_message.time_stamp;

  // Member: actual_joint_position
  {
    cdr >> ros_message.actual_joint_position;
  }

  // Member: actual_joint_position_abs
  {
    cdr >> ros_message.actual_joint_position_abs;
  }

  // Member: actual_joint_velocity
  {
    cdr >> ros_message.actual_joint_velocity;
  }

  // Member: actual_joint_velocity_abs
  {
    cdr >> ros_message.actual_joint_velocity_abs;
  }

  // Member: actual_tcp_position
  {
    cdr >> ros_message.actual_tcp_position;
  }

  // Member: actual_tcp_velocity
  {
    cdr >> ros_message.actual_tcp_velocity;
  }

  // Member: actual_flange_position
  {
    cdr >> ros_message.actual_flange_position;
  }

  // Member: actual_flange_velocity
  {
    cdr >> ros_message.actual_flange_velocity;
  }

  // Member: actual_motor_torque
  {
    cdr >> ros_message.actual_motor_torque;
  }

  // Member: actual_joint_torque
  {
    cdr >> ros_message.actual_joint_torque;
  }

  // Member: raw_joint_torque
  {
    cdr >> ros_message.raw_joint_torque;
  }

  // Member: raw_force_torque
  {
    cdr >> ros_message.raw_force_torque;
  }

  // Member: external_joint_torque
  {
    cdr >> ros_message.external_joint_torque;
  }

  // Member: external_tcp_force
  {
    cdr >> ros_message.external_tcp_force;
  }

  // Member: target_joint_position
  {
    cdr >> ros_message.target_joint_position;
  }

  // Member: target_joint_velocity
  {
    cdr >> ros_message.target_joint_velocity;
  }

  // Member: target_joint_acceleration
  {
    cdr >> ros_message.target_joint_acceleration;
  }

  // Member: target_motor_torque
  {
    cdr >> ros_message.target_motor_torque;
  }

  // Member: target_tcp_position
  {
    cdr >> ros_message.target_tcp_position;
  }

  // Member: target_tcp_velocity
  {
    cdr >> ros_message.target_tcp_velocity;
  }

  // Member: jacobian_matrix
  {
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

    ros_message.jacobian_matrix.resize(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.jacobian_matrix[i]);
    }
  }

  // Member: gravity_torque
  {
    cdr >> ros_message.gravity_torque;
  }

  // Member: coriolis_matrix
  {
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

    ros_message.coriolis_matrix.resize(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.coriolis_matrix[i]);
    }
  }

  // Member: mass_matrix
  {
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

    ros_message.mass_matrix.resize(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.mass_matrix[i]);
    }
  }

  // Member: solution_space
  cdr >> ros_message.solution_space;

  // Member: singularity
  cdr >> ros_message.singularity;

  // Member: operation_speed_rate
  cdr >> ros_message.operation_speed_rate;

  // Member: joint_temperature
  {
    cdr >> ros_message.joint_temperature;
  }

  // Member: controller_digital_input
  cdr >> ros_message.controller_digital_input;

  // Member: controller_digital_output
  cdr >> ros_message.controller_digital_output;

  // Member: controller_analog_input_type
  {
    cdr >> ros_message.controller_analog_input_type;
  }

  // Member: controller_analog_input
  {
    cdr >> ros_message.controller_analog_input;
  }

  // Member: controller_analog_output_type
  {
    cdr >> ros_message.controller_analog_output_type;
  }

  // Member: controller_analog_output
  {
    cdr >> ros_message.controller_analog_output;
  }

  // Member: flange_digital_input
  cdr >> ros_message.flange_digital_input;

  // Member: flange_digital_output
  cdr >> ros_message.flange_digital_output;

  // Member: flange_analog_input
  {
    cdr >> ros_message.flange_analog_input;
  }

  // Member: external_encoder_strobe_count
  {
    cdr >> ros_message.external_encoder_strobe_count;
  }

  // Member: external_encoder_count
  {
    cdr >> ros_message.external_encoder_count;
  }

  // Member: goal_joint_position
  {
    cdr >> ros_message.goal_joint_position;
  }

  // Member: goal_tcp_position
  {
    cdr >> ros_message.goal_tcp_position;
  }

  // Member: robot_mode
  cdr >> ros_message.robot_mode;

  // Member: robot_state
  cdr >> ros_message.robot_state;

  // Member: control_mode
  cdr >> ros_message.control_mode;

  // Member: reserved
  {
    cdr >> ros_message.reserved;
  }

  return true;
}  // NOLINT(readability/fn_size)

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
get_serialized_size(
  const dsr_msgs2::msg::RobotStateRt & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: time_stamp
  {
    size_t item_size = sizeof(ros_message.time_stamp);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_joint_position
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_joint_position[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_joint_position_abs
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_joint_position_abs[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_joint_velocity
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_joint_velocity[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_joint_velocity_abs
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_joint_velocity_abs[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_tcp_position
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_tcp_position[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_tcp_velocity
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_tcp_velocity[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_flange_position
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_flange_position[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_flange_velocity
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_flange_velocity[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_motor_torque
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_motor_torque[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_joint_torque
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_joint_torque[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: raw_joint_torque
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.raw_joint_torque[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: raw_force_torque
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.raw_force_torque[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: external_joint_torque
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.external_joint_torque[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: external_tcp_force
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.external_tcp_force[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_joint_position
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_joint_position[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_joint_velocity
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_joint_velocity[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_joint_acceleration
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_joint_acceleration[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_motor_torque
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_motor_torque[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_tcp_position
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_tcp_position[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_tcp_velocity
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_tcp_velocity[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: jacobian_matrix
  {
    size_t array_size = ros_message.jacobian_matrix.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.jacobian_matrix[index], current_alignment);
    }
  }
  // Member: gravity_torque
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.gravity_torque[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: coriolis_matrix
  {
    size_t array_size = ros_message.coriolis_matrix.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.coriolis_matrix[index], current_alignment);
    }
  }
  // Member: mass_matrix
  {
    size_t array_size = ros_message.mass_matrix.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.mass_matrix[index], current_alignment);
    }
  }
  // Member: solution_space
  {
    size_t item_size = sizeof(ros_message.solution_space);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: singularity
  {
    size_t item_size = sizeof(ros_message.singularity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: operation_speed_rate
  {
    size_t item_size = sizeof(ros_message.operation_speed_rate);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: joint_temperature
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.joint_temperature[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: controller_digital_input
  {
    size_t item_size = sizeof(ros_message.controller_digital_input);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: controller_digital_output
  {
    size_t item_size = sizeof(ros_message.controller_digital_output);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: controller_analog_input_type
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.controller_analog_input_type[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: controller_analog_input
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.controller_analog_input[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: controller_analog_output_type
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.controller_analog_output_type[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: controller_analog_output
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.controller_analog_output[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: flange_digital_input
  {
    size_t item_size = sizeof(ros_message.flange_digital_input);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: flange_digital_output
  {
    size_t item_size = sizeof(ros_message.flange_digital_output);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: flange_analog_input
  {
    size_t array_size = 4;
    size_t item_size = sizeof(ros_message.flange_analog_input[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: external_encoder_strobe_count
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.external_encoder_strobe_count[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: external_encoder_count
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.external_encoder_count[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: goal_joint_position
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.goal_joint_position[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: goal_tcp_position
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.goal_tcp_position[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: robot_mode
  {
    size_t item_size = sizeof(ros_message.robot_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: robot_state
  {
    size_t item_size = sizeof(ros_message.robot_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: control_mode
  {
    size_t item_size = sizeof(ros_message.control_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: reserved
  {
    size_t array_size = 256;
    size_t item_size = sizeof(ros_message.reserved[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
max_serialized_size_RobotStateRt(
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


  // Member: time_stamp
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_joint_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_joint_position_abs
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_joint_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_joint_velocity_abs
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_tcp_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_tcp_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_flange_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_flange_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_motor_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_joint_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: raw_joint_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: raw_force_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: external_joint_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: external_tcp_force
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_joint_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_joint_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_joint_acceleration
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_motor_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_tcp_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_tcp_velocity
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: jacobian_matrix
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
      size_t inner_size =
        std_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_Float64MultiArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: gravity_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: coriolis_matrix
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
      size_t inner_size =
        std_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_Float64MultiArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: mass_matrix
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
      size_t inner_size =
        std_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_Float64MultiArray(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: solution_space
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Member: singularity
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: operation_speed_rate
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: joint_temperature
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: controller_digital_input
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Member: controller_digital_output
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Member: controller_analog_input_type
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: controller_analog_input
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: controller_analog_output_type
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: controller_analog_output
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: flange_digital_input
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: flange_digital_output
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: flange_analog_input
  {
    size_t array_size = 4;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: external_encoder_strobe_count
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: external_encoder_count
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Member: goal_joint_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: goal_tcp_position
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: robot_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: robot_state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: control_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint16_t);
    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Member: reserved
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
    using DataType = dsr_msgs2::msg::RobotStateRt;
    is_plain =
      (
      offsetof(DataType, reserved) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _RobotStateRt__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const dsr_msgs2::msg::RobotStateRt *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _RobotStateRt__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<dsr_msgs2::msg::RobotStateRt *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _RobotStateRt__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const dsr_msgs2::msg::RobotStateRt *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _RobotStateRt__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_RobotStateRt(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _RobotStateRt__callbacks = {
  "dsr_msgs2::msg",
  "RobotStateRt",
  _RobotStateRt__cdr_serialize,
  _RobotStateRt__cdr_deserialize,
  _RobotStateRt__get_serialized_size,
  _RobotStateRt__max_serialized_size
};

static rosidl_message_type_support_t _RobotStateRt__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_RobotStateRt__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace dsr_msgs2

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_dsr_msgs2
const rosidl_message_type_support_t *
get_message_type_support_handle<dsr_msgs2::msg::RobotStateRt>()
{
  return &dsr_msgs2::msg::typesupport_fastrtps_cpp::_RobotStateRt__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, dsr_msgs2, msg, RobotStateRt)() {
  return &dsr_msgs2::msg::typesupport_fastrtps_cpp::_RobotStateRt__handle;
}

#ifdef __cplusplus
}
#endif
