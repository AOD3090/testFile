// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from dsr_msgs2:msg/RobotState.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/robot_state__rosidl_typesupport_fastrtps_cpp.hpp"
#include "dsr_msgs2/msg/detail/robot_state__struct.hpp"

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

namespace dsr_msgs2
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const dsr_msgs2::msg::ModbusState &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  dsr_msgs2::msg::ModbusState &);
size_t get_serialized_size(
  const dsr_msgs2::msg::ModbusState &,
  size_t current_alignment);
size_t
max_serialized_size_ModbusState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace dsr_msgs2

// functions for std_msgs::msg::Float64MultiArray already declared above

// functions for std_msgs::msg::Float64MultiArray already declared above

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
  const dsr_msgs2::msg::RobotState & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: robot_state
  cdr << ros_message.robot_state;
  // Member: robot_state_str
  cdr << ros_message.robot_state_str;
  // Member: actual_mode
  cdr << ros_message.actual_mode;
  // Member: actual_space
  cdr << ros_message.actual_space;
  // Member: current_posj
  {
    cdr << ros_message.current_posj;
  }
  // Member: current_velj
  {
    cdr << ros_message.current_velj;
  }
  // Member: joint_abs
  {
    cdr << ros_message.joint_abs;
  }
  // Member: joint_err
  {
    cdr << ros_message.joint_err;
  }
  // Member: target_posj
  {
    cdr << ros_message.target_posj;
  }
  // Member: target_velj
  {
    cdr << ros_message.target_velj;
  }
  // Member: current_posx
  {
    cdr << ros_message.current_posx;
  }
  // Member: current_tool_posx
  {
    cdr << ros_message.current_tool_posx;
  }
  // Member: current_velx
  {
    cdr << ros_message.current_velx;
  }
  // Member: task_err
  {
    cdr << ros_message.task_err;
  }
  // Member: target_velx
  {
    cdr << ros_message.target_velx;
  }
  // Member: target_posx
  {
    cdr << ros_message.target_posx;
  }
  // Member: dynamic_tor
  {
    cdr << ros_message.dynamic_tor;
  }
  // Member: actual_jts
  {
    cdr << ros_message.actual_jts;
  }
  // Member: actual_ejt
  {
    cdr << ros_message.actual_ejt;
  }
  // Member: actual_ett
  {
    cdr << ros_message.actual_ett;
  }
  // Member: actual_bk
  {
    cdr << ros_message.actual_bk;
  }
  // Member: actual_mc
  {
    cdr << ros_message.actual_mc;
  }
  // Member: actual_mt
  {
    cdr << ros_message.actual_mt;
  }
  // Member: solution_space
  cdr << ros_message.solution_space;
  // Member: sync_time
  cdr << ros_message.sync_time;
  // Member: actual_bt
  {
    cdr << ros_message.actual_bt;
  }
  // Member: rotation_matrix
  {
    size_t size = ros_message.rotation_matrix.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.rotation_matrix[i],
        cdr);
    }
  }
  // Member: ctrlbox_digital_input
  {
    cdr << ros_message.ctrlbox_digital_input;
  }
  // Member: ctrlbox_digital_output
  {
    cdr << ros_message.ctrlbox_digital_output;
  }
  // Member: flange_digital_input
  {
    cdr << ros_message.flange_digital_input;
  }
  // Member: flange_digital_output
  {
    cdr << ros_message.flange_digital_output;
  }
  // Member: modbus_state
  {
    size_t size = ros_message.modbus_state.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      dsr_msgs2::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.modbus_state[i],
        cdr);
    }
  }
  // Member: access_control
  cdr << ros_message.access_control;
  // Member: homming_completed
  cdr << (ros_message.homming_completed ? true : false);
  // Member: tp_initialized
  cdr << (ros_message.tp_initialized ? true : false);
  // Member: mastering_need
  cdr << ros_message.mastering_need;
  // Member: drl_stopped
  cdr << (ros_message.drl_stopped ? true : false);
  // Member: disconnected
  cdr << (ros_message.disconnected ? true : false);
  // Member: f_actual_w2b
  {
    cdr << ros_message.f_actual_w2b;
  }
  // Member: f_current_pos_world
  {
    size_t size = ros_message.f_current_pos_world.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.f_current_pos_world[i],
        cdr);
    }
  }
  // Member: f_current_vel_world
  {
    cdr << ros_message.f_current_vel_world;
  }
  // Member: f_world_ext_target_torque
  {
    cdr << ros_message.f_world_ext_target_torque;
  }
  // Member: f_target_pos_world
  {
    cdr << ros_message.f_target_pos_world;
  }
  // Member: f_target_vel_world
  {
    cdr << ros_message.f_target_vel_world;
  }
  // Member: f_rotation_matrix_world
  {
    size_t size = ros_message.f_rotation_matrix_world.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.f_rotation_matrix_world[i],
        cdr);
    }
  }
  // Member: i_actual_user_coord_num
  cdr << ros_message.i_actual_user_coord_num;
  // Member: i_coord_ref
  cdr << ros_message.i_coord_ref;
  // Member: f_current_pos_user
  {
    size_t size = ros_message.f_current_pos_user.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.f_current_pos_user[i],
        cdr);
    }
  }
  // Member: f_current_vel_user
  {
    cdr << ros_message.f_current_vel_user;
  }
  // Member: f_user_ext_task_torque
  {
    cdr << ros_message.f_user_ext_task_torque;
  }
  // Member: f_target_pos_user
  {
    cdr << ros_message.f_target_pos_user;
  }
  // Member: f_target_vel_user
  {
    cdr << ros_message.f_target_vel_user;
  }
  // Member: f_rotation_matrix_user
  {
    size_t size = ros_message.f_rotation_matrix_user.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.f_rotation_matrix_user[i],
        cdr);
    }
  }
  // Member: f_actual_analog_input
  {
    cdr << ros_message.f_actual_analog_input;
  }
  // Member: b_actual_switch_input
  {
    cdr << ros_message.b_actual_switch_input;
  }
  // Member: b_actual_safety_input
  {
    cdr << ros_message.b_actual_safety_input;
  }
  // Member: i_actual_analog_input_type
  {
    cdr << ros_message.i_actual_analog_input_type;
  }
  // Member: f_target_analog_output
  {
    cdr << ros_message.f_target_analog_output;
  }
  // Member: i_target_analog_output_type
  {
    cdr << ros_message.i_target_analog_output_type;
  }
  // Member: b_actual_encorder_strove_signal
  {
    cdr << ros_message.b_actual_encorder_strove_signal;
  }
  // Member: i_actual_encorder_raw_data
  {
    cdr << ros_message.i_actual_encorder_raw_data;
  }
  // Member: b_actual_encorder_reset_signal
  {
    cdr << ros_message.b_actual_encorder_reset_signal;
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  dsr_msgs2::msg::RobotState & ros_message)
{
  // Member: robot_state
  cdr >> ros_message.robot_state;

  // Member: robot_state_str
  cdr >> ros_message.robot_state_str;

  // Member: actual_mode
  cdr >> ros_message.actual_mode;

  // Member: actual_space
  cdr >> ros_message.actual_space;

  // Member: current_posj
  {
    cdr >> ros_message.current_posj;
  }

  // Member: current_velj
  {
    cdr >> ros_message.current_velj;
  }

  // Member: joint_abs
  {
    cdr >> ros_message.joint_abs;
  }

  // Member: joint_err
  {
    cdr >> ros_message.joint_err;
  }

  // Member: target_posj
  {
    cdr >> ros_message.target_posj;
  }

  // Member: target_velj
  {
    cdr >> ros_message.target_velj;
  }

  // Member: current_posx
  {
    cdr >> ros_message.current_posx;
  }

  // Member: current_tool_posx
  {
    cdr >> ros_message.current_tool_posx;
  }

  // Member: current_velx
  {
    cdr >> ros_message.current_velx;
  }

  // Member: task_err
  {
    cdr >> ros_message.task_err;
  }

  // Member: target_velx
  {
    cdr >> ros_message.target_velx;
  }

  // Member: target_posx
  {
    cdr >> ros_message.target_posx;
  }

  // Member: dynamic_tor
  {
    cdr >> ros_message.dynamic_tor;
  }

  // Member: actual_jts
  {
    cdr >> ros_message.actual_jts;
  }

  // Member: actual_ejt
  {
    cdr >> ros_message.actual_ejt;
  }

  // Member: actual_ett
  {
    cdr >> ros_message.actual_ett;
  }

  // Member: actual_bk
  {
    cdr >> ros_message.actual_bk;
  }

  // Member: actual_mc
  {
    cdr >> ros_message.actual_mc;
  }

  // Member: actual_mt
  {
    cdr >> ros_message.actual_mt;
  }

  // Member: solution_space
  cdr >> ros_message.solution_space;

  // Member: sync_time
  cdr >> ros_message.sync_time;

  // Member: actual_bt
  {
    cdr >> ros_message.actual_bt;
  }

  // Member: rotation_matrix
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

    ros_message.rotation_matrix.resize(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.rotation_matrix[i]);
    }
  }

  // Member: ctrlbox_digital_input
  {
    cdr >> ros_message.ctrlbox_digital_input;
  }

  // Member: ctrlbox_digital_output
  {
    cdr >> ros_message.ctrlbox_digital_output;
  }

  // Member: flange_digital_input
  {
    cdr >> ros_message.flange_digital_input;
  }

  // Member: flange_digital_output
  {
    cdr >> ros_message.flange_digital_output;
  }

  // Member: modbus_state
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

    ros_message.modbus_state.resize(size);
    for (size_t i = 0; i < size; i++) {
      dsr_msgs2::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.modbus_state[i]);
    }
  }

  // Member: access_control
  cdr >> ros_message.access_control;

  // Member: homming_completed
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.homming_completed = tmp ? true : false;
  }

  // Member: tp_initialized
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.tp_initialized = tmp ? true : false;
  }

  // Member: mastering_need
  cdr >> ros_message.mastering_need;

  // Member: drl_stopped
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.drl_stopped = tmp ? true : false;
  }

  // Member: disconnected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.disconnected = tmp ? true : false;
  }

  // Member: f_actual_w2b
  {
    cdr >> ros_message.f_actual_w2b;
  }

  // Member: f_current_pos_world
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

    ros_message.f_current_pos_world.resize(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.f_current_pos_world[i]);
    }
  }

  // Member: f_current_vel_world
  {
    cdr >> ros_message.f_current_vel_world;
  }

  // Member: f_world_ext_target_torque
  {
    cdr >> ros_message.f_world_ext_target_torque;
  }

  // Member: f_target_pos_world
  {
    cdr >> ros_message.f_target_pos_world;
  }

  // Member: f_target_vel_world
  {
    cdr >> ros_message.f_target_vel_world;
  }

  // Member: f_rotation_matrix_world
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

    ros_message.f_rotation_matrix_world.resize(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.f_rotation_matrix_world[i]);
    }
  }

  // Member: i_actual_user_coord_num
  cdr >> ros_message.i_actual_user_coord_num;

  // Member: i_coord_ref
  cdr >> ros_message.i_coord_ref;

  // Member: f_current_pos_user
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

    ros_message.f_current_pos_user.resize(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.f_current_pos_user[i]);
    }
  }

  // Member: f_current_vel_user
  {
    cdr >> ros_message.f_current_vel_user;
  }

  // Member: f_user_ext_task_torque
  {
    cdr >> ros_message.f_user_ext_task_torque;
  }

  // Member: f_target_pos_user
  {
    cdr >> ros_message.f_target_pos_user;
  }

  // Member: f_target_vel_user
  {
    cdr >> ros_message.f_target_vel_user;
  }

  // Member: f_rotation_matrix_user
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

    ros_message.f_rotation_matrix_user.resize(size);
    for (size_t i = 0; i < size; i++) {
      std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.f_rotation_matrix_user[i]);
    }
  }

  // Member: f_actual_analog_input
  {
    cdr >> ros_message.f_actual_analog_input;
  }

  // Member: b_actual_switch_input
  {
    cdr >> ros_message.b_actual_switch_input;
  }

  // Member: b_actual_safety_input
  {
    cdr >> ros_message.b_actual_safety_input;
  }

  // Member: i_actual_analog_input_type
  {
    cdr >> ros_message.i_actual_analog_input_type;
  }

  // Member: f_target_analog_output
  {
    cdr >> ros_message.f_target_analog_output;
  }

  // Member: i_target_analog_output_type
  {
    cdr >> ros_message.i_target_analog_output_type;
  }

  // Member: b_actual_encorder_strove_signal
  {
    cdr >> ros_message.b_actual_encorder_strove_signal;
  }

  // Member: i_actual_encorder_raw_data
  {
    cdr >> ros_message.i_actual_encorder_raw_data;
  }

  // Member: b_actual_encorder_reset_signal
  {
    cdr >> ros_message.b_actual_encorder_reset_signal;
  }

  return true;
}  // NOLINT(readability/fn_size)

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
get_serialized_size(
  const dsr_msgs2::msg::RobotState & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: robot_state
  {
    size_t item_size = sizeof(ros_message.robot_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: robot_state_str
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.robot_state_str.size() + 1);
  // Member: actual_mode
  {
    size_t item_size = sizeof(ros_message.actual_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_space
  {
    size_t item_size = sizeof(ros_message.actual_space);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: current_posj
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.current_posj[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: current_velj
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.current_velj[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: joint_abs
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.joint_abs[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: joint_err
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.joint_err[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_posj
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_posj[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_velj
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_velj[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: current_posx
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.current_posx[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: current_tool_posx
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.current_tool_posx[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: current_velx
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.current_velx[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: task_err
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.task_err[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_velx
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_velx[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: target_posx
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.target_posx[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: dynamic_tor
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.dynamic_tor[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_jts
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_jts[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_ejt
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_ejt[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_ett
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_ett[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_bk
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_bk[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_mc
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_mc[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_mt
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.actual_mt[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: solution_space
  {
    size_t item_size = sizeof(ros_message.solution_space);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: sync_time
  {
    size_t item_size = sizeof(ros_message.sync_time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: actual_bt
  {
    size_t array_size = 5;
    size_t item_size = sizeof(ros_message.actual_bt[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: rotation_matrix
  {
    size_t array_size = ros_message.rotation_matrix.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.rotation_matrix[index], current_alignment);
    }
  }
  // Member: ctrlbox_digital_input
  {
    size_t array_size = 16;
    size_t item_size = sizeof(ros_message.ctrlbox_digital_input[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ctrlbox_digital_output
  {
    size_t array_size = 16;
    size_t item_size = sizeof(ros_message.ctrlbox_digital_output[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: flange_digital_input
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.flange_digital_input[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: flange_digital_output
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.flange_digital_output[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: modbus_state
  {
    size_t array_size = ros_message.modbus_state.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        dsr_msgs2::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.modbus_state[index], current_alignment);
    }
  }
  // Member: access_control
  {
    size_t item_size = sizeof(ros_message.access_control);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: homming_completed
  {
    size_t item_size = sizeof(ros_message.homming_completed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: tp_initialized
  {
    size_t item_size = sizeof(ros_message.tp_initialized);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: mastering_need
  {
    size_t item_size = sizeof(ros_message.mastering_need);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: drl_stopped
  {
    size_t item_size = sizeof(ros_message.drl_stopped);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: disconnected
  {
    size_t item_size = sizeof(ros_message.disconnected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_actual_w2b
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_actual_w2b[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_current_pos_world
  {
    size_t array_size = ros_message.f_current_pos_world.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.f_current_pos_world[index], current_alignment);
    }
  }
  // Member: f_current_vel_world
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_current_vel_world[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_world_ext_target_torque
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_world_ext_target_torque[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_target_pos_world
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_target_pos_world[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_target_vel_world
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_target_vel_world[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_rotation_matrix_world
  {
    size_t array_size = ros_message.f_rotation_matrix_world.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.f_rotation_matrix_world[index], current_alignment);
    }
  }
  // Member: i_actual_user_coord_num
  {
    size_t item_size = sizeof(ros_message.i_actual_user_coord_num);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: i_coord_ref
  {
    size_t item_size = sizeof(ros_message.i_coord_ref);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_current_pos_user
  {
    size_t array_size = ros_message.f_current_pos_user.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.f_current_pos_user[index], current_alignment);
    }
  }
  // Member: f_current_vel_user
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_current_vel_user[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_user_ext_task_torque
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_user_ext_task_torque[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_target_pos_user
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_target_pos_user[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_target_vel_user
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_target_vel_user[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_rotation_matrix_user
  {
    size_t array_size = ros_message.f_rotation_matrix_user.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.f_rotation_matrix_user[index], current_alignment);
    }
  }
  // Member: f_actual_analog_input
  {
    size_t array_size = 6;
    size_t item_size = sizeof(ros_message.f_actual_analog_input[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: b_actual_switch_input
  {
    size_t array_size = 3;
    size_t item_size = sizeof(ros_message.b_actual_switch_input[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: b_actual_safety_input
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.b_actual_safety_input[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: i_actual_analog_input_type
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.i_actual_analog_input_type[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: f_target_analog_output
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.f_target_analog_output[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: i_target_analog_output_type
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.i_target_analog_output_type[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: b_actual_encorder_strove_signal
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.b_actual_encorder_strove_signal[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: i_actual_encorder_raw_data
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.i_actual_encorder_raw_data[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: b_actual_encorder_reset_signal
  {
    size_t array_size = 2;
    size_t item_size = sizeof(ros_message.b_actual_encorder_reset_signal[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
max_serialized_size_RobotState(
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


  // Member: robot_state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: robot_state_str
  {
    size_t array_size = 1;

    full_bounded = false;
    is_plain = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  // Member: actual_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: actual_space
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: current_posj
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: current_velj
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: joint_abs
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: joint_err
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_posj
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_velj
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: current_posx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: current_tool_posx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: current_velx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: task_err
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_velx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: target_posx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: dynamic_tor
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_jts
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_ejt
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_ett
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_bk
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: actual_mc
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_mt
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: solution_space
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: sync_time
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: actual_bt
  {
    size_t array_size = 5;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: rotation_matrix
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

  // Member: ctrlbox_digital_input
  {
    size_t array_size = 16;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: ctrlbox_digital_output
  {
    size_t array_size = 16;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: flange_digital_input
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: flange_digital_output
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: modbus_state
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
        dsr_msgs2::msg::typesupport_fastrtps_cpp::max_serialized_size_ModbusState(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }

  // Member: access_control
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: homming_completed
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: tp_initialized
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: mastering_need
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: drl_stopped
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: disconnected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: f_actual_w2b
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_current_pos_world
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

  // Member: f_current_vel_world
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_world_ext_target_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_target_pos_world
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_target_vel_world
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_rotation_matrix_world
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

  // Member: i_actual_user_coord_num
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: i_coord_ref
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: f_current_pos_user
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

  // Member: f_current_vel_user
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_user_ext_task_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_target_pos_user
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_target_vel_user
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: f_rotation_matrix_user
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

  // Member: f_actual_analog_input
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: b_actual_switch_input
  {
    size_t array_size = 3;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: b_actual_safety_input
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: i_actual_analog_input_type
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: f_target_analog_output
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: i_target_analog_output_type
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: b_actual_encorder_strove_signal
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: i_actual_encorder_raw_data
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: b_actual_encorder_reset_signal
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = dsr_msgs2::msg::RobotState;
    is_plain =
      (
      offsetof(DataType, b_actual_encorder_reset_signal) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _RobotState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const dsr_msgs2::msg::RobotState *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _RobotState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<dsr_msgs2::msg::RobotState *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _RobotState__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const dsr_msgs2::msg::RobotState *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _RobotState__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_RobotState(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _RobotState__callbacks = {
  "dsr_msgs2::msg",
  "RobotState",
  _RobotState__cdr_serialize,
  _RobotState__cdr_deserialize,
  _RobotState__get_serialized_size,
  _RobotState__max_serialized_size
};

static rosidl_message_type_support_t _RobotState__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_RobotState__callbacks,
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
get_message_type_support_handle<dsr_msgs2::msg::RobotState>()
{
  return &dsr_msgs2::msg::typesupport_fastrtps_cpp::_RobotState__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, dsr_msgs2, msg, RobotState)() {
  return &dsr_msgs2::msg::typesupport_fastrtps_cpp::_RobotState__handle;
}

#ifdef __cplusplus
}
#endif
