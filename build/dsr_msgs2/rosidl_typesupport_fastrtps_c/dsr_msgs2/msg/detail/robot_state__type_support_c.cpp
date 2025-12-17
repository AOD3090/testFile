// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from dsr_msgs2:msg/RobotState.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/robot_state__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "dsr_msgs2/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "dsr_msgs2/msg/detail/robot_state__struct.h"
#include "dsr_msgs2/msg/detail/robot_state__functions.h"
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

#include "dsr_msgs2/msg/detail/modbus_state__functions.h"  // modbus_state
#include "rosidl_runtime_c/string.h"  // robot_state_str
#include "rosidl_runtime_c/string_functions.h"  // robot_state_str
#include "std_msgs/msg/detail/float64_multi_array__functions.h"  // f_current_pos_user, f_current_pos_world, f_rotation_matrix_user, f_rotation_matrix_world, rotation_matrix

// forward declare type support functions
size_t get_serialized_size_dsr_msgs2__msg__ModbusState(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_dsr_msgs2__msg__ModbusState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, msg, ModbusState)();
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


using _RobotState__ros_msg_type = dsr_msgs2__msg__RobotState;

static bool _RobotState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _RobotState__ros_msg_type * ros_message = static_cast<const _RobotState__ros_msg_type *>(untyped_ros_message);
  // Field name: robot_state
  {
    cdr << ros_message->robot_state;
  }

  // Field name: robot_state_str
  {
    const rosidl_runtime_c__String * str = &ros_message->robot_state_str;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: actual_mode
  {
    cdr << ros_message->actual_mode;
  }

  // Field name: actual_space
  {
    cdr << ros_message->actual_space;
  }

  // Field name: current_posj
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_posj;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: current_velj
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_velj;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: joint_abs
  {
    size_t size = 6;
    auto array_ptr = ros_message->joint_abs;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: joint_err
  {
    size_t size = 6;
    auto array_ptr = ros_message->joint_err;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_posj
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_posj;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_velj
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_velj;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: current_posx
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_posx;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: current_tool_posx
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_tool_posx;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: current_velx
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_velx;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: task_err
  {
    size_t size = 6;
    auto array_ptr = ros_message->task_err;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_velx
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_velx;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: target_posx
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_posx;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: dynamic_tor
  {
    size_t size = 6;
    auto array_ptr = ros_message->dynamic_tor;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_jts
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_jts;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_ejt
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_ejt;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_ett
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_ett;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_bk
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_bk;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_mc
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_mc;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: actual_mt
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_mt;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: solution_space
  {
    cdr << ros_message->solution_space;
  }

  // Field name: sync_time
  {
    cdr << ros_message->sync_time;
  }

  // Field name: actual_bt
  {
    size_t size = 5;
    auto array_ptr = ros_message->actual_bt;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: rotation_matrix
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    size_t size = ros_message->rotation_matrix.size;
    auto array_ptr = ros_message->rotation_matrix.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: ctrlbox_digital_input
  {
    size_t size = 16;
    auto array_ptr = ros_message->ctrlbox_digital_input;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: ctrlbox_digital_output
  {
    size_t size = 16;
    auto array_ptr = ros_message->ctrlbox_digital_output;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: flange_digital_input
  {
    size_t size = 6;
    auto array_ptr = ros_message->flange_digital_input;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: flange_digital_output
  {
    size_t size = 6;
    auto array_ptr = ros_message->flange_digital_output;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: modbus_state
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, dsr_msgs2, msg, ModbusState
      )()->data);
    size_t size = ros_message->modbus_state.size;
    auto array_ptr = ros_message->modbus_state.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: access_control
  {
    cdr << ros_message->access_control;
  }

  // Field name: homming_completed
  {
    cdr << (ros_message->homming_completed ? true : false);
  }

  // Field name: tp_initialized
  {
    cdr << (ros_message->tp_initialized ? true : false);
  }

  // Field name: mastering_need
  {
    cdr << ros_message->mastering_need;
  }

  // Field name: drl_stopped
  {
    cdr << (ros_message->drl_stopped ? true : false);
  }

  // Field name: disconnected
  {
    cdr << (ros_message->disconnected ? true : false);
  }

  // Field name: f_actual_w2b
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_actual_w2b;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_current_pos_world
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    size_t size = ros_message->f_current_pos_world.size;
    auto array_ptr = ros_message->f_current_pos_world.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: f_current_vel_world
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_current_vel_world;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_world_ext_target_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_world_ext_target_torque;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_target_pos_world
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_target_pos_world;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_target_vel_world
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_target_vel_world;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_rotation_matrix_world
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    size_t size = ros_message->f_rotation_matrix_world.size;
    auto array_ptr = ros_message->f_rotation_matrix_world.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: i_actual_user_coord_num
  {
    cdr << ros_message->i_actual_user_coord_num;
  }

  // Field name: i_coord_ref
  {
    cdr << ros_message->i_coord_ref;
  }

  // Field name: f_current_pos_user
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    size_t size = ros_message->f_current_pos_user.size;
    auto array_ptr = ros_message->f_current_pos_user.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: f_current_vel_user
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_current_vel_user;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_user_ext_task_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_user_ext_task_torque;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_target_pos_user
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_target_pos_user;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_target_vel_user
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_target_vel_user;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_rotation_matrix_user
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Float64MultiArray
      )()->data);
    size_t size = ros_message->f_rotation_matrix_user.size;
    auto array_ptr = ros_message->f_rotation_matrix_user.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: f_actual_analog_input
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_actual_analog_input;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: b_actual_switch_input
  {
    size_t size = 3;
    auto array_ptr = ros_message->b_actual_switch_input;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: b_actual_safety_input
  {
    size_t size = 2;
    auto array_ptr = ros_message->b_actual_safety_input;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: i_actual_analog_input_type
  {
    size_t size = 2;
    auto array_ptr = ros_message->i_actual_analog_input_type;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: f_target_analog_output
  {
    size_t size = 2;
    auto array_ptr = ros_message->f_target_analog_output;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: i_target_analog_output_type
  {
    size_t size = 2;
    auto array_ptr = ros_message->i_target_analog_output_type;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: b_actual_encorder_strove_signal
  {
    size_t size = 2;
    auto array_ptr = ros_message->b_actual_encorder_strove_signal;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: i_actual_encorder_raw_data
  {
    size_t size = 2;
    auto array_ptr = ros_message->i_actual_encorder_raw_data;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: b_actual_encorder_reset_signal
  {
    size_t size = 2;
    auto array_ptr = ros_message->b_actual_encorder_reset_signal;
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _RobotState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _RobotState__ros_msg_type * ros_message = static_cast<_RobotState__ros_msg_type *>(untyped_ros_message);
  // Field name: robot_state
  {
    cdr >> ros_message->robot_state;
  }

  // Field name: robot_state_str
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->robot_state_str.data) {
      rosidl_runtime_c__String__init(&ros_message->robot_state_str);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->robot_state_str,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'robot_state_str'\n");
      return false;
    }
  }

  // Field name: actual_mode
  {
    cdr >> ros_message->actual_mode;
  }

  // Field name: actual_space
  {
    cdr >> ros_message->actual_space;
  }

  // Field name: current_posj
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_posj;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: current_velj
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_velj;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: joint_abs
  {
    size_t size = 6;
    auto array_ptr = ros_message->joint_abs;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: joint_err
  {
    size_t size = 6;
    auto array_ptr = ros_message->joint_err;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_posj
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_posj;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_velj
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_velj;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: current_posx
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_posx;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: current_tool_posx
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_tool_posx;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: current_velx
  {
    size_t size = 6;
    auto array_ptr = ros_message->current_velx;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: task_err
  {
    size_t size = 6;
    auto array_ptr = ros_message->task_err;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_velx
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_velx;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: target_posx
  {
    size_t size = 6;
    auto array_ptr = ros_message->target_posx;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: dynamic_tor
  {
    size_t size = 6;
    auto array_ptr = ros_message->dynamic_tor;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_jts
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_jts;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_ejt
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_ejt;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_ett
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_ett;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_bk
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_bk;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_mc
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_mc;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: actual_mt
  {
    size_t size = 6;
    auto array_ptr = ros_message->actual_mt;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: solution_space
  {
    cdr >> ros_message->solution_space;
  }

  // Field name: sync_time
  {
    cdr >> ros_message->sync_time;
  }

  // Field name: actual_bt
  {
    size_t size = 5;
    auto array_ptr = ros_message->actual_bt;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: rotation_matrix
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

    if (ros_message->rotation_matrix.data) {
      std_msgs__msg__Float64MultiArray__Sequence__fini(&ros_message->rotation_matrix);
    }
    if (!std_msgs__msg__Float64MultiArray__Sequence__init(&ros_message->rotation_matrix, size)) {
      fprintf(stderr, "failed to create array for field 'rotation_matrix'");
      return false;
    }
    auto array_ptr = ros_message->rotation_matrix.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: ctrlbox_digital_input
  {
    size_t size = 16;
    auto array_ptr = ros_message->ctrlbox_digital_input;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: ctrlbox_digital_output
  {
    size_t size = 16;
    auto array_ptr = ros_message->ctrlbox_digital_output;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: flange_digital_input
  {
    size_t size = 6;
    auto array_ptr = ros_message->flange_digital_input;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: flange_digital_output
  {
    size_t size = 6;
    auto array_ptr = ros_message->flange_digital_output;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: modbus_state
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, dsr_msgs2, msg, ModbusState
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

    if (ros_message->modbus_state.data) {
      dsr_msgs2__msg__ModbusState__Sequence__fini(&ros_message->modbus_state);
    }
    if (!dsr_msgs2__msg__ModbusState__Sequence__init(&ros_message->modbus_state, size)) {
      fprintf(stderr, "failed to create array for field 'modbus_state'");
      return false;
    }
    auto array_ptr = ros_message->modbus_state.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: access_control
  {
    cdr >> ros_message->access_control;
  }

  // Field name: homming_completed
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->homming_completed = tmp ? true : false;
  }

  // Field name: tp_initialized
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->tp_initialized = tmp ? true : false;
  }

  // Field name: mastering_need
  {
    cdr >> ros_message->mastering_need;
  }

  // Field name: drl_stopped
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->drl_stopped = tmp ? true : false;
  }

  // Field name: disconnected
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->disconnected = tmp ? true : false;
  }

  // Field name: f_actual_w2b
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_actual_w2b;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_current_pos_world
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

    if (ros_message->f_current_pos_world.data) {
      std_msgs__msg__Float64MultiArray__Sequence__fini(&ros_message->f_current_pos_world);
    }
    if (!std_msgs__msg__Float64MultiArray__Sequence__init(&ros_message->f_current_pos_world, size)) {
      fprintf(stderr, "failed to create array for field 'f_current_pos_world'");
      return false;
    }
    auto array_ptr = ros_message->f_current_pos_world.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: f_current_vel_world
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_current_vel_world;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_world_ext_target_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_world_ext_target_torque;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_target_pos_world
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_target_pos_world;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_target_vel_world
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_target_vel_world;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_rotation_matrix_world
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

    if (ros_message->f_rotation_matrix_world.data) {
      std_msgs__msg__Float64MultiArray__Sequence__fini(&ros_message->f_rotation_matrix_world);
    }
    if (!std_msgs__msg__Float64MultiArray__Sequence__init(&ros_message->f_rotation_matrix_world, size)) {
      fprintf(stderr, "failed to create array for field 'f_rotation_matrix_world'");
      return false;
    }
    auto array_ptr = ros_message->f_rotation_matrix_world.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: i_actual_user_coord_num
  {
    cdr >> ros_message->i_actual_user_coord_num;
  }

  // Field name: i_coord_ref
  {
    cdr >> ros_message->i_coord_ref;
  }

  // Field name: f_current_pos_user
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

    if (ros_message->f_current_pos_user.data) {
      std_msgs__msg__Float64MultiArray__Sequence__fini(&ros_message->f_current_pos_user);
    }
    if (!std_msgs__msg__Float64MultiArray__Sequence__init(&ros_message->f_current_pos_user, size)) {
      fprintf(stderr, "failed to create array for field 'f_current_pos_user'");
      return false;
    }
    auto array_ptr = ros_message->f_current_pos_user.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: f_current_vel_user
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_current_vel_user;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_user_ext_task_torque
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_user_ext_task_torque;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_target_pos_user
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_target_pos_user;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_target_vel_user
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_target_vel_user;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_rotation_matrix_user
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

    if (ros_message->f_rotation_matrix_user.data) {
      std_msgs__msg__Float64MultiArray__Sequence__fini(&ros_message->f_rotation_matrix_user);
    }
    if (!std_msgs__msg__Float64MultiArray__Sequence__init(&ros_message->f_rotation_matrix_user, size)) {
      fprintf(stderr, "failed to create array for field 'f_rotation_matrix_user'");
      return false;
    }
    auto array_ptr = ros_message->f_rotation_matrix_user.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: f_actual_analog_input
  {
    size_t size = 6;
    auto array_ptr = ros_message->f_actual_analog_input;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: b_actual_switch_input
  {
    size_t size = 3;
    auto array_ptr = ros_message->b_actual_switch_input;
    for (size_t i = 0; i < size; ++i) {
      uint8_t tmp;
      cdr >> tmp;
      array_ptr[i] = tmp ? true : false;
    }
  }

  // Field name: b_actual_safety_input
  {
    size_t size = 2;
    auto array_ptr = ros_message->b_actual_safety_input;
    for (size_t i = 0; i < size; ++i) {
      uint8_t tmp;
      cdr >> tmp;
      array_ptr[i] = tmp ? true : false;
    }
  }

  // Field name: i_actual_analog_input_type
  {
    size_t size = 2;
    auto array_ptr = ros_message->i_actual_analog_input_type;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: f_target_analog_output
  {
    size_t size = 2;
    auto array_ptr = ros_message->f_target_analog_output;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: i_target_analog_output_type
  {
    size_t size = 2;
    auto array_ptr = ros_message->i_target_analog_output_type;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: b_actual_encorder_strove_signal
  {
    size_t size = 2;
    auto array_ptr = ros_message->b_actual_encorder_strove_signal;
    for (size_t i = 0; i < size; ++i) {
      uint8_t tmp;
      cdr >> tmp;
      array_ptr[i] = tmp ? true : false;
    }
  }

  // Field name: i_actual_encorder_raw_data
  {
    size_t size = 2;
    auto array_ptr = ros_message->i_actual_encorder_raw_data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: b_actual_encorder_reset_signal
  {
    size_t size = 2;
    auto array_ptr = ros_message->b_actual_encorder_reset_signal;
    for (size_t i = 0; i < size; ++i) {
      uint8_t tmp;
      cdr >> tmp;
      array_ptr[i] = tmp ? true : false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t get_serialized_size_dsr_msgs2__msg__RobotState(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _RobotState__ros_msg_type * ros_message = static_cast<const _RobotState__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name robot_state
  {
    size_t item_size = sizeof(ros_message->robot_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name robot_state_str
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->robot_state_str.size + 1);
  // field.name actual_mode
  {
    size_t item_size = sizeof(ros_message->actual_mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_space
  {
    size_t item_size = sizeof(ros_message->actual_space);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name current_posj
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->current_posj;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name current_velj
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->current_velj;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name joint_abs
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->joint_abs;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name joint_err
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->joint_err;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_posj
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_posj;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_velj
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_velj;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name current_posx
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->current_posx;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name current_tool_posx
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->current_tool_posx;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name current_velx
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->current_velx;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name task_err
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->task_err;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_velx
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_velx;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name target_posx
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->target_posx;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name dynamic_tor
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->dynamic_tor;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_jts
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_jts;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_ejt
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_ejt;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_ett
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_ett;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_bk
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_bk;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_mc
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_mc;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_mt
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->actual_mt;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name solution_space
  {
    size_t item_size = sizeof(ros_message->solution_space);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name sync_time
  {
    size_t item_size = sizeof(ros_message->sync_time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name actual_bt
  {
    size_t array_size = 5;
    auto array_ptr = ros_message->actual_bt;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name rotation_matrix
  {
    size_t array_size = ros_message->rotation_matrix.size;
    auto array_ptr = ros_message->rotation_matrix.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_std_msgs__msg__Float64MultiArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name ctrlbox_digital_input
  {
    size_t array_size = 16;
    auto array_ptr = ros_message->ctrlbox_digital_input;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ctrlbox_digital_output
  {
    size_t array_size = 16;
    auto array_ptr = ros_message->ctrlbox_digital_output;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name flange_digital_input
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->flange_digital_input;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name flange_digital_output
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->flange_digital_output;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name modbus_state
  {
    size_t array_size = ros_message->modbus_state.size;
    auto array_ptr = ros_message->modbus_state.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_dsr_msgs2__msg__ModbusState(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name access_control
  {
    size_t item_size = sizeof(ros_message->access_control);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name homming_completed
  {
    size_t item_size = sizeof(ros_message->homming_completed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name tp_initialized
  {
    size_t item_size = sizeof(ros_message->tp_initialized);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name mastering_need
  {
    size_t item_size = sizeof(ros_message->mastering_need);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name drl_stopped
  {
    size_t item_size = sizeof(ros_message->drl_stopped);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name disconnected
  {
    size_t item_size = sizeof(ros_message->disconnected);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_actual_w2b
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_actual_w2b;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_current_pos_world
  {
    size_t array_size = ros_message->f_current_pos_world.size;
    auto array_ptr = ros_message->f_current_pos_world.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_std_msgs__msg__Float64MultiArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name f_current_vel_world
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_current_vel_world;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_world_ext_target_torque
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_world_ext_target_torque;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_target_pos_world
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_target_pos_world;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_target_vel_world
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_target_vel_world;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_rotation_matrix_world
  {
    size_t array_size = ros_message->f_rotation_matrix_world.size;
    auto array_ptr = ros_message->f_rotation_matrix_world.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_std_msgs__msg__Float64MultiArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name i_actual_user_coord_num
  {
    size_t item_size = sizeof(ros_message->i_actual_user_coord_num);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name i_coord_ref
  {
    size_t item_size = sizeof(ros_message->i_coord_ref);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_current_pos_user
  {
    size_t array_size = ros_message->f_current_pos_user.size;
    auto array_ptr = ros_message->f_current_pos_user.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_std_msgs__msg__Float64MultiArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name f_current_vel_user
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_current_vel_user;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_user_ext_task_torque
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_user_ext_task_torque;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_target_pos_user
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_target_pos_user;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_target_vel_user
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_target_vel_user;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_rotation_matrix_user
  {
    size_t array_size = ros_message->f_rotation_matrix_user.size;
    auto array_ptr = ros_message->f_rotation_matrix_user.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_std_msgs__msg__Float64MultiArray(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name f_actual_analog_input
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->f_actual_analog_input;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name b_actual_switch_input
  {
    size_t array_size = 3;
    auto array_ptr = ros_message->b_actual_switch_input;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name b_actual_safety_input
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->b_actual_safety_input;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name i_actual_analog_input_type
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->i_actual_analog_input_type;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name f_target_analog_output
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->f_target_analog_output;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name i_target_analog_output_type
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->i_target_analog_output_type;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name b_actual_encorder_strove_signal
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->b_actual_encorder_strove_signal;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name i_actual_encorder_raw_data
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->i_actual_encorder_raw_data;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name b_actual_encorder_reset_signal
  {
    size_t array_size = 2;
    auto array_ptr = ros_message->b_actual_encorder_reset_signal;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _RobotState__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_dsr_msgs2__msg__RobotState(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t max_serialized_size_dsr_msgs2__msg__RobotState(
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

  // member: robot_state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: robot_state_str
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
  // member: actual_mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: actual_space
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: current_posj
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: current_velj
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: joint_abs
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: joint_err
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_posj
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_velj
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: current_posx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: current_tool_posx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: current_velx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: task_err
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_velx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: target_posx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: dynamic_tor
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_jts
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_ejt
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_ett
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_bk
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: actual_mc
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_mt
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: solution_space
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: sync_time
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: actual_bt
  {
    size_t array_size = 5;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: rotation_matrix
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
  // member: ctrlbox_digital_input
  {
    size_t array_size = 16;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: ctrlbox_digital_output
  {
    size_t array_size = 16;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: flange_digital_input
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: flange_digital_output
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: modbus_state
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
        max_serialized_size_dsr_msgs2__msg__ModbusState(
        inner_full_bounded, inner_is_plain, current_alignment);
      last_member_size += inner_size;
      current_alignment += inner_size;
      full_bounded &= inner_full_bounded;
      is_plain &= inner_is_plain;
    }
  }
  // member: access_control
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: homming_completed
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: tp_initialized
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: mastering_need
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: drl_stopped
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: disconnected
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: f_actual_w2b
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_current_pos_world
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
  // member: f_current_vel_world
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_world_ext_target_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_target_pos_world
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_target_vel_world
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_rotation_matrix_world
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
  // member: i_actual_user_coord_num
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: i_coord_ref
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: f_current_pos_user
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
  // member: f_current_vel_user
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_user_ext_task_torque
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_target_pos_user
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_target_vel_user
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: f_rotation_matrix_user
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
  // member: f_actual_analog_input
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: b_actual_switch_input
  {
    size_t array_size = 3;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: b_actual_safety_input
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: i_actual_analog_input_type
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: f_target_analog_output
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: i_target_analog_output_type
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: b_actual_encorder_strove_signal
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: i_actual_encorder_raw_data
  {
    size_t array_size = 2;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: b_actual_encorder_reset_signal
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
    using DataType = dsr_msgs2__msg__RobotState;
    is_plain =
      (
      offsetof(DataType, b_actual_encorder_reset_signal) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _RobotState__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_dsr_msgs2__msg__RobotState(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_RobotState = {
  "dsr_msgs2::msg",
  "RobotState",
  _RobotState__cdr_serialize,
  _RobotState__cdr_deserialize,
  _RobotState__get_serialized_size,
  _RobotState__max_serialized_size
};

static rosidl_message_type_support_t _RobotState__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_RobotState,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, msg, RobotState)() {
  return &_RobotState__type_support;
}

#if defined(__cplusplus)
}
#endif
