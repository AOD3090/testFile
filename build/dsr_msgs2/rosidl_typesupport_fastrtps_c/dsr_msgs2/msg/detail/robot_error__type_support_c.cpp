// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from dsr_msgs2:msg/RobotError.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/robot_error__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "dsr_msgs2/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "dsr_msgs2/msg/detail/robot_error__struct.h"
#include "dsr_msgs2/msg/detail/robot_error__functions.h"
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

#include "rosidl_runtime_c/string.h"  // msg1, msg2, msg3
#include "rosidl_runtime_c/string_functions.h"  // msg1, msg2, msg3

// forward declare type support functions


using _RobotError__ros_msg_type = dsr_msgs2__msg__RobotError;

static bool _RobotError__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _RobotError__ros_msg_type * ros_message = static_cast<const _RobotError__ros_msg_type *>(untyped_ros_message);
  // Field name: level
  {
    cdr << ros_message->level;
  }

  // Field name: group
  {
    cdr << ros_message->group;
  }

  // Field name: code
  {
    cdr << ros_message->code;
  }

  // Field name: msg1
  {
    const rosidl_runtime_c__String * str = &ros_message->msg1;
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

  // Field name: msg2
  {
    const rosidl_runtime_c__String * str = &ros_message->msg2;
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

  // Field name: msg3
  {
    const rosidl_runtime_c__String * str = &ros_message->msg3;
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

  return true;
}

static bool _RobotError__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _RobotError__ros_msg_type * ros_message = static_cast<_RobotError__ros_msg_type *>(untyped_ros_message);
  // Field name: level
  {
    cdr >> ros_message->level;
  }

  // Field name: group
  {
    cdr >> ros_message->group;
  }

  // Field name: code
  {
    cdr >> ros_message->code;
  }

  // Field name: msg1
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->msg1.data) {
      rosidl_runtime_c__String__init(&ros_message->msg1);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->msg1,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'msg1'\n");
      return false;
    }
  }

  // Field name: msg2
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->msg2.data) {
      rosidl_runtime_c__String__init(&ros_message->msg2);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->msg2,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'msg2'\n");
      return false;
    }
  }

  // Field name: msg3
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->msg3.data) {
      rosidl_runtime_c__String__init(&ros_message->msg3);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->msg3,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'msg3'\n");
      return false;
    }
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t get_serialized_size_dsr_msgs2__msg__RobotError(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _RobotError__ros_msg_type * ros_message = static_cast<const _RobotError__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name level
  {
    size_t item_size = sizeof(ros_message->level);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name group
  {
    size_t item_size = sizeof(ros_message->group);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name code
  {
    size_t item_size = sizeof(ros_message->code);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name msg1
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->msg1.size + 1);
  // field.name msg2
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->msg2.size + 1);
  // field.name msg3
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->msg3.size + 1);

  return current_alignment - initial_alignment;
}

static uint32_t _RobotError__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_dsr_msgs2__msg__RobotError(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t max_serialized_size_dsr_msgs2__msg__RobotError(
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

  // member: level
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: group
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: code
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: msg1
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
  // member: msg2
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
  // member: msg3
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

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = dsr_msgs2__msg__RobotError;
    is_plain =
      (
      offsetof(DataType, msg3) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _RobotError__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_dsr_msgs2__msg__RobotError(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_RobotError = {
  "dsr_msgs2::msg",
  "RobotError",
  _RobotError__cdr_serialize,
  _RobotError__cdr_deserialize,
  _RobotError__get_serialized_size,
  _RobotError__max_serialized_size
};

static rosidl_message_type_support_t _RobotError__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_RobotError,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, msg, RobotError)() {
  return &_RobotError__type_support;
}

#if defined(__cplusplus)
}
#endif
