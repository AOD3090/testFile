// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from dsr_msgs2:msg/LogAlarm.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/log_alarm__rosidl_typesupport_fastrtps_cpp.hpp"
#include "dsr_msgs2/msg/detail/log_alarm__struct.hpp"

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

namespace dsr_msgs2
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
cdr_serialize(
  const dsr_msgs2::msg::LogAlarm & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: level
  cdr << ros_message.level;
  // Member: group
  cdr << ros_message.group;
  // Member: index
  cdr << ros_message.index;
  // Member: param
  {
    cdr << ros_message.param;
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  dsr_msgs2::msg::LogAlarm & ros_message)
{
  // Member: level
  cdr >> ros_message.level;

  // Member: group
  cdr >> ros_message.group;

  // Member: index
  cdr >> ros_message.index;

  // Member: param
  {
    cdr >> ros_message.param;
  }

  return true;
}  // NOLINT(readability/fn_size)

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
get_serialized_size(
  const dsr_msgs2::msg::LogAlarm & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: level
  {
    size_t item_size = sizeof(ros_message.level);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: group
  {
    size_t item_size = sizeof(ros_message.group);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: index
  {
    size_t item_size = sizeof(ros_message.index);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: param
  {
    size_t array_size = 3;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        (ros_message.param[index].size() + 1);
    }
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dsr_msgs2
max_serialized_size_LogAlarm(
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


  // Member: level
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: group
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: index
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: param
  {
    size_t array_size = 3;

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
    using DataType = dsr_msgs2::msg::LogAlarm;
    is_plain =
      (
      offsetof(DataType, param) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _LogAlarm__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const dsr_msgs2::msg::LogAlarm *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _LogAlarm__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<dsr_msgs2::msg::LogAlarm *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _LogAlarm__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const dsr_msgs2::msg::LogAlarm *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _LogAlarm__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_LogAlarm(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _LogAlarm__callbacks = {
  "dsr_msgs2::msg",
  "LogAlarm",
  _LogAlarm__cdr_serialize,
  _LogAlarm__cdr_deserialize,
  _LogAlarm__get_serialized_size,
  _LogAlarm__max_serialized_size
};

static rosidl_message_type_support_t _LogAlarm__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_LogAlarm__callbacks,
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
get_message_type_support_handle<dsr_msgs2::msg::LogAlarm>()
{
  return &dsr_msgs2::msg::typesupport_fastrtps_cpp::_LogAlarm__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, dsr_msgs2, msg, LogAlarm)() {
  return &dsr_msgs2::msg::typesupport_fastrtps_cpp::_LogAlarm__handle;
}

#ifdef __cplusplus
}
#endif
