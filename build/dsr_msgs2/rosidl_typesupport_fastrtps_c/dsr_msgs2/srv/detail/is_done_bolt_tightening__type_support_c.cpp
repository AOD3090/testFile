// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from dsr_msgs2:srv/IsDoneBoltTightening.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/srv/detail/is_done_bolt_tightening__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "dsr_msgs2/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "dsr_msgs2/srv/detail/is_done_bolt_tightening__struct.h"
#include "dsr_msgs2/srv/detail/is_done_bolt_tightening__functions.h"
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


// forward declare type support functions


using _IsDoneBoltTightening_Request__ros_msg_type = dsr_msgs2__srv__IsDoneBoltTightening_Request;

static bool _IsDoneBoltTightening_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _IsDoneBoltTightening_Request__ros_msg_type * ros_message = static_cast<const _IsDoneBoltTightening_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: m
  {
    cdr << ros_message->m;
  }

  // Field name: timeout
  {
    cdr << ros_message->timeout;
  }

  // Field name: axis
  {
    cdr << ros_message->axis;
  }

  return true;
}

static bool _IsDoneBoltTightening_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _IsDoneBoltTightening_Request__ros_msg_type * ros_message = static_cast<_IsDoneBoltTightening_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: m
  {
    cdr >> ros_message->m;
  }

  // Field name: timeout
  {
    cdr >> ros_message->timeout;
  }

  // Field name: axis
  {
    cdr >> ros_message->axis;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t get_serialized_size_dsr_msgs2__srv__IsDoneBoltTightening_Request(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _IsDoneBoltTightening_Request__ros_msg_type * ros_message = static_cast<const _IsDoneBoltTightening_Request__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name m
  {
    size_t item_size = sizeof(ros_message->m);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name timeout
  {
    size_t item_size = sizeof(ros_message->timeout);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name axis
  {
    size_t item_size = sizeof(ros_message->axis);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _IsDoneBoltTightening_Request__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_dsr_msgs2__srv__IsDoneBoltTightening_Request(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t max_serialized_size_dsr_msgs2__srv__IsDoneBoltTightening_Request(
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

  // member: m
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: timeout
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: axis
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = dsr_msgs2__srv__IsDoneBoltTightening_Request;
    is_plain =
      (
      offsetof(DataType, axis) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _IsDoneBoltTightening_Request__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_dsr_msgs2__srv__IsDoneBoltTightening_Request(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_IsDoneBoltTightening_Request = {
  "dsr_msgs2::srv",
  "IsDoneBoltTightening_Request",
  _IsDoneBoltTightening_Request__cdr_serialize,
  _IsDoneBoltTightening_Request__cdr_deserialize,
  _IsDoneBoltTightening_Request__get_serialized_size,
  _IsDoneBoltTightening_Request__max_serialized_size
};

static rosidl_message_type_support_t _IsDoneBoltTightening_Request__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_IsDoneBoltTightening_Request,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, IsDoneBoltTightening_Request)() {
  return &_IsDoneBoltTightening_Request__type_support;
}

#if defined(__cplusplus)
}
#endif

// already included above
// #include <cassert>
// already included above
// #include <limits>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "dsr_msgs2/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
// already included above
// #include "dsr_msgs2/srv/detail/is_done_bolt_tightening__struct.h"
// already included above
// #include "dsr_msgs2/srv/detail/is_done_bolt_tightening__functions.h"
// already included above
// #include "fastcdr/Cdr.h"

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


// forward declare type support functions


using _IsDoneBoltTightening_Response__ros_msg_type = dsr_msgs2__srv__IsDoneBoltTightening_Response;

static bool _IsDoneBoltTightening_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _IsDoneBoltTightening_Response__ros_msg_type * ros_message = static_cast<const _IsDoneBoltTightening_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: success
  {
    cdr << (ros_message->success ? true : false);
  }

  return true;
}

static bool _IsDoneBoltTightening_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _IsDoneBoltTightening_Response__ros_msg_type * ros_message = static_cast<_IsDoneBoltTightening_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: success
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->success = tmp ? true : false;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t get_serialized_size_dsr_msgs2__srv__IsDoneBoltTightening_Response(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _IsDoneBoltTightening_Response__ros_msg_type * ros_message = static_cast<const _IsDoneBoltTightening_Response__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name success
  {
    size_t item_size = sizeof(ros_message->success);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _IsDoneBoltTightening_Response__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_dsr_msgs2__srv__IsDoneBoltTightening_Response(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t max_serialized_size_dsr_msgs2__srv__IsDoneBoltTightening_Response(
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

  // member: success
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = dsr_msgs2__srv__IsDoneBoltTightening_Response;
    is_plain =
      (
      offsetof(DataType, success) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _IsDoneBoltTightening_Response__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_dsr_msgs2__srv__IsDoneBoltTightening_Response(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_IsDoneBoltTightening_Response = {
  "dsr_msgs2::srv",
  "IsDoneBoltTightening_Response",
  _IsDoneBoltTightening_Response__cdr_serialize,
  _IsDoneBoltTightening_Response__cdr_deserialize,
  _IsDoneBoltTightening_Response__get_serialized_size,
  _IsDoneBoltTightening_Response__max_serialized_size
};

static rosidl_message_type_support_t _IsDoneBoltTightening_Response__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_IsDoneBoltTightening_Response,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, IsDoneBoltTightening_Response)() {
  return &_IsDoneBoltTightening_Response__type_support;
}

#if defined(__cplusplus)
}
#endif

#include "rosidl_typesupport_fastrtps_cpp/service_type_support.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "dsr_msgs2/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "dsr_msgs2/srv/is_done_bolt_tightening.h"

#if defined(__cplusplus)
extern "C"
{
#endif

static service_type_support_callbacks_t IsDoneBoltTightening__callbacks = {
  "dsr_msgs2::srv",
  "IsDoneBoltTightening",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, IsDoneBoltTightening_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, IsDoneBoltTightening_Response)(),
};

static rosidl_service_type_support_t IsDoneBoltTightening__handle = {
  rosidl_typesupport_fastrtps_c__identifier,
  &IsDoneBoltTightening__callbacks,
  get_service_typesupport_handle_function,
};

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, IsDoneBoltTightening)() {
  return &IsDoneBoltTightening__handle;
}

#if defined(__cplusplus)
}
#endif
