// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from dsr_msgs2:srv/CoordTransform.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/srv/detail/coord_transform__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "dsr_msgs2/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "dsr_msgs2/srv/detail/coord_transform__struct.h"
#include "dsr_msgs2/srv/detail/coord_transform__functions.h"
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


using _CoordTransform_Request__ros_msg_type = dsr_msgs2__srv__CoordTransform_Request;

static bool _CoordTransform_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _CoordTransform_Request__ros_msg_type * ros_message = static_cast<const _CoordTransform_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: pos_in
  {
    size_t size = 6;
    auto array_ptr = ros_message->pos_in;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: ref_in
  {
    cdr << ros_message->ref_in;
  }

  // Field name: ref_out
  {
    cdr << ros_message->ref_out;
  }

  return true;
}

static bool _CoordTransform_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _CoordTransform_Request__ros_msg_type * ros_message = static_cast<_CoordTransform_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: pos_in
  {
    size_t size = 6;
    auto array_ptr = ros_message->pos_in;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: ref_in
  {
    cdr >> ros_message->ref_in;
  }

  // Field name: ref_out
  {
    cdr >> ros_message->ref_out;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t get_serialized_size_dsr_msgs2__srv__CoordTransform_Request(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _CoordTransform_Request__ros_msg_type * ros_message = static_cast<const _CoordTransform_Request__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name pos_in
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->pos_in;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ref_in
  {
    size_t item_size = sizeof(ros_message->ref_in);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ref_out
  {
    size_t item_size = sizeof(ros_message->ref_out);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _CoordTransform_Request__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_dsr_msgs2__srv__CoordTransform_Request(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t max_serialized_size_dsr_msgs2__srv__CoordTransform_Request(
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

  // member: pos_in
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: ref_in
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: ref_out
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
    using DataType = dsr_msgs2__srv__CoordTransform_Request;
    is_plain =
      (
      offsetof(DataType, ref_out) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _CoordTransform_Request__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_dsr_msgs2__srv__CoordTransform_Request(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_CoordTransform_Request = {
  "dsr_msgs2::srv",
  "CoordTransform_Request",
  _CoordTransform_Request__cdr_serialize,
  _CoordTransform_Request__cdr_deserialize,
  _CoordTransform_Request__get_serialized_size,
  _CoordTransform_Request__max_serialized_size
};

static rosidl_message_type_support_t _CoordTransform_Request__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_CoordTransform_Request,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, CoordTransform_Request)() {
  return &_CoordTransform_Request__type_support;
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
// #include "dsr_msgs2/srv/detail/coord_transform__struct.h"
// already included above
// #include "dsr_msgs2/srv/detail/coord_transform__functions.h"
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


using _CoordTransform_Response__ros_msg_type = dsr_msgs2__srv__CoordTransform_Response;

static bool _CoordTransform_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _CoordTransform_Response__ros_msg_type * ros_message = static_cast<const _CoordTransform_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: conv_posx
  {
    size_t size = 6;
    auto array_ptr = ros_message->conv_posx;
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: success
  {
    cdr << (ros_message->success ? true : false);
  }

  return true;
}

static bool _CoordTransform_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _CoordTransform_Response__ros_msg_type * ros_message = static_cast<_CoordTransform_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: conv_posx
  {
    size_t size = 6;
    auto array_ptr = ros_message->conv_posx;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: success
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->success = tmp ? true : false;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t get_serialized_size_dsr_msgs2__srv__CoordTransform_Response(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _CoordTransform_Response__ros_msg_type * ros_message = static_cast<const _CoordTransform_Response__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name conv_posx
  {
    size_t array_size = 6;
    auto array_ptr = ros_message->conv_posx;
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name success
  {
    size_t item_size = sizeof(ros_message->success);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _CoordTransform_Response__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_dsr_msgs2__srv__CoordTransform_Response(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_dsr_msgs2
size_t max_serialized_size_dsr_msgs2__srv__CoordTransform_Response(
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

  // member: conv_posx
  {
    size_t array_size = 6;

    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
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
    using DataType = dsr_msgs2__srv__CoordTransform_Response;
    is_plain =
      (
      offsetof(DataType, success) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _CoordTransform_Response__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_dsr_msgs2__srv__CoordTransform_Response(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_CoordTransform_Response = {
  "dsr_msgs2::srv",
  "CoordTransform_Response",
  _CoordTransform_Response__cdr_serialize,
  _CoordTransform_Response__cdr_deserialize,
  _CoordTransform_Response__get_serialized_size,
  _CoordTransform_Response__max_serialized_size
};

static rosidl_message_type_support_t _CoordTransform_Response__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_CoordTransform_Response,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, CoordTransform_Response)() {
  return &_CoordTransform_Response__type_support;
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
#include "dsr_msgs2/srv/coord_transform.h"

#if defined(__cplusplus)
extern "C"
{
#endif

static service_type_support_callbacks_t CoordTransform__callbacks = {
  "dsr_msgs2::srv",
  "CoordTransform",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, CoordTransform_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, CoordTransform_Response)(),
};

static rosidl_service_type_support_t CoordTransform__handle = {
  rosidl_typesupport_fastrtps_c__identifier,
  &CoordTransform__callbacks,
  get_service_typesupport_handle_function,
};

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, dsr_msgs2, srv, CoordTransform)() {
  return &CoordTransform__handle;
}

#if defined(__cplusplus)
}
#endif
