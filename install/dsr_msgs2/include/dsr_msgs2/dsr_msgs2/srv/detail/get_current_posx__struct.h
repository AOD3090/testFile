// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/GetCurrentPosx.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_CURRENT_POSX__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__GET_CURRENT_POSX__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetCurrentPosx in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetCurrentPosx_Request
{
  /// DR_BASE(0), DR_WORLD(2), user coord(101~200)
  /// <ref is only available in M2.40 or later>
  int8_t ref;
} dsr_msgs2__srv__GetCurrentPosx_Request;

// Struct for a sequence of dsr_msgs2__srv__GetCurrentPosx_Request.
typedef struct dsr_msgs2__srv__GetCurrentPosx_Request__Sequence
{
  dsr_msgs2__srv__GetCurrentPosx_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetCurrentPosx_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'task_pos_info'
#include "std_msgs/msg/detail/float64_multi_array__struct.h"

/// Struct defined in srv/GetCurrentPosx in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetCurrentPosx_Response
{
  /// task pos = task_pos_info[0][0:5], solution sapce = task_pos_info[0][6]
  std_msgs__msg__Float64MultiArray__Sequence task_pos_info;
  bool success;
} dsr_msgs2__srv__GetCurrentPosx_Response;

// Struct for a sequence of dsr_msgs2__srv__GetCurrentPosx_Response.
typedef struct dsr_msgs2__srv__GetCurrentPosx_Response__Sequence
{
  dsr_msgs2__srv__GetCurrentPosx_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetCurrentPosx_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__GET_CURRENT_POSX__STRUCT_H_
