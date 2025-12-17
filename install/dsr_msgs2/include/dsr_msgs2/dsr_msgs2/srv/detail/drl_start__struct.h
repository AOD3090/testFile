// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/DrlStart.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__DRL_START__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__DRL_START__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'code'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/DrlStart in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__DrlStart_Request
{
  /// Robot System Mode 0 : Real, 1 : virtual
  int8_t robot_system;
  /// drl code
  rosidl_runtime_c__String code;
} dsr_msgs2__srv__DrlStart_Request;

// Struct for a sequence of dsr_msgs2__srv__DrlStart_Request.
typedef struct dsr_msgs2__srv__DrlStart_Request__Sequence
{
  dsr_msgs2__srv__DrlStart_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__DrlStart_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/DrlStart in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__DrlStart_Response
{
  bool success;
} dsr_msgs2__srv__DrlStart_Response;

// Struct for a sequence of dsr_msgs2__srv__DrlStart_Response.
typedef struct dsr_msgs2__srv__DrlStart_Response__Sequence
{
  dsr_msgs2__srv__DrlStart_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__DrlStart_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__DRL_START__STRUCT_H_
