// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/GetDesiredVelx.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELX__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELX__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetDesiredVelx in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetDesiredVelx_Request
{
  /// DR_BASE(0), DR_WORLD(2)
  int8_t ref;
} dsr_msgs2__srv__GetDesiredVelx_Request;

// Struct for a sequence of dsr_msgs2__srv__GetDesiredVelx_Request.
typedef struct dsr_msgs2__srv__GetDesiredVelx_Request__Sequence
{
  dsr_msgs2__srv__GetDesiredVelx_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetDesiredVelx_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/GetDesiredVelx in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetDesiredVelx_Response
{
  /// Tool velocity
  double vel[6];
  bool success;
} dsr_msgs2__srv__GetDesiredVelx_Response;

// Struct for a sequence of dsr_msgs2__srv__GetDesiredVelx_Response.
typedef struct dsr_msgs2__srv__GetDesiredVelx_Response__Sequence
{
  dsr_msgs2__srv__GetDesiredVelx_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetDesiredVelx_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__GET_DESIRED_VELX__STRUCT_H_
