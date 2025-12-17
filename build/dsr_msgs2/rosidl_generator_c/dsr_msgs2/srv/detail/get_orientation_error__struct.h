// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/GetOrientationError.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_ORIENTATION_ERROR__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__GET_ORIENTATION_ERROR__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetOrientationError in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetOrientationError_Request
{
  /// task pos(posx)
  double xd[6];
  /// task pos(posx)
  double xc[6];
  /// DR_AXIS_X(0), DR_AXIS_Y(1), DR_AXIS_Z(2)
  int8_t axis;
} dsr_msgs2__srv__GetOrientationError_Request;

// Struct for a sequence of dsr_msgs2__srv__GetOrientationError_Request.
typedef struct dsr_msgs2__srv__GetOrientationError_Request__Sequence
{
  dsr_msgs2__srv__GetOrientationError_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetOrientationError_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/GetOrientationError in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetOrientationError_Response
{
  /// orientation error
  float ori_error;
  bool success;
} dsr_msgs2__srv__GetOrientationError_Response;

// Struct for a sequence of dsr_msgs2__srv__GetOrientationError_Response.
typedef struct dsr_msgs2__srv__GetOrientationError_Response__Sequence
{
  dsr_msgs2__srv__GetOrientationError_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetOrientationError_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__GET_ORIENTATION_ERROR__STRUCT_H_
