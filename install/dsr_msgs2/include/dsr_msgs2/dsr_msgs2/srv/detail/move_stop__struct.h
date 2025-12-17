// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/MoveStop.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_STOP__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__MOVE_STOP__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/MoveStop in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveStop_Request
{
  /// DR_QSTOP_STO(0) : Quick stop (Stop Category 1 without STO(Safe Torque Off)
  /// DR_QSTOP(1)     : Quick stop (Stop Category 2)
  /// DR_SSTO(2)      : Soft Stop
  /// DR_HOLD(3)      : HOLD stop
  int32_t stop_mode;
} dsr_msgs2__srv__MoveStop_Request;

// Struct for a sequence of dsr_msgs2__srv__MoveStop_Request.
typedef struct dsr_msgs2__srv__MoveStop_Request__Sequence
{
  dsr_msgs2__srv__MoveStop_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveStop_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/MoveStop in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveStop_Response
{
  bool success;
} dsr_msgs2__srv__MoveStop_Response;

// Struct for a sequence of dsr_msgs2__srv__MoveStop_Response.
typedef struct dsr_msgs2__srv__MoveStop_Response__Sequence
{
  dsr_msgs2__srv__MoveStop_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveStop_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_STOP__STRUCT_H_
