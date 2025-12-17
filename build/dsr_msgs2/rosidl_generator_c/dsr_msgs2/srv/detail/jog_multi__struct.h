// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/JogMulti.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__JOG_MULTI__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__JOG_MULTI__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/JogMulti in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__JogMulti_Request
{
  /// unit vecter of Task space [Tx, Ty, Tz, Rx, Ry, Rz] : -1.0 ~ +1.0
  double jog_axis[6];
  /// 0 : MOVE_REFERENCE_BASE, 1 : MOVE_REFERENCE_TOOL, 2 : MOVE_REFERENCE_WORLD
  int8_t move_reference;
  /// jog speed
  double speed;
} dsr_msgs2__srv__JogMulti_Request;

// Struct for a sequence of dsr_msgs2__srv__JogMulti_Request.
typedef struct dsr_msgs2__srv__JogMulti_Request__Sequence
{
  dsr_msgs2__srv__JogMulti_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__JogMulti_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/JogMulti in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__JogMulti_Response
{
  bool success;
} dsr_msgs2__srv__JogMulti_Response;

// Struct for a sequence of dsr_msgs2__srv__JogMulti_Response.
typedef struct dsr_msgs2__srv__JogMulti_Response__Sequence
{
  dsr_msgs2__srv__JogMulti_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__JogMulti_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__JOG_MULTI__STRUCT_H_
