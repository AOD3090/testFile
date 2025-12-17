// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/MoveHome.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_HOME__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__MOVE_HOME__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/MoveHome in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveHome_Request
{
  /// DR_HOME_TARGET_MECHANIC(0) : Mechanical home, joint angle (0,0,0,0,0,0)
  /// DR_HOME_TARGET_USER(1)     : user home
  int8_t target;
} dsr_msgs2__srv__MoveHome_Request;

// Struct for a sequence of dsr_msgs2__srv__MoveHome_Request.
typedef struct dsr_msgs2__srv__MoveHome_Request__Sequence
{
  dsr_msgs2__srv__MoveHome_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveHome_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/MoveHome in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__MoveHome_Response
{
  /// 0=success, otherwise fail
  int8_t res;
  bool success;
} dsr_msgs2__srv__MoveHome_Response;

// Struct for a sequence of dsr_msgs2__srv__MoveHome_Response.
typedef struct dsr_msgs2__srv__MoveHome_Response__Sequence
{
  dsr_msgs2__srv__MoveHome_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__MoveHome_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_HOME__STRUCT_H_
