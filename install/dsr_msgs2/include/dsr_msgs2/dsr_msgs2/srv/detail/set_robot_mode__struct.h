// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/SetRobotMode.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_ROBOT_MODE__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__SET_ROBOT_MODE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SetRobotMode in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetRobotMode_Request
{
  /// <Robot_Mode>
  int8_t robot_mode;
} dsr_msgs2__srv__SetRobotMode_Request;

// Struct for a sequence of dsr_msgs2__srv__SetRobotMode_Request.
typedef struct dsr_msgs2__srv__SetRobotMode_Request__Sequence
{
  dsr_msgs2__srv__SetRobotMode_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetRobotMode_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetRobotMode in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetRobotMode_Response
{
  bool success;
} dsr_msgs2__srv__SetRobotMode_Response;

// Struct for a sequence of dsr_msgs2__srv__SetRobotMode_Response.
typedef struct dsr_msgs2__srv__SetRobotMode_Response__Sequence
{
  dsr_msgs2__srv__SetRobotMode_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetRobotMode_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__SET_ROBOT_MODE__STRUCT_H_
