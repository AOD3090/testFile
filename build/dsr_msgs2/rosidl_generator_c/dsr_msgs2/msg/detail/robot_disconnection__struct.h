// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/RobotDisconnection.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_DISCONNECTION__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_DISCONNECTION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/RobotDisconnection in the package dsr_msgs2.
/**
  * Event driven when the robot connection losts.
 */
typedef struct dsr_msgs2__msg__RobotDisconnection
{
  uint8_t structure_needs_at_least_one_member;
} dsr_msgs2__msg__RobotDisconnection;

// Struct for a sequence of dsr_msgs2__msg__RobotDisconnection.
typedef struct dsr_msgs2__msg__RobotDisconnection__Sequence
{
  dsr_msgs2__msg__RobotDisconnection * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__RobotDisconnection__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_DISCONNECTION__STRUCT_H_
