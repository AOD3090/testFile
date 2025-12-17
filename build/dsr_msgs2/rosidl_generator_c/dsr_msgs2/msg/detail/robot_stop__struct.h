// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/RobotStop.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/RobotStop in the package dsr_msgs2.
/**
  * ____________________________________________________________________________________________
  *  [ robot stop mode ] 
  *  0 : STOP_TYPE_QUICK_STO
  *  1 : STOP_TYPE_QUICK
  *  2 : STOP_TYPE_SLOW
  *  3 : STOP_TYPE_HOLD = STOP_TYPE_EMERGENCY
  *
  * ____________________________________________________________________________________________
 */
typedef struct dsr_msgs2__msg__RobotStop
{
  int32_t stop_mode;
} dsr_msgs2__msg__RobotStop;

// Struct for a sequence of dsr_msgs2__msg__RobotStop.
typedef struct dsr_msgs2__msg__RobotStop__Sequence
{
  dsr_msgs2__msg__RobotStop * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__RobotStop__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__STRUCT_H_
