// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/SpeedlStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SPEEDL_STREAM__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__SPEEDL_STREAM__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/SpeedlStream in the package dsr_msgs2.
/**
  * ____________________________________________________________________________________________
  *  speedl
  *
  * ____________________________________________________________________________________________
 */
typedef struct dsr_msgs2__msg__SpeedlStream
{
  /// velocity
  double vel[6];
  /// acceleration
  double acc[2];
  /// time
  double time;
} dsr_msgs2__msg__SpeedlStream;

// Struct for a sequence of dsr_msgs2__msg__SpeedlStream.
typedef struct dsr_msgs2__msg__SpeedlStream__Sequence
{
  dsr_msgs2__msg__SpeedlStream * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__SpeedlStream__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__SPEEDL_STREAM__STRUCT_H_
