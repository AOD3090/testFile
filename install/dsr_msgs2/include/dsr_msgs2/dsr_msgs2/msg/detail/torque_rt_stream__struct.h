// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/TorqueRtStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__TORQUE_RT_STREAM__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__TORQUE_RT_STREAM__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/TorqueRtStream in the package dsr_msgs2.
/**
  * ____________________________________________________________________________________________
  *  torque_rt
  *
  * ____________________________________________________________________________________________
 */
typedef struct dsr_msgs2__msg__TorqueRtStream
{
  /// motor torque
  double tor[6];
  /// time
  double time;
} dsr_msgs2__msg__TorqueRtStream;

// Struct for a sequence of dsr_msgs2__msg__TorqueRtStream.
typedef struct dsr_msgs2__msg__TorqueRtStream__Sequence
{
  dsr_msgs2__msg__TorqueRtStream * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__TorqueRtStream__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__TORQUE_RT_STREAM__STRUCT_H_
