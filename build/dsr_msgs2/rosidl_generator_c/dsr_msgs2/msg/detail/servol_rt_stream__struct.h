// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/ServolRtStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SERVOL_RT_STREAM__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__SERVOL_RT_STREAM__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/ServolRtStream in the package dsr_msgs2.
/**
  * ____________________________________________________________________________________________
  *  servol_rt
  *
  * ____________________________________________________________________________________________
 */
typedef struct dsr_msgs2__msg__ServolRtStream
{
  /// position
  double pos[6];
  /// velocity
  double vel[6];
  /// acceleration
  double acc[6];
  /// time
  double time;
} dsr_msgs2__msg__ServolRtStream;

// Struct for a sequence of dsr_msgs2__msg__ServolRtStream.
typedef struct dsr_msgs2__msg__ServolRtStream__Sequence
{
  dsr_msgs2__msg__ServolRtStream * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__ServolRtStream__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__SERVOL_RT_STREAM__STRUCT_H_
