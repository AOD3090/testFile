// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/ServolStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SERVOL_STREAM__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__SERVOL_STREAM__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/ServolStream in the package dsr_msgs2.
/**
  * ____________________________________________________________________________________________
  *  servol
  *
  * ____________________________________________________________________________________________
 */
typedef struct dsr_msgs2__msg__ServolStream
{
  /// position
  double pos[6];
  /// velocity
  double vel[2];
  /// acceleration
  double acc[2];
  /// time
  double time;
} dsr_msgs2__msg__ServolStream;

// Struct for a sequence of dsr_msgs2__msg__ServolStream.
typedef struct dsr_msgs2__msg__ServolStream__Sequence
{
  dsr_msgs2__msg__ServolStream * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__ServolStream__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__SERVOL_STREAM__STRUCT_H_
