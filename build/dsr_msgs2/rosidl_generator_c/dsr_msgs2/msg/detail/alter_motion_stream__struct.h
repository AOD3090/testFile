// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/AlterMotionStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ALTER_MOTION_STREAM__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__ALTER_MOTION_STREAM__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/AlterMotionStream in the package dsr_msgs2.
/**
  * ____________________________________________________________________________________________
  *  alter_motion  
  *
  * ____________________________________________________________________________________________
 */
typedef struct dsr_msgs2__msg__AlterMotionStream
{
  /// position
  double pos[6];
} dsr_msgs2__msg__AlterMotionStream;

// Struct for a sequence of dsr_msgs2__msg__AlterMotionStream.
typedef struct dsr_msgs2__msg__AlterMotionStream__Sequence
{
  dsr_msgs2__msg__AlterMotionStream * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__AlterMotionStream__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__ALTER_MOTION_STREAM__STRUCT_H_
