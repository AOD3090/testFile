// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/JogMultiAxis.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__JOG_MULTI_AXIS__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__JOG_MULTI_AXIS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/JogMultiAxis in the package dsr_msgs2.
/**
  * ____________________________________________________________________________________________
  *  multi jog
  *  multi jog speed = (250mm/s x 1.73) x unit vecter x speed 
  * ____________________________________________________________________________________________
 */
typedef struct dsr_msgs2__msg__JogMultiAxis
{
  /// unit vecter of Task space [Tx, Ty, Tz, Rx, Ry, Rz] : -1.0 ~ +1.0
  double jog_axis[6];
  /// 0 : MOVE_REFERENCE_BASE, 1 : MOVE_REFERENCE_TOOL, 2 : MOVE_REFERENCE_WORLD
  int8_t move_reference;
  /// jog speed
  double speed;
} dsr_msgs2__msg__JogMultiAxis;

// Struct for a sequence of dsr_msgs2__msg__JogMultiAxis.
typedef struct dsr_msgs2__msg__JogMultiAxis__Sequence
{
  dsr_msgs2__msg__JogMultiAxis * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__JogMultiAxis__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__JOG_MULTI_AXIS__STRUCT_H_
