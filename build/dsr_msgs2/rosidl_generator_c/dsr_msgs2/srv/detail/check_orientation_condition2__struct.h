// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/CheckOrientationCondition2.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CHECK_ORIENTATION_CONDITION2__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__CHECK_ORIENTATION_CONDITION2__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/CheckOrientationCondition2 in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__CheckOrientationCondition2_Request
{
  /// DR_AXIS_A(10), DR_AXIS_B(11), DR_AXIS_C(12)
  int8_t axis;
  /// minimum value
  double min;
  /// maximum value
  double max;
  /// = 0         # DR_BASE(0), DR_TOOL(1), DR_WORLD(2), user_coordinate(101~200)
  ///  <DR_WORLD is only available in M2.40 or later>
  int8_t ref;
  /// = 1         # DR_MV_MOD_REL(1)
  int8_t mode;
  /// task pos(pos)
  double pos[6];
} dsr_msgs2__srv__CheckOrientationCondition2_Request;

// Struct for a sequence of dsr_msgs2__srv__CheckOrientationCondition2_Request.
typedef struct dsr_msgs2__srv__CheckOrientationCondition2_Request__Sequence
{
  dsr_msgs2__srv__CheckOrientationCondition2_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__CheckOrientationCondition2_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/CheckOrientationCondition2 in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__CheckOrientationCondition2_Response
{
  /// True or False
  bool success;
} dsr_msgs2__srv__CheckOrientationCondition2_Response;

// Struct for a sequence of dsr_msgs2__srv__CheckOrientationCondition2_Response.
typedef struct dsr_msgs2__srv__CheckOrientationCondition2_Response__Sequence
{
  dsr_msgs2__srv__CheckOrientationCondition2_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__CheckOrientationCondition2_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__CHECK_ORIENTATION_CONDITION2__STRUCT_H_
