// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/ReadDataRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__READ_DATA_RT__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__READ_DATA_RT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/ReadDataRt in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__ReadDataRt_Request
{
  uint8_t structure_needs_at_least_one_member;
} dsr_msgs2__srv__ReadDataRt_Request;

// Struct for a sequence of dsr_msgs2__srv__ReadDataRt_Request.
typedef struct dsr_msgs2__srv__ReadDataRt_Request__Sequence
{
  dsr_msgs2__srv__ReadDataRt_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__ReadDataRt_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'data'
#include "dsr_msgs2/msg/detail/robot_state_rt__struct.h"

/// Struct defined in srv/ReadDataRt in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__ReadDataRt_Response
{
  dsr_msgs2__msg__RobotStateRt data;
} dsr_msgs2__srv__ReadDataRt_Response;

// Struct for a sequence of dsr_msgs2__srv__ReadDataRt_Response.
typedef struct dsr_msgs2__srv__ReadDataRt_Response__Sequence
{
  dsr_msgs2__srv__ReadDataRt_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__ReadDataRt_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__READ_DATA_RT__STRUCT_H_
