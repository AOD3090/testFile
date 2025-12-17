// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/EnableAlterMotion.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__ENABLE_ALTER_MOTION__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__ENABLE_ALTER_MOTION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/EnableAlterMotion in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__EnableAlterMotion_Request
{
  /// Cycle time number
  int32_t n;
  /// DR_DPOS(0) : accumulation amount, DR_DVEL(1) : increment amount
  int8_t mode;
  /// DR_BASE(0), DR_TOOL(1), DR_WORLD(2), user coord(101~200)
  /// <ref is only available in M2.40 or later>
  int8_t ref;
  /// First value : limitation of position[mm], Second value : limitation of orientation[deg]
  double limit_dpos[2];
  /// First value : limitation of position[mm], Second value : limitation of orientation[deg]
  double limit_dpos_per[2];
} dsr_msgs2__srv__EnableAlterMotion_Request;

// Struct for a sequence of dsr_msgs2__srv__EnableAlterMotion_Request.
typedef struct dsr_msgs2__srv__EnableAlterMotion_Request__Sequence
{
  dsr_msgs2__srv__EnableAlterMotion_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__EnableAlterMotion_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/EnableAlterMotion in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__EnableAlterMotion_Response
{
  bool success;
} dsr_msgs2__srv__EnableAlterMotion_Response;

// Struct for a sequence of dsr_msgs2__srv__EnableAlterMotion_Response.
typedef struct dsr_msgs2__srv__EnableAlterMotion_Response__Sequence
{
  dsr_msgs2__srv__EnableAlterMotion_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__EnableAlterMotion_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__ENABLE_ALTER_MOTION__STRUCT_H_
