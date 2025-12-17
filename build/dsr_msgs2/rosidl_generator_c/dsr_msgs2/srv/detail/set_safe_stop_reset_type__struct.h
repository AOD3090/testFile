// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/SetSafeStopResetType.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_SAFE_STOP_RESET_TYPE__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__SET_SAFE_STOP_RESET_TYPE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SetSafeStopResetType in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetSafeStopResetType_Request
{
  /// 0=SAFE_STOP_RESET_TYPE_DEFAULT = SAFE_STOP_RESET_TYPE_PROGRAM_STOP , 1= SAFE_STOP_RESET_TYPE_PROGRAM_RESUME
  int8_t reset_type;
} dsr_msgs2__srv__SetSafeStopResetType_Request;

// Struct for a sequence of dsr_msgs2__srv__SetSafeStopResetType_Request.
typedef struct dsr_msgs2__srv__SetSafeStopResetType_Request__Sequence
{
  dsr_msgs2__srv__SetSafeStopResetType_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetSafeStopResetType_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetSafeStopResetType in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetSafeStopResetType_Response
{
  bool success;
} dsr_msgs2__srv__SetSafeStopResetType_Response;

// Struct for a sequence of dsr_msgs2__srv__SetSafeStopResetType_Response.
typedef struct dsr_msgs2__srv__SetSafeStopResetType_Response__Sequence
{
  dsr_msgs2__srv__SetSafeStopResetType_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetSafeStopResetType_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__SET_SAFE_STOP_RESET_TYPE__STRUCT_H_
