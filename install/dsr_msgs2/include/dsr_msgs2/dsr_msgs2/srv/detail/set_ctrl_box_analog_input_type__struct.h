// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/SetCtrlBoxAnalogInputType.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_INPUT_TYPE__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_INPUT_TYPE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/SetCtrlBoxAnalogInputType in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request
{
  /// 1 = ch1, 2= ch2
  int8_t channel;
  /// 0 = current, 1 = voltage
  int8_t mode;
} dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request;

// Struct for a sequence of dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request.
typedef struct dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request__Sequence
{
  dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/SetCtrlBoxAnalogInputType in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response
{
  bool success;
} dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response;

// Struct for a sequence of dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response.
typedef struct dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response__Sequence
{
  dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__SetCtrlBoxAnalogInputType_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_INPUT_TYPE__STRUCT_H_
