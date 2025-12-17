// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:srv/GetToolDigitalInput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_TOOL_DIGITAL_INPUT__STRUCT_H_
#define DSR_MSGS2__SRV__DETAIL__GET_TOOL_DIGITAL_INPUT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/GetToolDigitalInput in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetToolDigitalInput_Request
{
  /// Digital Input in Flange(1 ~ 6)
  /// <GPIO_TOOL_DIGITAL_INDEX>
  int8_t index;
} dsr_msgs2__srv__GetToolDigitalInput_Request;

// Struct for a sequence of dsr_msgs2__srv__GetToolDigitalInput_Request.
typedef struct dsr_msgs2__srv__GetToolDigitalInput_Request__Sequence
{
  dsr_msgs2__srv__GetToolDigitalInput_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetToolDigitalInput_Request__Sequence;


// Constants defined in the message

/// Struct defined in srv/GetToolDigitalInput in the package dsr_msgs2.
typedef struct dsr_msgs2__srv__GetToolDigitalInput_Response
{
  /// 0=OFF, 1=ON
  int8_t value;
  bool success;
} dsr_msgs2__srv__GetToolDigitalInput_Response;

// Struct for a sequence of dsr_msgs2__srv__GetToolDigitalInput_Response.
typedef struct dsr_msgs2__srv__GetToolDigitalInput_Response__Sequence
{
  dsr_msgs2__srv__GetToolDigitalInput_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__srv__GetToolDigitalInput_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__GET_TOOL_DIGITAL_INPUT__STRUCT_H_
