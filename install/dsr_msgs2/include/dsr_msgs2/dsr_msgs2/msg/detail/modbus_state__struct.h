// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dsr_msgs2:msg/ModbusState.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__MODBUS_STATE__STRUCT_H_
#define DSR_MSGS2__MSG__DETAIL__MODBUS_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'modbus_symbol'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/ModbusState in the package dsr_msgs2.
/**
  * ____________________________________________________________________________________________
  * Custom msg for RobotState.msg -- MAX_SIZE = 100
  * ____________________________________________________________________________________________
 */
typedef struct dsr_msgs2__msg__ModbusState
{
  /// Modbus Signal Name
  rosidl_runtime_c__String modbus_symbol;
  /// Modbus Register Value (Unsigned : 0 ~ 65535)
  int32_t modbus_value;
} dsr_msgs2__msg__ModbusState;

// Struct for a sequence of dsr_msgs2__msg__ModbusState.
typedef struct dsr_msgs2__msg__ModbusState__Sequence
{
  dsr_msgs2__msg__ModbusState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dsr_msgs2__msg__ModbusState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__MODBUS_STATE__STRUCT_H_
