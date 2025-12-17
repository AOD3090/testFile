// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from dsr_msgs2:msg/RobotStateRt.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__FUNCTIONS_H_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "dsr_msgs2/msg/rosidl_generator_c__visibility_control.h"

#include "dsr_msgs2/msg/detail/robot_state_rt__struct.h"

/// Initialize msg/RobotStateRt message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * dsr_msgs2__msg__RobotStateRt
 * )) before or use
 * dsr_msgs2__msg__RobotStateRt__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__msg__RobotStateRt__init(dsr_msgs2__msg__RobotStateRt * msg);

/// Finalize msg/RobotStateRt message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__msg__RobotStateRt__fini(dsr_msgs2__msg__RobotStateRt * msg);

/// Create msg/RobotStateRt message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * dsr_msgs2__msg__RobotStateRt__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
dsr_msgs2__msg__RobotStateRt *
dsr_msgs2__msg__RobotStateRt__create();

/// Destroy msg/RobotStateRt message.
/**
 * It calls
 * dsr_msgs2__msg__RobotStateRt__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__msg__RobotStateRt__destroy(dsr_msgs2__msg__RobotStateRt * msg);

/// Check for msg/RobotStateRt message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__msg__RobotStateRt__are_equal(const dsr_msgs2__msg__RobotStateRt * lhs, const dsr_msgs2__msg__RobotStateRt * rhs);

/// Copy a msg/RobotStateRt message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__msg__RobotStateRt__copy(
  const dsr_msgs2__msg__RobotStateRt * input,
  dsr_msgs2__msg__RobotStateRt * output);

/// Initialize array of msg/RobotStateRt messages.
/**
 * It allocates the memory for the number of elements and calls
 * dsr_msgs2__msg__RobotStateRt__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__msg__RobotStateRt__Sequence__init(dsr_msgs2__msg__RobotStateRt__Sequence * array, size_t size);

/// Finalize array of msg/RobotStateRt messages.
/**
 * It calls
 * dsr_msgs2__msg__RobotStateRt__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__msg__RobotStateRt__Sequence__fini(dsr_msgs2__msg__RobotStateRt__Sequence * array);

/// Create array of msg/RobotStateRt messages.
/**
 * It allocates the memory for the array and calls
 * dsr_msgs2__msg__RobotStateRt__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
dsr_msgs2__msg__RobotStateRt__Sequence *
dsr_msgs2__msg__RobotStateRt__Sequence__create(size_t size);

/// Destroy array of msg/RobotStateRt messages.
/**
 * It calls
 * dsr_msgs2__msg__RobotStateRt__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__msg__RobotStateRt__Sequence__destroy(dsr_msgs2__msg__RobotStateRt__Sequence * array);

/// Check for msg/RobotStateRt message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__msg__RobotStateRt__Sequence__are_equal(const dsr_msgs2__msg__RobotStateRt__Sequence * lhs, const dsr_msgs2__msg__RobotStateRt__Sequence * rhs);

/// Copy an array of msg/RobotStateRt messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__msg__RobotStateRt__Sequence__copy(
  const dsr_msgs2__msg__RobotStateRt__Sequence * input,
  dsr_msgs2__msg__RobotStateRt__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STATE_RT__FUNCTIONS_H_
