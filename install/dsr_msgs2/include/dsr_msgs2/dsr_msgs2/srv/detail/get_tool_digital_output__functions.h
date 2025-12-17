// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from dsr_msgs2:srv/GetToolDigitalOutput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_TOOL_DIGITAL_OUTPUT__FUNCTIONS_H_
#define DSR_MSGS2__SRV__DETAIL__GET_TOOL_DIGITAL_OUTPUT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "dsr_msgs2/msg/rosidl_generator_c__visibility_control.h"

#include "dsr_msgs2/srv/detail/get_tool_digital_output__struct.h"

/// Initialize srv/GetToolDigitalOutput message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * dsr_msgs2__srv__GetToolDigitalOutput_Request
 * )) before or use
 * dsr_msgs2__srv__GetToolDigitalOutput_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__srv__GetToolDigitalOutput_Request__init(dsr_msgs2__srv__GetToolDigitalOutput_Request * msg);

/// Finalize srv/GetToolDigitalOutput message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__srv__GetToolDigitalOutput_Request__fini(dsr_msgs2__srv__GetToolDigitalOutput_Request * msg);

/// Create srv/GetToolDigitalOutput message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
dsr_msgs2__srv__GetToolDigitalOutput_Request *
dsr_msgs2__srv__GetToolDigitalOutput_Request__create();

/// Destroy srv/GetToolDigitalOutput message.
/**
 * It calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__srv__GetToolDigitalOutput_Request__destroy(dsr_msgs2__srv__GetToolDigitalOutput_Request * msg);

/// Check for srv/GetToolDigitalOutput message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__srv__GetToolDigitalOutput_Request__are_equal(const dsr_msgs2__srv__GetToolDigitalOutput_Request * lhs, const dsr_msgs2__srv__GetToolDigitalOutput_Request * rhs);

/// Copy a srv/GetToolDigitalOutput message.
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
dsr_msgs2__srv__GetToolDigitalOutput_Request__copy(
  const dsr_msgs2__srv__GetToolDigitalOutput_Request * input,
  dsr_msgs2__srv__GetToolDigitalOutput_Request * output);

/// Initialize array of srv/GetToolDigitalOutput messages.
/**
 * It allocates the memory for the number of elements and calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence__init(dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence * array, size_t size);

/// Finalize array of srv/GetToolDigitalOutput messages.
/**
 * It calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence__fini(dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence * array);

/// Create array of srv/GetToolDigitalOutput messages.
/**
 * It allocates the memory for the array and calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence *
dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence__create(size_t size);

/// Destroy array of srv/GetToolDigitalOutput messages.
/**
 * It calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence__destroy(dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence * array);

/// Check for srv/GetToolDigitalOutput message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence__are_equal(const dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence * lhs, const dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence * rhs);

/// Copy an array of srv/GetToolDigitalOutput messages.
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
dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence__copy(
  const dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence * input,
  dsr_msgs2__srv__GetToolDigitalOutput_Request__Sequence * output);

/// Initialize srv/GetToolDigitalOutput message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * dsr_msgs2__srv__GetToolDigitalOutput_Response
 * )) before or use
 * dsr_msgs2__srv__GetToolDigitalOutput_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__srv__GetToolDigitalOutput_Response__init(dsr_msgs2__srv__GetToolDigitalOutput_Response * msg);

/// Finalize srv/GetToolDigitalOutput message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__srv__GetToolDigitalOutput_Response__fini(dsr_msgs2__srv__GetToolDigitalOutput_Response * msg);

/// Create srv/GetToolDigitalOutput message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
dsr_msgs2__srv__GetToolDigitalOutput_Response *
dsr_msgs2__srv__GetToolDigitalOutput_Response__create();

/// Destroy srv/GetToolDigitalOutput message.
/**
 * It calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__srv__GetToolDigitalOutput_Response__destroy(dsr_msgs2__srv__GetToolDigitalOutput_Response * msg);

/// Check for srv/GetToolDigitalOutput message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__srv__GetToolDigitalOutput_Response__are_equal(const dsr_msgs2__srv__GetToolDigitalOutput_Response * lhs, const dsr_msgs2__srv__GetToolDigitalOutput_Response * rhs);

/// Copy a srv/GetToolDigitalOutput message.
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
dsr_msgs2__srv__GetToolDigitalOutput_Response__copy(
  const dsr_msgs2__srv__GetToolDigitalOutput_Response * input,
  dsr_msgs2__srv__GetToolDigitalOutput_Response * output);

/// Initialize array of srv/GetToolDigitalOutput messages.
/**
 * It allocates the memory for the number of elements and calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence__init(dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence * array, size_t size);

/// Finalize array of srv/GetToolDigitalOutput messages.
/**
 * It calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence__fini(dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence * array);

/// Create array of srv/GetToolDigitalOutput messages.
/**
 * It allocates the memory for the array and calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence *
dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence__create(size_t size);

/// Destroy array of srv/GetToolDigitalOutput messages.
/**
 * It calls
 * dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
void
dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence__destroy(dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence * array);

/// Check for srv/GetToolDigitalOutput message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dsr_msgs2
bool
dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence__are_equal(const dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence * lhs, const dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence * rhs);

/// Copy an array of srv/GetToolDigitalOutput messages.
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
dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence__copy(
  const dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence * input,
  dsr_msgs2__srv__GetToolDigitalOutput_Response__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // DSR_MSGS2__SRV__DETAIL__GET_TOOL_DIGITAL_OUTPUT__FUNCTIONS_H_
