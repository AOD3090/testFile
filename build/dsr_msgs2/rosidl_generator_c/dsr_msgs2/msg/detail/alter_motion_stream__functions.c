// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dsr_msgs2:msg/AlterMotionStream.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/alter_motion_stream__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
dsr_msgs2__msg__AlterMotionStream__init(dsr_msgs2__msg__AlterMotionStream * msg)
{
  if (!msg) {
    return false;
  }
  // pos
  return true;
}

void
dsr_msgs2__msg__AlterMotionStream__fini(dsr_msgs2__msg__AlterMotionStream * msg)
{
  if (!msg) {
    return;
  }
  // pos
}

bool
dsr_msgs2__msg__AlterMotionStream__are_equal(const dsr_msgs2__msg__AlterMotionStream * lhs, const dsr_msgs2__msg__AlterMotionStream * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // pos
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->pos[i] != rhs->pos[i]) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__msg__AlterMotionStream__copy(
  const dsr_msgs2__msg__AlterMotionStream * input,
  dsr_msgs2__msg__AlterMotionStream * output)
{
  if (!input || !output) {
    return false;
  }
  // pos
  for (size_t i = 0; i < 6; ++i) {
    output->pos[i] = input->pos[i];
  }
  return true;
}

dsr_msgs2__msg__AlterMotionStream *
dsr_msgs2__msg__AlterMotionStream__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__AlterMotionStream * msg = (dsr_msgs2__msg__AlterMotionStream *)allocator.allocate(sizeof(dsr_msgs2__msg__AlterMotionStream), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dsr_msgs2__msg__AlterMotionStream));
  bool success = dsr_msgs2__msg__AlterMotionStream__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dsr_msgs2__msg__AlterMotionStream__destroy(dsr_msgs2__msg__AlterMotionStream * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dsr_msgs2__msg__AlterMotionStream__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dsr_msgs2__msg__AlterMotionStream__Sequence__init(dsr_msgs2__msg__AlterMotionStream__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__AlterMotionStream * data = NULL;

  if (size) {
    data = (dsr_msgs2__msg__AlterMotionStream *)allocator.zero_allocate(size, sizeof(dsr_msgs2__msg__AlterMotionStream), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dsr_msgs2__msg__AlterMotionStream__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dsr_msgs2__msg__AlterMotionStream__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
dsr_msgs2__msg__AlterMotionStream__Sequence__fini(dsr_msgs2__msg__AlterMotionStream__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      dsr_msgs2__msg__AlterMotionStream__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

dsr_msgs2__msg__AlterMotionStream__Sequence *
dsr_msgs2__msg__AlterMotionStream__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__AlterMotionStream__Sequence * array = (dsr_msgs2__msg__AlterMotionStream__Sequence *)allocator.allocate(sizeof(dsr_msgs2__msg__AlterMotionStream__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dsr_msgs2__msg__AlterMotionStream__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dsr_msgs2__msg__AlterMotionStream__Sequence__destroy(dsr_msgs2__msg__AlterMotionStream__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dsr_msgs2__msg__AlterMotionStream__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dsr_msgs2__msg__AlterMotionStream__Sequence__are_equal(const dsr_msgs2__msg__AlterMotionStream__Sequence * lhs, const dsr_msgs2__msg__AlterMotionStream__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dsr_msgs2__msg__AlterMotionStream__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__msg__AlterMotionStream__Sequence__copy(
  const dsr_msgs2__msg__AlterMotionStream__Sequence * input,
  dsr_msgs2__msg__AlterMotionStream__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dsr_msgs2__msg__AlterMotionStream);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dsr_msgs2__msg__AlterMotionStream * data =
      (dsr_msgs2__msg__AlterMotionStream *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dsr_msgs2__msg__AlterMotionStream__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dsr_msgs2__msg__AlterMotionStream__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dsr_msgs2__msg__AlterMotionStream__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
