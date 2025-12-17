// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dsr_msgs2:msg/JogMultiAxis.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/msg/detail/jog_multi_axis__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
dsr_msgs2__msg__JogMultiAxis__init(dsr_msgs2__msg__JogMultiAxis * msg)
{
  if (!msg) {
    return false;
  }
  // jog_axis
  // move_reference
  // speed
  return true;
}

void
dsr_msgs2__msg__JogMultiAxis__fini(dsr_msgs2__msg__JogMultiAxis * msg)
{
  if (!msg) {
    return;
  }
  // jog_axis
  // move_reference
  // speed
}

bool
dsr_msgs2__msg__JogMultiAxis__are_equal(const dsr_msgs2__msg__JogMultiAxis * lhs, const dsr_msgs2__msg__JogMultiAxis * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // jog_axis
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->jog_axis[i] != rhs->jog_axis[i]) {
      return false;
    }
  }
  // move_reference
  if (lhs->move_reference != rhs->move_reference) {
    return false;
  }
  // speed
  if (lhs->speed != rhs->speed) {
    return false;
  }
  return true;
}

bool
dsr_msgs2__msg__JogMultiAxis__copy(
  const dsr_msgs2__msg__JogMultiAxis * input,
  dsr_msgs2__msg__JogMultiAxis * output)
{
  if (!input || !output) {
    return false;
  }
  // jog_axis
  for (size_t i = 0; i < 6; ++i) {
    output->jog_axis[i] = input->jog_axis[i];
  }
  // move_reference
  output->move_reference = input->move_reference;
  // speed
  output->speed = input->speed;
  return true;
}

dsr_msgs2__msg__JogMultiAxis *
dsr_msgs2__msg__JogMultiAxis__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__JogMultiAxis * msg = (dsr_msgs2__msg__JogMultiAxis *)allocator.allocate(sizeof(dsr_msgs2__msg__JogMultiAxis), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dsr_msgs2__msg__JogMultiAxis));
  bool success = dsr_msgs2__msg__JogMultiAxis__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dsr_msgs2__msg__JogMultiAxis__destroy(dsr_msgs2__msg__JogMultiAxis * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dsr_msgs2__msg__JogMultiAxis__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dsr_msgs2__msg__JogMultiAxis__Sequence__init(dsr_msgs2__msg__JogMultiAxis__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__JogMultiAxis * data = NULL;

  if (size) {
    data = (dsr_msgs2__msg__JogMultiAxis *)allocator.zero_allocate(size, sizeof(dsr_msgs2__msg__JogMultiAxis), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dsr_msgs2__msg__JogMultiAxis__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dsr_msgs2__msg__JogMultiAxis__fini(&data[i - 1]);
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
dsr_msgs2__msg__JogMultiAxis__Sequence__fini(dsr_msgs2__msg__JogMultiAxis__Sequence * array)
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
      dsr_msgs2__msg__JogMultiAxis__fini(&array->data[i]);
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

dsr_msgs2__msg__JogMultiAxis__Sequence *
dsr_msgs2__msg__JogMultiAxis__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__msg__JogMultiAxis__Sequence * array = (dsr_msgs2__msg__JogMultiAxis__Sequence *)allocator.allocate(sizeof(dsr_msgs2__msg__JogMultiAxis__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dsr_msgs2__msg__JogMultiAxis__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dsr_msgs2__msg__JogMultiAxis__Sequence__destroy(dsr_msgs2__msg__JogMultiAxis__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dsr_msgs2__msg__JogMultiAxis__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dsr_msgs2__msg__JogMultiAxis__Sequence__are_equal(const dsr_msgs2__msg__JogMultiAxis__Sequence * lhs, const dsr_msgs2__msg__JogMultiAxis__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dsr_msgs2__msg__JogMultiAxis__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__msg__JogMultiAxis__Sequence__copy(
  const dsr_msgs2__msg__JogMultiAxis__Sequence * input,
  dsr_msgs2__msg__JogMultiAxis__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dsr_msgs2__msg__JogMultiAxis);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dsr_msgs2__msg__JogMultiAxis * data =
      (dsr_msgs2__msg__JogMultiAxis *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dsr_msgs2__msg__JogMultiAxis__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dsr_msgs2__msg__JogMultiAxis__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dsr_msgs2__msg__JogMultiAxis__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
