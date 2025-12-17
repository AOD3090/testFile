// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dsr_msgs2:srv/Ikin.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/srv/detail/ikin__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
dsr_msgs2__srv__Ikin_Request__init(dsr_msgs2__srv__Ikin_Request * msg)
{
  if (!msg) {
    return false;
  }
  // pos
  // sol_space
  // ref
  return true;
}

void
dsr_msgs2__srv__Ikin_Request__fini(dsr_msgs2__srv__Ikin_Request * msg)
{
  if (!msg) {
    return;
  }
  // pos
  // sol_space
  // ref
}

bool
dsr_msgs2__srv__Ikin_Request__are_equal(const dsr_msgs2__srv__Ikin_Request * lhs, const dsr_msgs2__srv__Ikin_Request * rhs)
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
  // sol_space
  if (lhs->sol_space != rhs->sol_space) {
    return false;
  }
  // ref
  if (lhs->ref != rhs->ref) {
    return false;
  }
  return true;
}

bool
dsr_msgs2__srv__Ikin_Request__copy(
  const dsr_msgs2__srv__Ikin_Request * input,
  dsr_msgs2__srv__Ikin_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // pos
  for (size_t i = 0; i < 6; ++i) {
    output->pos[i] = input->pos[i];
  }
  // sol_space
  output->sol_space = input->sol_space;
  // ref
  output->ref = input->ref;
  return true;
}

dsr_msgs2__srv__Ikin_Request *
dsr_msgs2__srv__Ikin_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__Ikin_Request * msg = (dsr_msgs2__srv__Ikin_Request *)allocator.allocate(sizeof(dsr_msgs2__srv__Ikin_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dsr_msgs2__srv__Ikin_Request));
  bool success = dsr_msgs2__srv__Ikin_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dsr_msgs2__srv__Ikin_Request__destroy(dsr_msgs2__srv__Ikin_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dsr_msgs2__srv__Ikin_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dsr_msgs2__srv__Ikin_Request__Sequence__init(dsr_msgs2__srv__Ikin_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__Ikin_Request * data = NULL;

  if (size) {
    data = (dsr_msgs2__srv__Ikin_Request *)allocator.zero_allocate(size, sizeof(dsr_msgs2__srv__Ikin_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dsr_msgs2__srv__Ikin_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dsr_msgs2__srv__Ikin_Request__fini(&data[i - 1]);
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
dsr_msgs2__srv__Ikin_Request__Sequence__fini(dsr_msgs2__srv__Ikin_Request__Sequence * array)
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
      dsr_msgs2__srv__Ikin_Request__fini(&array->data[i]);
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

dsr_msgs2__srv__Ikin_Request__Sequence *
dsr_msgs2__srv__Ikin_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__Ikin_Request__Sequence * array = (dsr_msgs2__srv__Ikin_Request__Sequence *)allocator.allocate(sizeof(dsr_msgs2__srv__Ikin_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dsr_msgs2__srv__Ikin_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dsr_msgs2__srv__Ikin_Request__Sequence__destroy(dsr_msgs2__srv__Ikin_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dsr_msgs2__srv__Ikin_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dsr_msgs2__srv__Ikin_Request__Sequence__are_equal(const dsr_msgs2__srv__Ikin_Request__Sequence * lhs, const dsr_msgs2__srv__Ikin_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dsr_msgs2__srv__Ikin_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__srv__Ikin_Request__Sequence__copy(
  const dsr_msgs2__srv__Ikin_Request__Sequence * input,
  dsr_msgs2__srv__Ikin_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dsr_msgs2__srv__Ikin_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dsr_msgs2__srv__Ikin_Request * data =
      (dsr_msgs2__srv__Ikin_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dsr_msgs2__srv__Ikin_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dsr_msgs2__srv__Ikin_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dsr_msgs2__srv__Ikin_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
dsr_msgs2__srv__Ikin_Response__init(dsr_msgs2__srv__Ikin_Response * msg)
{
  if (!msg) {
    return false;
  }
  // conv_posj
  // success
  return true;
}

void
dsr_msgs2__srv__Ikin_Response__fini(dsr_msgs2__srv__Ikin_Response * msg)
{
  if (!msg) {
    return;
  }
  // conv_posj
  // success
}

bool
dsr_msgs2__srv__Ikin_Response__are_equal(const dsr_msgs2__srv__Ikin_Response * lhs, const dsr_msgs2__srv__Ikin_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // conv_posj
  for (size_t i = 0; i < 6; ++i) {
    if (lhs->conv_posj[i] != rhs->conv_posj[i]) {
      return false;
    }
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  return true;
}

bool
dsr_msgs2__srv__Ikin_Response__copy(
  const dsr_msgs2__srv__Ikin_Response * input,
  dsr_msgs2__srv__Ikin_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // conv_posj
  for (size_t i = 0; i < 6; ++i) {
    output->conv_posj[i] = input->conv_posj[i];
  }
  // success
  output->success = input->success;
  return true;
}

dsr_msgs2__srv__Ikin_Response *
dsr_msgs2__srv__Ikin_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__Ikin_Response * msg = (dsr_msgs2__srv__Ikin_Response *)allocator.allocate(sizeof(dsr_msgs2__srv__Ikin_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dsr_msgs2__srv__Ikin_Response));
  bool success = dsr_msgs2__srv__Ikin_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dsr_msgs2__srv__Ikin_Response__destroy(dsr_msgs2__srv__Ikin_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dsr_msgs2__srv__Ikin_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dsr_msgs2__srv__Ikin_Response__Sequence__init(dsr_msgs2__srv__Ikin_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__Ikin_Response * data = NULL;

  if (size) {
    data = (dsr_msgs2__srv__Ikin_Response *)allocator.zero_allocate(size, sizeof(dsr_msgs2__srv__Ikin_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dsr_msgs2__srv__Ikin_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dsr_msgs2__srv__Ikin_Response__fini(&data[i - 1]);
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
dsr_msgs2__srv__Ikin_Response__Sequence__fini(dsr_msgs2__srv__Ikin_Response__Sequence * array)
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
      dsr_msgs2__srv__Ikin_Response__fini(&array->data[i]);
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

dsr_msgs2__srv__Ikin_Response__Sequence *
dsr_msgs2__srv__Ikin_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__Ikin_Response__Sequence * array = (dsr_msgs2__srv__Ikin_Response__Sequence *)allocator.allocate(sizeof(dsr_msgs2__srv__Ikin_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dsr_msgs2__srv__Ikin_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dsr_msgs2__srv__Ikin_Response__Sequence__destroy(dsr_msgs2__srv__Ikin_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dsr_msgs2__srv__Ikin_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dsr_msgs2__srv__Ikin_Response__Sequence__are_equal(const dsr_msgs2__srv__Ikin_Response__Sequence * lhs, const dsr_msgs2__srv__Ikin_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dsr_msgs2__srv__Ikin_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__srv__Ikin_Response__Sequence__copy(
  const dsr_msgs2__srv__Ikin_Response__Sequence * input,
  dsr_msgs2__srv__Ikin_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dsr_msgs2__srv__Ikin_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dsr_msgs2__srv__Ikin_Response * data =
      (dsr_msgs2__srv__Ikin_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dsr_msgs2__srv__Ikin_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dsr_msgs2__srv__Ikin_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dsr_msgs2__srv__Ikin_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
