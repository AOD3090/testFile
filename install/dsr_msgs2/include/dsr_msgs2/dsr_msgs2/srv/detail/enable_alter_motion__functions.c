// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dsr_msgs2:srv/EnableAlterMotion.idl
// generated code does not contain a copyright notice
#include "dsr_msgs2/srv/detail/enable_alter_motion__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

bool
dsr_msgs2__srv__EnableAlterMotion_Request__init(dsr_msgs2__srv__EnableAlterMotion_Request * msg)
{
  if (!msg) {
    return false;
  }
  // n
  // mode
  // ref
  // limit_dpos
  // limit_dpos_per
  return true;
}

void
dsr_msgs2__srv__EnableAlterMotion_Request__fini(dsr_msgs2__srv__EnableAlterMotion_Request * msg)
{
  if (!msg) {
    return;
  }
  // n
  // mode
  // ref
  // limit_dpos
  // limit_dpos_per
}

bool
dsr_msgs2__srv__EnableAlterMotion_Request__are_equal(const dsr_msgs2__srv__EnableAlterMotion_Request * lhs, const dsr_msgs2__srv__EnableAlterMotion_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // n
  if (lhs->n != rhs->n) {
    return false;
  }
  // mode
  if (lhs->mode != rhs->mode) {
    return false;
  }
  // ref
  if (lhs->ref != rhs->ref) {
    return false;
  }
  // limit_dpos
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->limit_dpos[i] != rhs->limit_dpos[i]) {
      return false;
    }
  }
  // limit_dpos_per
  for (size_t i = 0; i < 2; ++i) {
    if (lhs->limit_dpos_per[i] != rhs->limit_dpos_per[i]) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__srv__EnableAlterMotion_Request__copy(
  const dsr_msgs2__srv__EnableAlterMotion_Request * input,
  dsr_msgs2__srv__EnableAlterMotion_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // n
  output->n = input->n;
  // mode
  output->mode = input->mode;
  // ref
  output->ref = input->ref;
  // limit_dpos
  for (size_t i = 0; i < 2; ++i) {
    output->limit_dpos[i] = input->limit_dpos[i];
  }
  // limit_dpos_per
  for (size_t i = 0; i < 2; ++i) {
    output->limit_dpos_per[i] = input->limit_dpos_per[i];
  }
  return true;
}

dsr_msgs2__srv__EnableAlterMotion_Request *
dsr_msgs2__srv__EnableAlterMotion_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__EnableAlterMotion_Request * msg = (dsr_msgs2__srv__EnableAlterMotion_Request *)allocator.allocate(sizeof(dsr_msgs2__srv__EnableAlterMotion_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dsr_msgs2__srv__EnableAlterMotion_Request));
  bool success = dsr_msgs2__srv__EnableAlterMotion_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dsr_msgs2__srv__EnableAlterMotion_Request__destroy(dsr_msgs2__srv__EnableAlterMotion_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dsr_msgs2__srv__EnableAlterMotion_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dsr_msgs2__srv__EnableAlterMotion_Request__Sequence__init(dsr_msgs2__srv__EnableAlterMotion_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__EnableAlterMotion_Request * data = NULL;

  if (size) {
    data = (dsr_msgs2__srv__EnableAlterMotion_Request *)allocator.zero_allocate(size, sizeof(dsr_msgs2__srv__EnableAlterMotion_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dsr_msgs2__srv__EnableAlterMotion_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dsr_msgs2__srv__EnableAlterMotion_Request__fini(&data[i - 1]);
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
dsr_msgs2__srv__EnableAlterMotion_Request__Sequence__fini(dsr_msgs2__srv__EnableAlterMotion_Request__Sequence * array)
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
      dsr_msgs2__srv__EnableAlterMotion_Request__fini(&array->data[i]);
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

dsr_msgs2__srv__EnableAlterMotion_Request__Sequence *
dsr_msgs2__srv__EnableAlterMotion_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__EnableAlterMotion_Request__Sequence * array = (dsr_msgs2__srv__EnableAlterMotion_Request__Sequence *)allocator.allocate(sizeof(dsr_msgs2__srv__EnableAlterMotion_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dsr_msgs2__srv__EnableAlterMotion_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dsr_msgs2__srv__EnableAlterMotion_Request__Sequence__destroy(dsr_msgs2__srv__EnableAlterMotion_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dsr_msgs2__srv__EnableAlterMotion_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dsr_msgs2__srv__EnableAlterMotion_Request__Sequence__are_equal(const dsr_msgs2__srv__EnableAlterMotion_Request__Sequence * lhs, const dsr_msgs2__srv__EnableAlterMotion_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dsr_msgs2__srv__EnableAlterMotion_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__srv__EnableAlterMotion_Request__Sequence__copy(
  const dsr_msgs2__srv__EnableAlterMotion_Request__Sequence * input,
  dsr_msgs2__srv__EnableAlterMotion_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dsr_msgs2__srv__EnableAlterMotion_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dsr_msgs2__srv__EnableAlterMotion_Request * data =
      (dsr_msgs2__srv__EnableAlterMotion_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dsr_msgs2__srv__EnableAlterMotion_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dsr_msgs2__srv__EnableAlterMotion_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dsr_msgs2__srv__EnableAlterMotion_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
dsr_msgs2__srv__EnableAlterMotion_Response__init(dsr_msgs2__srv__EnableAlterMotion_Response * msg)
{
  if (!msg) {
    return false;
  }
  // success
  return true;
}

void
dsr_msgs2__srv__EnableAlterMotion_Response__fini(dsr_msgs2__srv__EnableAlterMotion_Response * msg)
{
  if (!msg) {
    return;
  }
  // success
}

bool
dsr_msgs2__srv__EnableAlterMotion_Response__are_equal(const dsr_msgs2__srv__EnableAlterMotion_Response * lhs, const dsr_msgs2__srv__EnableAlterMotion_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // success
  if (lhs->success != rhs->success) {
    return false;
  }
  return true;
}

bool
dsr_msgs2__srv__EnableAlterMotion_Response__copy(
  const dsr_msgs2__srv__EnableAlterMotion_Response * input,
  dsr_msgs2__srv__EnableAlterMotion_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // success
  output->success = input->success;
  return true;
}

dsr_msgs2__srv__EnableAlterMotion_Response *
dsr_msgs2__srv__EnableAlterMotion_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__EnableAlterMotion_Response * msg = (dsr_msgs2__srv__EnableAlterMotion_Response *)allocator.allocate(sizeof(dsr_msgs2__srv__EnableAlterMotion_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dsr_msgs2__srv__EnableAlterMotion_Response));
  bool success = dsr_msgs2__srv__EnableAlterMotion_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dsr_msgs2__srv__EnableAlterMotion_Response__destroy(dsr_msgs2__srv__EnableAlterMotion_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dsr_msgs2__srv__EnableAlterMotion_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dsr_msgs2__srv__EnableAlterMotion_Response__Sequence__init(dsr_msgs2__srv__EnableAlterMotion_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__EnableAlterMotion_Response * data = NULL;

  if (size) {
    data = (dsr_msgs2__srv__EnableAlterMotion_Response *)allocator.zero_allocate(size, sizeof(dsr_msgs2__srv__EnableAlterMotion_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dsr_msgs2__srv__EnableAlterMotion_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dsr_msgs2__srv__EnableAlterMotion_Response__fini(&data[i - 1]);
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
dsr_msgs2__srv__EnableAlterMotion_Response__Sequence__fini(dsr_msgs2__srv__EnableAlterMotion_Response__Sequence * array)
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
      dsr_msgs2__srv__EnableAlterMotion_Response__fini(&array->data[i]);
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

dsr_msgs2__srv__EnableAlterMotion_Response__Sequence *
dsr_msgs2__srv__EnableAlterMotion_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dsr_msgs2__srv__EnableAlterMotion_Response__Sequence * array = (dsr_msgs2__srv__EnableAlterMotion_Response__Sequence *)allocator.allocate(sizeof(dsr_msgs2__srv__EnableAlterMotion_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dsr_msgs2__srv__EnableAlterMotion_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dsr_msgs2__srv__EnableAlterMotion_Response__Sequence__destroy(dsr_msgs2__srv__EnableAlterMotion_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dsr_msgs2__srv__EnableAlterMotion_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dsr_msgs2__srv__EnableAlterMotion_Response__Sequence__are_equal(const dsr_msgs2__srv__EnableAlterMotion_Response__Sequence * lhs, const dsr_msgs2__srv__EnableAlterMotion_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dsr_msgs2__srv__EnableAlterMotion_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dsr_msgs2__srv__EnableAlterMotion_Response__Sequence__copy(
  const dsr_msgs2__srv__EnableAlterMotion_Response__Sequence * input,
  dsr_msgs2__srv__EnableAlterMotion_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dsr_msgs2__srv__EnableAlterMotion_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dsr_msgs2__srv__EnableAlterMotion_Response * data =
      (dsr_msgs2__srv__EnableAlterMotion_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dsr_msgs2__srv__EnableAlterMotion_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dsr_msgs2__srv__EnableAlterMotion_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dsr_msgs2__srv__EnableAlterMotion_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
