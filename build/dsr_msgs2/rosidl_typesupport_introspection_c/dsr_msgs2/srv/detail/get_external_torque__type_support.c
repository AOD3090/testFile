// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from dsr_msgs2:srv/GetExternalTorque.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "dsr_msgs2/srv/detail/get_external_torque__rosidl_typesupport_introspection_c.h"
#include "dsr_msgs2/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "dsr_msgs2/srv/detail/get_external_torque__functions.h"
#include "dsr_msgs2/srv/detail/get_external_torque__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  dsr_msgs2__srv__GetExternalTorque_Request__init(message_memory);
}

void dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_fini_function(void * message_memory)
{
  dsr_msgs2__srv__GetExternalTorque_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__srv__GetExternalTorque_Request, structure_needs_at_least_one_member),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_message_members = {
  "dsr_msgs2__srv",  // message namespace
  "GetExternalTorque_Request",  // message name
  1,  // number of fields
  sizeof(dsr_msgs2__srv__GetExternalTorque_Request),
  dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_message_member_array,  // message members
  dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_message_type_support_handle = {
  0,
  &dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_dsr_msgs2
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, srv, GetExternalTorque_Request)() {
  if (!dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_message_type_support_handle.typesupport_identifier) {
    dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &dsr_msgs2__srv__GetExternalTorque_Request__rosidl_typesupport_introspection_c__GetExternalTorque_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "dsr_msgs2/srv/detail/get_external_torque__rosidl_typesupport_introspection_c.h"
// already included above
// #include "dsr_msgs2/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "dsr_msgs2/srv/detail/get_external_torque__functions.h"
// already included above
// #include "dsr_msgs2/srv/detail/get_external_torque__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  dsr_msgs2__srv__GetExternalTorque_Response__init(message_memory);
}

void dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_fini_function(void * message_memory)
{
  dsr_msgs2__srv__GetExternalTorque_Response__fini(message_memory);
}

size_t dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__size_function__GetExternalTorque_Response__ext_torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__get_const_function__GetExternalTorque_Response__ext_torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__get_function__GetExternalTorque_Response__ext_torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__fetch_function__GetExternalTorque_Response__ext_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__get_const_function__GetExternalTorque_Response__ext_torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__assign_function__GetExternalTorque_Response__ext_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__get_function__GetExternalTorque_Response__ext_torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_message_member_array[2] = {
  {
    "ext_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__srv__GetExternalTorque_Response, ext_torque),  // bytes offset in struct
    NULL,  // default value
    dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__size_function__GetExternalTorque_Response__ext_torque,  // size() function pointer
    dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__get_const_function__GetExternalTorque_Response__ext_torque,  // get_const(index) function pointer
    dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__get_function__GetExternalTorque_Response__ext_torque,  // get(index) function pointer
    dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__fetch_function__GetExternalTorque_Response__ext_torque,  // fetch(index, &value) function pointer
    dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__assign_function__GetExternalTorque_Response__ext_torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2__srv__GetExternalTorque_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_message_members = {
  "dsr_msgs2__srv",  // message namespace
  "GetExternalTorque_Response",  // message name
  2,  // number of fields
  sizeof(dsr_msgs2__srv__GetExternalTorque_Response),
  dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_message_member_array,  // message members
  dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_message_type_support_handle = {
  0,
  &dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_dsr_msgs2
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, srv, GetExternalTorque_Response)() {
  if (!dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_message_type_support_handle.typesupport_identifier) {
    dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &dsr_msgs2__srv__GetExternalTorque_Response__rosidl_typesupport_introspection_c__GetExternalTorque_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "dsr_msgs2/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "dsr_msgs2/srv/detail/get_external_torque__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers dsr_msgs2__srv__detail__get_external_torque__rosidl_typesupport_introspection_c__GetExternalTorque_service_members = {
  "dsr_msgs2__srv",  // service namespace
  "GetExternalTorque",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // dsr_msgs2__srv__detail__get_external_torque__rosidl_typesupport_introspection_c__GetExternalTorque_Request_message_type_support_handle,
  NULL  // response message
  // dsr_msgs2__srv__detail__get_external_torque__rosidl_typesupport_introspection_c__GetExternalTorque_Response_message_type_support_handle
};

static rosidl_service_type_support_t dsr_msgs2__srv__detail__get_external_torque__rosidl_typesupport_introspection_c__GetExternalTorque_service_type_support_handle = {
  0,
  &dsr_msgs2__srv__detail__get_external_torque__rosidl_typesupport_introspection_c__GetExternalTorque_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, srv, GetExternalTorque_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, srv, GetExternalTorque_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_dsr_msgs2
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, srv, GetExternalTorque)() {
  if (!dsr_msgs2__srv__detail__get_external_torque__rosidl_typesupport_introspection_c__GetExternalTorque_service_type_support_handle.typesupport_identifier) {
    dsr_msgs2__srv__detail__get_external_torque__rosidl_typesupport_introspection_c__GetExternalTorque_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)dsr_msgs2__srv__detail__get_external_torque__rosidl_typesupport_introspection_c__GetExternalTorque_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, srv, GetExternalTorque_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dsr_msgs2, srv, GetExternalTorque_Response)()->data;
  }

  return &dsr_msgs2__srv__detail__get_external_torque__rosidl_typesupport_introspection_c__GetExternalTorque_service_type_support_handle;
}
