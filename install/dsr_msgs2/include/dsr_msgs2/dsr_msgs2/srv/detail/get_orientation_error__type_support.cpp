// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from dsr_msgs2:srv/GetOrientationError.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "dsr_msgs2/srv/detail/get_orientation_error__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace dsr_msgs2
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void GetOrientationError_Request_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) dsr_msgs2::srv::GetOrientationError_Request(_init);
}

void GetOrientationError_Request_fini_function(void * message_memory)
{
  auto typed_message = static_cast<dsr_msgs2::srv::GetOrientationError_Request *>(message_memory);
  typed_message->~GetOrientationError_Request();
}

size_t size_function__GetOrientationError_Request__xd(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__GetOrientationError_Request__xd(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__GetOrientationError_Request__xd(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__GetOrientationError_Request__xd(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__GetOrientationError_Request__xd(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__GetOrientationError_Request__xd(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__GetOrientationError_Request__xd(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__GetOrientationError_Request__xc(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__GetOrientationError_Request__xc(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__GetOrientationError_Request__xc(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__GetOrientationError_Request__xc(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__GetOrientationError_Request__xc(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__GetOrientationError_Request__xc(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__GetOrientationError_Request__xc(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GetOrientationError_Request_message_member_array[3] = {
  {
    "xd",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::srv::GetOrientationError_Request, xd),  // bytes offset in struct
    nullptr,  // default value
    size_function__GetOrientationError_Request__xd,  // size() function pointer
    get_const_function__GetOrientationError_Request__xd,  // get_const(index) function pointer
    get_function__GetOrientationError_Request__xd,  // get(index) function pointer
    fetch_function__GetOrientationError_Request__xd,  // fetch(index, &value) function pointer
    assign_function__GetOrientationError_Request__xd,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "xc",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::srv::GetOrientationError_Request, xc),  // bytes offset in struct
    nullptr,  // default value
    size_function__GetOrientationError_Request__xc,  // size() function pointer
    get_const_function__GetOrientationError_Request__xc,  // get_const(index) function pointer
    get_function__GetOrientationError_Request__xc,  // get(index) function pointer
    fetch_function__GetOrientationError_Request__xc,  // fetch(index, &value) function pointer
    assign_function__GetOrientationError_Request__xc,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "axis",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::srv::GetOrientationError_Request, axis),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GetOrientationError_Request_message_members = {
  "dsr_msgs2::srv",  // message namespace
  "GetOrientationError_Request",  // message name
  3,  // number of fields
  sizeof(dsr_msgs2::srv::GetOrientationError_Request),
  GetOrientationError_Request_message_member_array,  // message members
  GetOrientationError_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  GetOrientationError_Request_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GetOrientationError_Request_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GetOrientationError_Request_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace dsr_msgs2


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<dsr_msgs2::srv::GetOrientationError_Request>()
{
  return &::dsr_msgs2::srv::rosidl_typesupport_introspection_cpp::GetOrientationError_Request_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, dsr_msgs2, srv, GetOrientationError_Request)() {
  return &::dsr_msgs2::srv::rosidl_typesupport_introspection_cpp::GetOrientationError_Request_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

// already included above
// #include "array"
// already included above
// #include "cstddef"
// already included above
// #include "string"
// already included above
// #include "vector"
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "dsr_msgs2/srv/detail/get_orientation_error__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/field_types.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace dsr_msgs2
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

void GetOrientationError_Response_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) dsr_msgs2::srv::GetOrientationError_Response(_init);
}

void GetOrientationError_Response_fini_function(void * message_memory)
{
  auto typed_message = static_cast<dsr_msgs2::srv::GetOrientationError_Response *>(message_memory);
  typed_message->~GetOrientationError_Response();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember GetOrientationError_Response_message_member_array[2] = {
  {
    "ori_error",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::srv::GetOrientationError_Response, ori_error),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "success",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(dsr_msgs2::srv::GetOrientationError_Response, success),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers GetOrientationError_Response_message_members = {
  "dsr_msgs2::srv",  // message namespace
  "GetOrientationError_Response",  // message name
  2,  // number of fields
  sizeof(dsr_msgs2::srv::GetOrientationError_Response),
  GetOrientationError_Response_message_member_array,  // message members
  GetOrientationError_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  GetOrientationError_Response_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t GetOrientationError_Response_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GetOrientationError_Response_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace dsr_msgs2


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<dsr_msgs2::srv::GetOrientationError_Response>()
{
  return &::dsr_msgs2::srv::rosidl_typesupport_introspection_cpp::GetOrientationError_Response_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, dsr_msgs2, srv, GetOrientationError_Response)() {
  return &::dsr_msgs2::srv::rosidl_typesupport_introspection_cpp::GetOrientationError_Response_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "rosidl_typesupport_introspection_cpp/visibility_control.h"
// already included above
// #include "dsr_msgs2/srv/detail/get_orientation_error__struct.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/identifier.hpp"
// already included above
// #include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/service_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/service_type_support_decl.hpp"

namespace dsr_msgs2
{

namespace srv
{

namespace rosidl_typesupport_introspection_cpp
{

// this is intentionally not const to allow initialization later to prevent an initialization race
static ::rosidl_typesupport_introspection_cpp::ServiceMembers GetOrientationError_service_members = {
  "dsr_msgs2::srv",  // service namespace
  "GetOrientationError",  // service name
  // these two fields are initialized below on the first access
  // see get_service_type_support_handle<dsr_msgs2::srv::GetOrientationError>()
  nullptr,  // request message
  nullptr  // response message
};

static const rosidl_service_type_support_t GetOrientationError_service_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &GetOrientationError_service_members,
  get_service_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace srv

}  // namespace dsr_msgs2


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
get_service_type_support_handle<dsr_msgs2::srv::GetOrientationError>()
{
  // get a handle to the value to be returned
  auto service_type_support =
    &::dsr_msgs2::srv::rosidl_typesupport_introspection_cpp::GetOrientationError_service_type_support_handle;
  // get a non-const and properly typed version of the data void *
  auto service_members = const_cast<::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
    static_cast<const ::rosidl_typesupport_introspection_cpp::ServiceMembers *>(
      service_type_support->data));
  // make sure that both the request_members_ and the response_members_ are initialized
  // if they are not, initialize them
  if (
    service_members->request_members_ == nullptr ||
    service_members->response_members_ == nullptr)
  {
    // initialize the request_members_ with the static function from the external library
    service_members->request_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::dsr_msgs2::srv::GetOrientationError_Request
      >()->data
      );
    // initialize the response_members_ with the static function from the external library
    service_members->response_members_ = static_cast<
      const ::rosidl_typesupport_introspection_cpp::MessageMembers *
      >(
      ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<
        ::dsr_msgs2::srv::GetOrientationError_Response
      >()->data
      );
  }
  // finally return the properly initialized service_type_support handle
  return service_type_support;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, dsr_msgs2, srv, GetOrientationError)() {
  return ::rosidl_typesupport_introspection_cpp::get_service_type_support_handle<dsr_msgs2::srv::GetOrientationError>();
}

#ifdef __cplusplus
}
#endif
