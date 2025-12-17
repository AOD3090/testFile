// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetStiffnessx.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_STIFFNESSX__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_STIFFNESSX__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_stiffnessx__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetStiffnessx_Request_time
{
public:
  explicit Init_SetStiffnessx_Request_time(::dsr_msgs2::srv::SetStiffnessx_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetStiffnessx_Request time(::dsr_msgs2::srv::SetStiffnessx_Request::_time_type arg)
  {
    msg_.time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetStiffnessx_Request msg_;
};

class Init_SetStiffnessx_Request_ref
{
public:
  explicit Init_SetStiffnessx_Request_ref(::dsr_msgs2::srv::SetStiffnessx_Request & msg)
  : msg_(msg)
  {}
  Init_SetStiffnessx_Request_time ref(::dsr_msgs2::srv::SetStiffnessx_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return Init_SetStiffnessx_Request_time(msg_);
  }

private:
  ::dsr_msgs2::srv::SetStiffnessx_Request msg_;
};

class Init_SetStiffnessx_Request_stx
{
public:
  Init_SetStiffnessx_Request_stx()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetStiffnessx_Request_ref stx(::dsr_msgs2::srv::SetStiffnessx_Request::_stx_type arg)
  {
    msg_.stx = std::move(arg);
    return Init_SetStiffnessx_Request_ref(msg_);
  }

private:
  ::dsr_msgs2::srv::SetStiffnessx_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetStiffnessx_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetStiffnessx_Request_stx();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetStiffnessx_Response_success
{
public:
  Init_SetStiffnessx_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetStiffnessx_Response success(::dsr_msgs2::srv::SetStiffnessx_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetStiffnessx_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetStiffnessx_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetStiffnessx_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_STIFFNESSX__BUILDER_HPP_
