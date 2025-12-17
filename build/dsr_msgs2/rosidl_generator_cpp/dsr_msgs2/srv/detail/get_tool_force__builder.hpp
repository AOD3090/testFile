// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetToolForce.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_TOOL_FORCE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_TOOL_FORCE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_tool_force__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetToolForce_Request_ref
{
public:
  Init_GetToolForce_Request_ref()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetToolForce_Request ref(::dsr_msgs2::srv::GetToolForce_Request::_ref_type arg)
  {
    msg_.ref = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetToolForce_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetToolForce_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetToolForce_Request_ref();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetToolForce_Response_success
{
public:
  explicit Init_GetToolForce_Response_success(::dsr_msgs2::srv::GetToolForce_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetToolForce_Response success(::dsr_msgs2::srv::GetToolForce_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetToolForce_Response msg_;
};

class Init_GetToolForce_Response_tool_force
{
public:
  Init_GetToolForce_Response_tool_force()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetToolForce_Response_success tool_force(::dsr_msgs2::srv::GetToolForce_Response::_tool_force_type arg)
  {
    msg_.tool_force = std::move(arg);
    return Init_GetToolForce_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetToolForce_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetToolForce_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetToolForce_Response_tool_force();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_TOOL_FORCE__BUILDER_HPP_
