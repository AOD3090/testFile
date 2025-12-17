// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetSafetyMode.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_SAFETY_MODE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_SAFETY_MODE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_safety_mode__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetSafetyMode_Request_safety_event
{
public:
  explicit Init_SetSafetyMode_Request_safety_event(::dsr_msgs2::srv::SetSafetyMode_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetSafetyMode_Request safety_event(::dsr_msgs2::srv::SetSafetyMode_Request::_safety_event_type arg)
  {
    msg_.safety_event = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetSafetyMode_Request msg_;
};

class Init_SetSafetyMode_Request_safety_mode
{
public:
  Init_SetSafetyMode_Request_safety_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetSafetyMode_Request_safety_event safety_mode(::dsr_msgs2::srv::SetSafetyMode_Request::_safety_mode_type arg)
  {
    msg_.safety_mode = std::move(arg);
    return Init_SetSafetyMode_Request_safety_event(msg_);
  }

private:
  ::dsr_msgs2::srv::SetSafetyMode_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetSafetyMode_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetSafetyMode_Request_safety_mode();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetSafetyMode_Response_success
{
public:
  Init_SetSafetyMode_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetSafetyMode_Response success(::dsr_msgs2::srv::SetSafetyMode_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetSafetyMode_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetSafetyMode_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetSafetyMode_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_SAFETY_MODE__BUILDER_HPP_
