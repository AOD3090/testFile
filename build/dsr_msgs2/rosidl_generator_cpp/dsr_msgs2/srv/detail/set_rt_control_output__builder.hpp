// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetRtControlOutput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_RT_CONTROL_OUTPUT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_RT_CONTROL_OUTPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_rt_control_output__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRtControlOutput_Request_loss
{
public:
  explicit Init_SetRtControlOutput_Request_loss(::dsr_msgs2::srv::SetRtControlOutput_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetRtControlOutput_Request loss(::dsr_msgs2::srv::SetRtControlOutput_Request::_loss_type arg)
  {
    msg_.loss = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRtControlOutput_Request msg_;
};

class Init_SetRtControlOutput_Request_period
{
public:
  explicit Init_SetRtControlOutput_Request_period(::dsr_msgs2::srv::SetRtControlOutput_Request & msg)
  : msg_(msg)
  {}
  Init_SetRtControlOutput_Request_loss period(::dsr_msgs2::srv::SetRtControlOutput_Request::_period_type arg)
  {
    msg_.period = std::move(arg);
    return Init_SetRtControlOutput_Request_loss(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRtControlOutput_Request msg_;
};

class Init_SetRtControlOutput_Request_version
{
public:
  Init_SetRtControlOutput_Request_version()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetRtControlOutput_Request_period version(::dsr_msgs2::srv::SetRtControlOutput_Request::_version_type arg)
  {
    msg_.version = std::move(arg);
    return Init_SetRtControlOutput_Request_period(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRtControlOutput_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRtControlOutput_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetRtControlOutput_Request_version();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRtControlOutput_Response_success
{
public:
  Init_SetRtControlOutput_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRtControlOutput_Response success(::dsr_msgs2::srv::SetRtControlOutput_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRtControlOutput_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRtControlOutput_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetRtControlOutput_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_RT_CONTROL_OUTPUT__BUILDER_HPP_
