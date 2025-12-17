// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetRtControlInput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_RT_CONTROL_INPUT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_RT_CONTROL_INPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_rt_control_input__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRtControlInput_Request_loss
{
public:
  explicit Init_SetRtControlInput_Request_loss(::dsr_msgs2::srv::SetRtControlInput_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetRtControlInput_Request loss(::dsr_msgs2::srv::SetRtControlInput_Request::_loss_type arg)
  {
    msg_.loss = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRtControlInput_Request msg_;
};

class Init_SetRtControlInput_Request_period
{
public:
  explicit Init_SetRtControlInput_Request_period(::dsr_msgs2::srv::SetRtControlInput_Request & msg)
  : msg_(msg)
  {}
  Init_SetRtControlInput_Request_loss period(::dsr_msgs2::srv::SetRtControlInput_Request::_period_type arg)
  {
    msg_.period = std::move(arg);
    return Init_SetRtControlInput_Request_loss(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRtControlInput_Request msg_;
};

class Init_SetRtControlInput_Request_version
{
public:
  Init_SetRtControlInput_Request_version()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetRtControlInput_Request_period version(::dsr_msgs2::srv::SetRtControlInput_Request::_version_type arg)
  {
    msg_.version = std::move(arg);
    return Init_SetRtControlInput_Request_period(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRtControlInput_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRtControlInput_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetRtControlInput_Request_version();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetRtControlInput_Response_success
{
public:
  Init_SetRtControlInput_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetRtControlInput_Response success(::dsr_msgs2::srv::SetRtControlInput_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetRtControlInput_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetRtControlInput_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetRtControlInput_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_RT_CONTROL_INPUT__BUILDER_HPP_
