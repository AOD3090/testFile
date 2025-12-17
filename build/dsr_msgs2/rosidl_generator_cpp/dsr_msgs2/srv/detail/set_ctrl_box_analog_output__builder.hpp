// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetCtrlBoxAnalogOutput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_OUTPUT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_OUTPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_ctrl_box_analog_output__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetCtrlBoxAnalogOutput_Request_value
{
public:
  explicit Init_SetCtrlBoxAnalogOutput_Request_value(::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Request value(::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Request::_value_type arg)
  {
    msg_.value = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Request msg_;
};

class Init_SetCtrlBoxAnalogOutput_Request_channel
{
public:
  Init_SetCtrlBoxAnalogOutput_Request_channel()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetCtrlBoxAnalogOutput_Request_value channel(::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Request::_channel_type arg)
  {
    msg_.channel = std::move(arg);
    return Init_SetCtrlBoxAnalogOutput_Request_value(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetCtrlBoxAnalogOutput_Request_channel();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetCtrlBoxAnalogOutput_Response_success
{
public:
  Init_SetCtrlBoxAnalogOutput_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Response success(::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetCtrlBoxAnalogOutput_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetCtrlBoxAnalogOutput_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_OUTPUT__BUILDER_HPP_
