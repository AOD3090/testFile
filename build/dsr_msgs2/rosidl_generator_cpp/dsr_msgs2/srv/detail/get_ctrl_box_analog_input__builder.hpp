// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetCtrlBoxAnalogInput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_CTRL_BOX_ANALOG_INPUT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_CTRL_BOX_ANALOG_INPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_ctrl_box_analog_input__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetCtrlBoxAnalogInput_Request_channel
{
public:
  Init_GetCtrlBoxAnalogInput_Request_channel()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Request channel(::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Request::_channel_type arg)
  {
    msg_.channel = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetCtrlBoxAnalogInput_Request_channel();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetCtrlBoxAnalogInput_Response_success
{
public:
  explicit Init_GetCtrlBoxAnalogInput_Response_success(::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Response success(::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Response msg_;
};

class Init_GetCtrlBoxAnalogInput_Response_value
{
public:
  Init_GetCtrlBoxAnalogInput_Response_value()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetCtrlBoxAnalogInput_Response_success value(::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Response::_value_type arg)
  {
    msg_.value = std::move(arg);
    return Init_GetCtrlBoxAnalogInput_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetCtrlBoxAnalogInput_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetCtrlBoxAnalogInput_Response_value();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_CTRL_BOX_ANALOG_INPUT__BUILDER_HPP_
