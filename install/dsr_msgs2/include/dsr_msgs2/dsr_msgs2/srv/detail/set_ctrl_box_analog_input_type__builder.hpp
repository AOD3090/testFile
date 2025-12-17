// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetCtrlBoxAnalogInputType.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_INPUT_TYPE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_INPUT_TYPE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_ctrl_box_analog_input_type__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetCtrlBoxAnalogInputType_Request_mode
{
public:
  explicit Init_SetCtrlBoxAnalogInputType_Request_mode(::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request mode(::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request msg_;
};

class Init_SetCtrlBoxAnalogInputType_Request_channel
{
public:
  Init_SetCtrlBoxAnalogInputType_Request_channel()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetCtrlBoxAnalogInputType_Request_mode channel(::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request::_channel_type arg)
  {
    msg_.channel = std::move(arg);
    return Init_SetCtrlBoxAnalogInputType_Request_mode(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetCtrlBoxAnalogInputType_Request_channel();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetCtrlBoxAnalogInputType_Response_success
{
public:
  Init_SetCtrlBoxAnalogInputType_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response success(::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetCtrlBoxAnalogInputType_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetCtrlBoxAnalogInputType_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_ANALOG_INPUT_TYPE__BUILDER_HPP_
