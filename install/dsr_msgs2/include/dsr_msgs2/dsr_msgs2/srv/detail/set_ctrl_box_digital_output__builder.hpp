// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetCtrlBoxDigitalOutput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_DIGITAL_OUTPUT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_DIGITAL_OUTPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_ctrl_box_digital_output__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetCtrlBoxDigitalOutput_Request_value
{
public:
  explicit Init_SetCtrlBoxDigitalOutput_Request_value(::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Request value(::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Request::_value_type arg)
  {
    msg_.value = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Request msg_;
};

class Init_SetCtrlBoxDigitalOutput_Request_index
{
public:
  Init_SetCtrlBoxDigitalOutput_Request_index()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetCtrlBoxDigitalOutput_Request_value index(::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Request::_index_type arg)
  {
    msg_.index = std::move(arg);
    return Init_SetCtrlBoxDigitalOutput_Request_value(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetCtrlBoxDigitalOutput_Request_index();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetCtrlBoxDigitalOutput_Response_success
{
public:
  Init_SetCtrlBoxDigitalOutput_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Response success(::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetCtrlBoxDigitalOutput_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetCtrlBoxDigitalOutput_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_CTRL_BOX_DIGITAL_OUTPUT__BUILDER_HPP_
