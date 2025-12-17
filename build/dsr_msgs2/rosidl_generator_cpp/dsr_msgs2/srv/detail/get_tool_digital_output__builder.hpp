// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetToolDigitalOutput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_TOOL_DIGITAL_OUTPUT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_TOOL_DIGITAL_OUTPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_tool_digital_output__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetToolDigitalOutput_Request_index
{
public:
  Init_GetToolDigitalOutput_Request_index()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetToolDigitalOutput_Request index(::dsr_msgs2::srv::GetToolDigitalOutput_Request::_index_type arg)
  {
    msg_.index = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetToolDigitalOutput_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetToolDigitalOutput_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetToolDigitalOutput_Request_index();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetToolDigitalOutput_Response_success
{
public:
  explicit Init_GetToolDigitalOutput_Response_success(::dsr_msgs2::srv::GetToolDigitalOutput_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetToolDigitalOutput_Response success(::dsr_msgs2::srv::GetToolDigitalOutput_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetToolDigitalOutput_Response msg_;
};

class Init_GetToolDigitalOutput_Response_value
{
public:
  Init_GetToolDigitalOutput_Response_value()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetToolDigitalOutput_Response_success value(::dsr_msgs2::srv::GetToolDigitalOutput_Response::_value_type arg)
  {
    msg_.value = std::move(arg);
    return Init_GetToolDigitalOutput_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetToolDigitalOutput_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetToolDigitalOutput_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetToolDigitalOutput_Response_value();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_TOOL_DIGITAL_OUTPUT__BUILDER_HPP_
