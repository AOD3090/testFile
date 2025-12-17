// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetModbusOutput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_MODBUS_OUTPUT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_MODBUS_OUTPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_modbus_output__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetModbusOutput_Request_value
{
public:
  explicit Init_SetModbusOutput_Request_value(::dsr_msgs2::srv::SetModbusOutput_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::SetModbusOutput_Request value(::dsr_msgs2::srv::SetModbusOutput_Request::_value_type arg)
  {
    msg_.value = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetModbusOutput_Request msg_;
};

class Init_SetModbusOutput_Request_name
{
public:
  Init_SetModbusOutput_Request_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetModbusOutput_Request_value name(::dsr_msgs2::srv::SetModbusOutput_Request::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_SetModbusOutput_Request_value(msg_);
  }

private:
  ::dsr_msgs2::srv::SetModbusOutput_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetModbusOutput_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetModbusOutput_Request_name();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetModbusOutput_Response_success
{
public:
  Init_SetModbusOutput_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetModbusOutput_Response success(::dsr_msgs2::srv::SetModbusOutput_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetModbusOutput_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetModbusOutput_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetModbusOutput_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_MODBUS_OUTPUT__BUILDER_HPP_
