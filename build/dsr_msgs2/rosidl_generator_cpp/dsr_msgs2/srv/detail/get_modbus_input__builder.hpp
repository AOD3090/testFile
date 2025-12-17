// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetModbusInput.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_MODBUS_INPUT__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_MODBUS_INPUT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_modbus_input__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetModbusInput_Request_name
{
public:
  Init_GetModbusInput_Request_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::GetModbusInput_Request name(::dsr_msgs2::srv::GetModbusInput_Request::_name_type arg)
  {
    msg_.name = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetModbusInput_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetModbusInput_Request>()
{
  return dsr_msgs2::srv::builder::Init_GetModbusInput_Request_name();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetModbusInput_Response_success
{
public:
  explicit Init_GetModbusInput_Response_success(::dsr_msgs2::srv::GetModbusInput_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetModbusInput_Response success(::dsr_msgs2::srv::GetModbusInput_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetModbusInput_Response msg_;
};

class Init_GetModbusInput_Response_value
{
public:
  Init_GetModbusInput_Response_value()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetModbusInput_Response_success value(::dsr_msgs2::srv::GetModbusInput_Response::_value_type arg)
  {
    msg_.value = std::move(arg);
    return Init_GetModbusInput_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetModbusInput_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetModbusInput_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetModbusInput_Response_value();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_MODBUS_INPUT__BUILDER_HPP_
