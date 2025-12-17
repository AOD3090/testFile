// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/ModbusState.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__MODBUS_STATE__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__MODBUS_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/modbus_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_ModbusState_modbus_value
{
public:
  explicit Init_ModbusState_modbus_value(::dsr_msgs2::msg::ModbusState & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::ModbusState modbus_value(::dsr_msgs2::msg::ModbusState::_modbus_value_type arg)
  {
    msg_.modbus_value = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::ModbusState msg_;
};

class Init_ModbusState_modbus_symbol
{
public:
  Init_ModbusState_modbus_symbol()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ModbusState_modbus_value modbus_symbol(::dsr_msgs2::msg::ModbusState::_modbus_symbol_type arg)
  {
    msg_.modbus_symbol = std::move(arg);
    return Init_ModbusState_modbus_value(msg_);
  }

private:
  ::dsr_msgs2::msg::ModbusState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::ModbusState>()
{
  return dsr_msgs2::msg::builder::Init_ModbusState_modbus_symbol();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__MODBUS_STATE__BUILDER_HPP_
