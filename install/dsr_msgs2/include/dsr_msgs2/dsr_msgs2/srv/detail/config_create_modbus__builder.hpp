// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/ConfigCreateModbus.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CONFIG_CREATE_MODBUS__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__CONFIG_CREATE_MODBUS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/config_create_modbus__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ConfigCreateModbus_Request_slave_id
{
public:
  explicit Init_ConfigCreateModbus_Request_slave_id(::dsr_msgs2::srv::ConfigCreateModbus_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::ConfigCreateModbus_Request slave_id(::dsr_msgs2::srv::ConfigCreateModbus_Request::_slave_id_type arg)
  {
    msg_.slave_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateModbus_Request msg_;
};

class Init_ConfigCreateModbus_Request_value
{
public:
  explicit Init_ConfigCreateModbus_Request_value(::dsr_msgs2::srv::ConfigCreateModbus_Request & msg)
  : msg_(msg)
  {}
  Init_ConfigCreateModbus_Request_slave_id value(::dsr_msgs2::srv::ConfigCreateModbus_Request::_value_type arg)
  {
    msg_.value = std::move(arg);
    return Init_ConfigCreateModbus_Request_slave_id(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateModbus_Request msg_;
};

class Init_ConfigCreateModbus_Request_index
{
public:
  explicit Init_ConfigCreateModbus_Request_index(::dsr_msgs2::srv::ConfigCreateModbus_Request & msg)
  : msg_(msg)
  {}
  Init_ConfigCreateModbus_Request_value index(::dsr_msgs2::srv::ConfigCreateModbus_Request::_index_type arg)
  {
    msg_.index = std::move(arg);
    return Init_ConfigCreateModbus_Request_value(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateModbus_Request msg_;
};

class Init_ConfigCreateModbus_Request_reg_type
{
public:
  explicit Init_ConfigCreateModbus_Request_reg_type(::dsr_msgs2::srv::ConfigCreateModbus_Request & msg)
  : msg_(msg)
  {}
  Init_ConfigCreateModbus_Request_index reg_type(::dsr_msgs2::srv::ConfigCreateModbus_Request::_reg_type_type arg)
  {
    msg_.reg_type = std::move(arg);
    return Init_ConfigCreateModbus_Request_index(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateModbus_Request msg_;
};

class Init_ConfigCreateModbus_Request_port
{
public:
  explicit Init_ConfigCreateModbus_Request_port(::dsr_msgs2::srv::ConfigCreateModbus_Request & msg)
  : msg_(msg)
  {}
  Init_ConfigCreateModbus_Request_reg_type port(::dsr_msgs2::srv::ConfigCreateModbus_Request::_port_type arg)
  {
    msg_.port = std::move(arg);
    return Init_ConfigCreateModbus_Request_reg_type(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateModbus_Request msg_;
};

class Init_ConfigCreateModbus_Request_ip
{
public:
  explicit Init_ConfigCreateModbus_Request_ip(::dsr_msgs2::srv::ConfigCreateModbus_Request & msg)
  : msg_(msg)
  {}
  Init_ConfigCreateModbus_Request_port ip(::dsr_msgs2::srv::ConfigCreateModbus_Request::_ip_type arg)
  {
    msg_.ip = std::move(arg);
    return Init_ConfigCreateModbus_Request_port(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateModbus_Request msg_;
};

class Init_ConfigCreateModbus_Request_name
{
public:
  Init_ConfigCreateModbus_Request_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ConfigCreateModbus_Request_ip name(::dsr_msgs2::srv::ConfigCreateModbus_Request::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_ConfigCreateModbus_Request_ip(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateModbus_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ConfigCreateModbus_Request>()
{
  return dsr_msgs2::srv::builder::Init_ConfigCreateModbus_Request_name();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ConfigCreateModbus_Response_success
{
public:
  Init_ConfigCreateModbus_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ConfigCreateModbus_Response success(::dsr_msgs2::srv::ConfigCreateModbus_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateModbus_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ConfigCreateModbus_Response>()
{
  return dsr_msgs2::srv::builder::Init_ConfigCreateModbus_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__CONFIG_CREATE_MODBUS__BUILDER_HPP_
