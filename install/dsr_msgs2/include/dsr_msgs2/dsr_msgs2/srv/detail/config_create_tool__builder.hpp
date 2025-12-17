// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/ConfigCreateTool.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CONFIG_CREATE_TOOL__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__CONFIG_CREATE_TOOL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/config_create_tool__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ConfigCreateTool_Request_inertia
{
public:
  explicit Init_ConfigCreateTool_Request_inertia(::dsr_msgs2::srv::ConfigCreateTool_Request & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::ConfigCreateTool_Request inertia(::dsr_msgs2::srv::ConfigCreateTool_Request::_inertia_type arg)
  {
    msg_.inertia = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateTool_Request msg_;
};

class Init_ConfigCreateTool_Request_cog
{
public:
  explicit Init_ConfigCreateTool_Request_cog(::dsr_msgs2::srv::ConfigCreateTool_Request & msg)
  : msg_(msg)
  {}
  Init_ConfigCreateTool_Request_inertia cog(::dsr_msgs2::srv::ConfigCreateTool_Request::_cog_type arg)
  {
    msg_.cog = std::move(arg);
    return Init_ConfigCreateTool_Request_inertia(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateTool_Request msg_;
};

class Init_ConfigCreateTool_Request_weight
{
public:
  explicit Init_ConfigCreateTool_Request_weight(::dsr_msgs2::srv::ConfigCreateTool_Request & msg)
  : msg_(msg)
  {}
  Init_ConfigCreateTool_Request_cog weight(::dsr_msgs2::srv::ConfigCreateTool_Request::_weight_type arg)
  {
    msg_.weight = std::move(arg);
    return Init_ConfigCreateTool_Request_cog(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateTool_Request msg_;
};

class Init_ConfigCreateTool_Request_name
{
public:
  Init_ConfigCreateTool_Request_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ConfigCreateTool_Request_weight name(::dsr_msgs2::srv::ConfigCreateTool_Request::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_ConfigCreateTool_Request_weight(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateTool_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ConfigCreateTool_Request>()
{
  return dsr_msgs2::srv::builder::Init_ConfigCreateTool_Request_name();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ConfigCreateTool_Response_success
{
public:
  Init_ConfigCreateTool_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ConfigCreateTool_Response success(::dsr_msgs2::srv::ConfigCreateTool_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigCreateTool_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ConfigCreateTool_Response>()
{
  return dsr_msgs2::srv::builder::Init_ConfigCreateTool_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__CONFIG_CREATE_TOOL__BUILDER_HPP_
