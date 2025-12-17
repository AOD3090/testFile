// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/ConfigDeleteTool.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CONFIG_DELETE_TOOL__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__CONFIG_DELETE_TOOL__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/config_delete_tool__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ConfigDeleteTool_Request_name
{
public:
  Init_ConfigDeleteTool_Request_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ConfigDeleteTool_Request name(::dsr_msgs2::srv::ConfigDeleteTool_Request::_name_type arg)
  {
    msg_.name = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigDeleteTool_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ConfigDeleteTool_Request>()
{
  return dsr_msgs2::srv::builder::Init_ConfigDeleteTool_Request_name();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ConfigDeleteTool_Response_success
{
public:
  Init_ConfigDeleteTool_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ConfigDeleteTool_Response success(::dsr_msgs2::srv::ConfigDeleteTool_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ConfigDeleteTool_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ConfigDeleteTool_Response>()
{
  return dsr_msgs2::srv::builder::Init_ConfigDeleteTool_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__CONFIG_DELETE_TOOL__BUILDER_HPP_
