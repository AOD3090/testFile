// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetCurrentTcp.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_CURRENT_TCP__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_CURRENT_TCP__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_current_tcp__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetCurrentTcp_Request_name
{
public:
  Init_SetCurrentTcp_Request_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetCurrentTcp_Request name(::dsr_msgs2::srv::SetCurrentTcp_Request::_name_type arg)
  {
    msg_.name = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCurrentTcp_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetCurrentTcp_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetCurrentTcp_Request_name();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetCurrentTcp_Response_success
{
public:
  Init_SetCurrentTcp_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetCurrentTcp_Response success(::dsr_msgs2::srv::SetCurrentTcp_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetCurrentTcp_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetCurrentTcp_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetCurrentTcp_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_CURRENT_TCP__BUILDER_HPP_
