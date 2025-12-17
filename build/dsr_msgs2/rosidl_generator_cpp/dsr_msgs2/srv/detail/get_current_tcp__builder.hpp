// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/GetCurrentTcp.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__GET_CURRENT_TCP__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__GET_CURRENT_TCP__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/get_current_tcp__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{


}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetCurrentTcp_Request>()
{
  return ::dsr_msgs2::srv::GetCurrentTcp_Request(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_GetCurrentTcp_Response_success
{
public:
  explicit Init_GetCurrentTcp_Response_success(::dsr_msgs2::srv::GetCurrentTcp_Response & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::srv::GetCurrentTcp_Response success(::dsr_msgs2::srv::GetCurrentTcp_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCurrentTcp_Response msg_;
};

class Init_GetCurrentTcp_Response_info
{
public:
  Init_GetCurrentTcp_Response_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GetCurrentTcp_Response_success info(::dsr_msgs2::srv::GetCurrentTcp_Response::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_GetCurrentTcp_Response_success(msg_);
  }

private:
  ::dsr_msgs2::srv::GetCurrentTcp_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::GetCurrentTcp_Response>()
{
  return dsr_msgs2::srv::builder::Init_GetCurrentTcp_Response_info();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__GET_CURRENT_TCP__BUILDER_HPP_
