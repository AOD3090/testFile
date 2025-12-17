// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/SetSafeStopResetType.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SET_SAFE_STOP_RESET_TYPE__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SET_SAFE_STOP_RESET_TYPE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/set_safe_stop_reset_type__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetSafeStopResetType_Request_reset_type
{
public:
  Init_SetSafeStopResetType_Request_reset_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetSafeStopResetType_Request reset_type(::dsr_msgs2::srv::SetSafeStopResetType_Request::_reset_type_type arg)
  {
    msg_.reset_type = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetSafeStopResetType_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetSafeStopResetType_Request>()
{
  return dsr_msgs2::srv::builder::Init_SetSafeStopResetType_Request_reset_type();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_SetSafeStopResetType_Response_success
{
public:
  Init_SetSafeStopResetType_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::SetSafeStopResetType_Response success(::dsr_msgs2::srv::SetSafeStopResetType_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::SetSafeStopResetType_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::SetSafeStopResetType_Response>()
{
  return dsr_msgs2::srv::builder::Init_SetSafeStopResetType_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SET_SAFE_STOP_RESET_TYPE__BUILDER_HPP_
