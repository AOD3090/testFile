// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/ChangeOperationSpeed.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CHANGE_OPERATION_SPEED__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__CHANGE_OPERATION_SPEED__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/change_operation_speed__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ChangeOperationSpeed_Request_speed
{
public:
  Init_ChangeOperationSpeed_Request_speed()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ChangeOperationSpeed_Request speed(::dsr_msgs2::srv::ChangeOperationSpeed_Request::_speed_type arg)
  {
    msg_.speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ChangeOperationSpeed_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ChangeOperationSpeed_Request>()
{
  return dsr_msgs2::srv::builder::Init_ChangeOperationSpeed_Request_speed();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ChangeOperationSpeed_Response_success
{
public:
  Init_ChangeOperationSpeed_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ChangeOperationSpeed_Response success(::dsr_msgs2::srv::ChangeOperationSpeed_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ChangeOperationSpeed_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ChangeOperationSpeed_Response>()
{
  return dsr_msgs2::srv::builder::Init_ChangeOperationSpeed_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__CHANGE_OPERATION_SPEED__BUILDER_HPP_
