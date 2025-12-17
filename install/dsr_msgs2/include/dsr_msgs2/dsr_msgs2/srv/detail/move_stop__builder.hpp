// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/MoveStop.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__MOVE_STOP__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__MOVE_STOP__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/move_stop__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MoveStop_Request_stop_mode
{
public:
  Init_MoveStop_Request_stop_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::MoveStop_Request stop_mode(::dsr_msgs2::srv::MoveStop_Request::_stop_mode_type arg)
  {
    msg_.stop_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveStop_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MoveStop_Request>()
{
  return dsr_msgs2::srv::builder::Init_MoveStop_Request_stop_mode();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_MoveStop_Response_success
{
public:
  Init_MoveStop_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::MoveStop_Response success(::dsr_msgs2::srv::MoveStop_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::MoveStop_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::MoveStop_Response>()
{
  return dsr_msgs2::srv::builder::Init_MoveStop_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__MOVE_STOP__BUILDER_HPP_
