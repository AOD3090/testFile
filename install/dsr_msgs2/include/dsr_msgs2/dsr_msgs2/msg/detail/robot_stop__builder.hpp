// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/RobotStop.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/robot_stop__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_RobotStop_stop_mode
{
public:
  Init_RobotStop_stop_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::msg::RobotStop stop_mode(::dsr_msgs2::msg::RobotStop::_stop_mode_type arg)
  {
    msg_.stop_mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::RobotStop msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::RobotStop>()
{
  return dsr_msgs2::msg::builder::Init_RobotStop_stop_mode();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__ROBOT_STOP__BUILDER_HPP_
