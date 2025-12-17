// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/JogMultiAxis.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__JOG_MULTI_AXIS__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__JOG_MULTI_AXIS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/jog_multi_axis__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_JogMultiAxis_speed
{
public:
  explicit Init_JogMultiAxis_speed(::dsr_msgs2::msg::JogMultiAxis & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::JogMultiAxis speed(::dsr_msgs2::msg::JogMultiAxis::_speed_type arg)
  {
    msg_.speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::JogMultiAxis msg_;
};

class Init_JogMultiAxis_move_reference
{
public:
  explicit Init_JogMultiAxis_move_reference(::dsr_msgs2::msg::JogMultiAxis & msg)
  : msg_(msg)
  {}
  Init_JogMultiAxis_speed move_reference(::dsr_msgs2::msg::JogMultiAxis::_move_reference_type arg)
  {
    msg_.move_reference = std::move(arg);
    return Init_JogMultiAxis_speed(msg_);
  }

private:
  ::dsr_msgs2::msg::JogMultiAxis msg_;
};

class Init_JogMultiAxis_jog_axis
{
public:
  Init_JogMultiAxis_jog_axis()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JogMultiAxis_move_reference jog_axis(::dsr_msgs2::msg::JogMultiAxis::_jog_axis_type arg)
  {
    msg_.jog_axis = std::move(arg);
    return Init_JogMultiAxis_move_reference(msg_);
  }

private:
  ::dsr_msgs2::msg::JogMultiAxis msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::JogMultiAxis>()
{
  return dsr_msgs2::msg::builder::Init_JogMultiAxis_jog_axis();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__JOG_MULTI_AXIS__BUILDER_HPP_
