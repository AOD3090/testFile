// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/TorqueRtStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__TORQUE_RT_STREAM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__TORQUE_RT_STREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/torque_rt_stream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_TorqueRtStream_time
{
public:
  explicit Init_TorqueRtStream_time(::dsr_msgs2::msg::TorqueRtStream & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::TorqueRtStream time(::dsr_msgs2::msg::TorqueRtStream::_time_type arg)
  {
    msg_.time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::TorqueRtStream msg_;
};

class Init_TorqueRtStream_tor
{
public:
  Init_TorqueRtStream_tor()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TorqueRtStream_time tor(::dsr_msgs2::msg::TorqueRtStream::_tor_type arg)
  {
    msg_.tor = std::move(arg);
    return Init_TorqueRtStream_time(msg_);
  }

private:
  ::dsr_msgs2::msg::TorqueRtStream msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::TorqueRtStream>()
{
  return dsr_msgs2::msg::builder::Init_TorqueRtStream_tor();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__TORQUE_RT_STREAM__BUILDER_HPP_
