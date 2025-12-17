// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/SpeedlRtStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SPEEDL_RT_STREAM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__SPEEDL_RT_STREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/speedl_rt_stream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_SpeedlRtStream_time
{
public:
  explicit Init_SpeedlRtStream_time(::dsr_msgs2::msg::SpeedlRtStream & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::SpeedlRtStream time(::dsr_msgs2::msg::SpeedlRtStream::_time_type arg)
  {
    msg_.time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::SpeedlRtStream msg_;
};

class Init_SpeedlRtStream_acc
{
public:
  explicit Init_SpeedlRtStream_acc(::dsr_msgs2::msg::SpeedlRtStream & msg)
  : msg_(msg)
  {}
  Init_SpeedlRtStream_time acc(::dsr_msgs2::msg::SpeedlRtStream::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_SpeedlRtStream_time(msg_);
  }

private:
  ::dsr_msgs2::msg::SpeedlRtStream msg_;
};

class Init_SpeedlRtStream_vel
{
public:
  Init_SpeedlRtStream_vel()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SpeedlRtStream_acc vel(::dsr_msgs2::msg::SpeedlRtStream::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_SpeedlRtStream_acc(msg_);
  }

private:
  ::dsr_msgs2::msg::SpeedlRtStream msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::SpeedlRtStream>()
{
  return dsr_msgs2::msg::builder::Init_SpeedlRtStream_vel();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__SPEEDL_RT_STREAM__BUILDER_HPP_
