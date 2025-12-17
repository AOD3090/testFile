// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/SpeedjRtStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SPEEDJ_RT_STREAM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__SPEEDJ_RT_STREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/speedj_rt_stream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_SpeedjRtStream_time
{
public:
  explicit Init_SpeedjRtStream_time(::dsr_msgs2::msg::SpeedjRtStream & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::SpeedjRtStream time(::dsr_msgs2::msg::SpeedjRtStream::_time_type arg)
  {
    msg_.time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::SpeedjRtStream msg_;
};

class Init_SpeedjRtStream_acc
{
public:
  explicit Init_SpeedjRtStream_acc(::dsr_msgs2::msg::SpeedjRtStream & msg)
  : msg_(msg)
  {}
  Init_SpeedjRtStream_time acc(::dsr_msgs2::msg::SpeedjRtStream::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_SpeedjRtStream_time(msg_);
  }

private:
  ::dsr_msgs2::msg::SpeedjRtStream msg_;
};

class Init_SpeedjRtStream_vel
{
public:
  Init_SpeedjRtStream_vel()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SpeedjRtStream_acc vel(::dsr_msgs2::msg::SpeedjRtStream::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_SpeedjRtStream_acc(msg_);
  }

private:
  ::dsr_msgs2::msg::SpeedjRtStream msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::SpeedjRtStream>()
{
  return dsr_msgs2::msg::builder::Init_SpeedjRtStream_vel();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__SPEEDJ_RT_STREAM__BUILDER_HPP_
