// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/ServolRtStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SERVOL_RT_STREAM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__SERVOL_RT_STREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/servol_rt_stream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_ServolRtStream_time
{
public:
  explicit Init_ServolRtStream_time(::dsr_msgs2::msg::ServolRtStream & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::ServolRtStream time(::dsr_msgs2::msg::ServolRtStream::_time_type arg)
  {
    msg_.time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::ServolRtStream msg_;
};

class Init_ServolRtStream_acc
{
public:
  explicit Init_ServolRtStream_acc(::dsr_msgs2::msg::ServolRtStream & msg)
  : msg_(msg)
  {}
  Init_ServolRtStream_time acc(::dsr_msgs2::msg::ServolRtStream::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_ServolRtStream_time(msg_);
  }

private:
  ::dsr_msgs2::msg::ServolRtStream msg_;
};

class Init_ServolRtStream_vel
{
public:
  explicit Init_ServolRtStream_vel(::dsr_msgs2::msg::ServolRtStream & msg)
  : msg_(msg)
  {}
  Init_ServolRtStream_acc vel(::dsr_msgs2::msg::ServolRtStream::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_ServolRtStream_acc(msg_);
  }

private:
  ::dsr_msgs2::msg::ServolRtStream msg_;
};

class Init_ServolRtStream_pos
{
public:
  Init_ServolRtStream_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ServolRtStream_vel pos(::dsr_msgs2::msg::ServolRtStream::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_ServolRtStream_vel(msg_);
  }

private:
  ::dsr_msgs2::msg::ServolRtStream msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::ServolRtStream>()
{
  return dsr_msgs2::msg::builder::Init_ServolRtStream_pos();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__SERVOL_RT_STREAM__BUILDER_HPP_
