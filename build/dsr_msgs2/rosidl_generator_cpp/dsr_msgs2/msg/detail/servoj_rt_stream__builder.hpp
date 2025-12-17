// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/ServojRtStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SERVOJ_RT_STREAM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__SERVOJ_RT_STREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/servoj_rt_stream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_ServojRtStream_time
{
public:
  explicit Init_ServojRtStream_time(::dsr_msgs2::msg::ServojRtStream & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::ServojRtStream time(::dsr_msgs2::msg::ServojRtStream::_time_type arg)
  {
    msg_.time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::ServojRtStream msg_;
};

class Init_ServojRtStream_acc
{
public:
  explicit Init_ServojRtStream_acc(::dsr_msgs2::msg::ServojRtStream & msg)
  : msg_(msg)
  {}
  Init_ServojRtStream_time acc(::dsr_msgs2::msg::ServojRtStream::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_ServojRtStream_time(msg_);
  }

private:
  ::dsr_msgs2::msg::ServojRtStream msg_;
};

class Init_ServojRtStream_vel
{
public:
  explicit Init_ServojRtStream_vel(::dsr_msgs2::msg::ServojRtStream & msg)
  : msg_(msg)
  {}
  Init_ServojRtStream_acc vel(::dsr_msgs2::msg::ServojRtStream::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_ServojRtStream_acc(msg_);
  }

private:
  ::dsr_msgs2::msg::ServojRtStream msg_;
};

class Init_ServojRtStream_pos
{
public:
  Init_ServojRtStream_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ServojRtStream_vel pos(::dsr_msgs2::msg::ServojRtStream::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_ServojRtStream_vel(msg_);
  }

private:
  ::dsr_msgs2::msg::ServojRtStream msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::ServojRtStream>()
{
  return dsr_msgs2::msg::builder::Init_ServojRtStream_pos();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__SERVOJ_RT_STREAM__BUILDER_HPP_
