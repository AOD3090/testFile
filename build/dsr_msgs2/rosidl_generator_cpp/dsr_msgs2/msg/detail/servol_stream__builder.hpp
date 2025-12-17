// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/ServolStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SERVOL_STREAM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__SERVOL_STREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/servol_stream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_ServolStream_time
{
public:
  explicit Init_ServolStream_time(::dsr_msgs2::msg::ServolStream & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::ServolStream time(::dsr_msgs2::msg::ServolStream::_time_type arg)
  {
    msg_.time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::ServolStream msg_;
};

class Init_ServolStream_acc
{
public:
  explicit Init_ServolStream_acc(::dsr_msgs2::msg::ServolStream & msg)
  : msg_(msg)
  {}
  Init_ServolStream_time acc(::dsr_msgs2::msg::ServolStream::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_ServolStream_time(msg_);
  }

private:
  ::dsr_msgs2::msg::ServolStream msg_;
};

class Init_ServolStream_vel
{
public:
  explicit Init_ServolStream_vel(::dsr_msgs2::msg::ServolStream & msg)
  : msg_(msg)
  {}
  Init_ServolStream_acc vel(::dsr_msgs2::msg::ServolStream::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_ServolStream_acc(msg_);
  }

private:
  ::dsr_msgs2::msg::ServolStream msg_;
};

class Init_ServolStream_pos
{
public:
  Init_ServolStream_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ServolStream_vel pos(::dsr_msgs2::msg::ServolStream::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_ServolStream_vel(msg_);
  }

private:
  ::dsr_msgs2::msg::ServolStream msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::ServolStream>()
{
  return dsr_msgs2::msg::builder::Init_ServolStream_pos();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__SERVOL_STREAM__BUILDER_HPP_
