// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/SpeedjStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SPEEDJ_STREAM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__SPEEDJ_STREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/speedj_stream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_SpeedjStream_time
{
public:
  explicit Init_SpeedjStream_time(::dsr_msgs2::msg::SpeedjStream & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::SpeedjStream time(::dsr_msgs2::msg::SpeedjStream::_time_type arg)
  {
    msg_.time = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::SpeedjStream msg_;
};

class Init_SpeedjStream_acc
{
public:
  explicit Init_SpeedjStream_acc(::dsr_msgs2::msg::SpeedjStream & msg)
  : msg_(msg)
  {}
  Init_SpeedjStream_time acc(::dsr_msgs2::msg::SpeedjStream::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_SpeedjStream_time(msg_);
  }

private:
  ::dsr_msgs2::msg::SpeedjStream msg_;
};

class Init_SpeedjStream_vel
{
public:
  Init_SpeedjStream_vel()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SpeedjStream_acc vel(::dsr_msgs2::msg::SpeedjStream::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_SpeedjStream_acc(msg_);
  }

private:
  ::dsr_msgs2::msg::SpeedjStream msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::SpeedjStream>()
{
  return dsr_msgs2::msg::builder::Init_SpeedjStream_vel();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__SPEEDJ_STREAM__BUILDER_HPP_
