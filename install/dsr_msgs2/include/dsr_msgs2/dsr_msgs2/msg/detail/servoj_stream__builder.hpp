// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/ServojStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__SERVOJ_STREAM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__SERVOJ_STREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/servoj_stream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_ServojStream_mode
{
public:
  explicit Init_ServojStream_mode(::dsr_msgs2::msg::ServojStream & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::ServojStream mode(::dsr_msgs2::msg::ServojStream::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::ServojStream msg_;
};

class Init_ServojStream_time
{
public:
  explicit Init_ServojStream_time(::dsr_msgs2::msg::ServojStream & msg)
  : msg_(msg)
  {}
  Init_ServojStream_mode time(::dsr_msgs2::msg::ServojStream::_time_type arg)
  {
    msg_.time = std::move(arg);
    return Init_ServojStream_mode(msg_);
  }

private:
  ::dsr_msgs2::msg::ServojStream msg_;
};

class Init_ServojStream_acc
{
public:
  explicit Init_ServojStream_acc(::dsr_msgs2::msg::ServojStream & msg)
  : msg_(msg)
  {}
  Init_ServojStream_time acc(::dsr_msgs2::msg::ServojStream::_acc_type arg)
  {
    msg_.acc = std::move(arg);
    return Init_ServojStream_time(msg_);
  }

private:
  ::dsr_msgs2::msg::ServojStream msg_;
};

class Init_ServojStream_vel
{
public:
  explicit Init_ServojStream_vel(::dsr_msgs2::msg::ServojStream & msg)
  : msg_(msg)
  {}
  Init_ServojStream_acc vel(::dsr_msgs2::msg::ServojStream::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_ServojStream_acc(msg_);
  }

private:
  ::dsr_msgs2::msg::ServojStream msg_;
};

class Init_ServojStream_pos
{
public:
  Init_ServojStream_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ServojStream_vel pos(::dsr_msgs2::msg::ServojStream::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_ServojStream_vel(msg_);
  }

private:
  ::dsr_msgs2::msg::ServojStream msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::ServojStream>()
{
  return dsr_msgs2::msg::builder::Init_ServojStream_pos();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__SERVOJ_STREAM__BUILDER_HPP_
