// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/AlterMotionStream.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__ALTER_MOTION_STREAM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__ALTER_MOTION_STREAM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/alter_motion_stream__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_AlterMotionStream_pos
{
public:
  Init_AlterMotionStream_pos()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::msg::AlterMotionStream pos(::dsr_msgs2::msg::AlterMotionStream::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::AlterMotionStream msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::AlterMotionStream>()
{
  return dsr_msgs2::msg::builder::Init_AlterMotionStream_pos();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__ALTER_MOTION_STREAM__BUILDER_HPP_
