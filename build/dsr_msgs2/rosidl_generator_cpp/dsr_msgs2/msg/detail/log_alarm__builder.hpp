// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:msg/LogAlarm.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__MSG__DETAIL__LOG_ALARM__BUILDER_HPP_
#define DSR_MSGS2__MSG__DETAIL__LOG_ALARM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/msg/detail/log_alarm__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace msg
{

namespace builder
{

class Init_LogAlarm_param
{
public:
  explicit Init_LogAlarm_param(::dsr_msgs2::msg::LogAlarm & msg)
  : msg_(msg)
  {}
  ::dsr_msgs2::msg::LogAlarm param(::dsr_msgs2::msg::LogAlarm::_param_type arg)
  {
    msg_.param = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::msg::LogAlarm msg_;
};

class Init_LogAlarm_index
{
public:
  explicit Init_LogAlarm_index(::dsr_msgs2::msg::LogAlarm & msg)
  : msg_(msg)
  {}
  Init_LogAlarm_param index(::dsr_msgs2::msg::LogAlarm::_index_type arg)
  {
    msg_.index = std::move(arg);
    return Init_LogAlarm_param(msg_);
  }

private:
  ::dsr_msgs2::msg::LogAlarm msg_;
};

class Init_LogAlarm_group
{
public:
  explicit Init_LogAlarm_group(::dsr_msgs2::msg::LogAlarm & msg)
  : msg_(msg)
  {}
  Init_LogAlarm_index group(::dsr_msgs2::msg::LogAlarm::_group_type arg)
  {
    msg_.group = std::move(arg);
    return Init_LogAlarm_index(msg_);
  }

private:
  ::dsr_msgs2::msg::LogAlarm msg_;
};

class Init_LogAlarm_level
{
public:
  Init_LogAlarm_level()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LogAlarm_group level(::dsr_msgs2::msg::LogAlarm::_level_type arg)
  {
    msg_.level = std::move(arg);
    return Init_LogAlarm_group(msg_);
  }

private:
  ::dsr_msgs2::msg::LogAlarm msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::msg::LogAlarm>()
{
  return dsr_msgs2::msg::builder::Init_LogAlarm_level();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__MSG__DETAIL__LOG_ALARM__BUILDER_HPP_
