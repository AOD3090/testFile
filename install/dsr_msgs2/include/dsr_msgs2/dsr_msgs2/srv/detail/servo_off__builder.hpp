// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dsr_msgs2:srv/ServoOff.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__SERVO_OFF__BUILDER_HPP_
#define DSR_MSGS2__SRV__DETAIL__SERVO_OFF__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dsr_msgs2/srv/detail/servo_off__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ServoOff_Request_stop_type
{
public:
  Init_ServoOff_Request_stop_type()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ServoOff_Request stop_type(::dsr_msgs2::srv::ServoOff_Request::_stop_type_type arg)
  {
    msg_.stop_type = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ServoOff_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ServoOff_Request>()
{
  return dsr_msgs2::srv::builder::Init_ServoOff_Request_stop_type();
}

}  // namespace dsr_msgs2


namespace dsr_msgs2
{

namespace srv
{

namespace builder
{

class Init_ServoOff_Response_success
{
public:
  Init_ServoOff_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::dsr_msgs2::srv::ServoOff_Response success(::dsr_msgs2::srv::ServoOff_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dsr_msgs2::srv::ServoOff_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::dsr_msgs2::srv::ServoOff_Response>()
{
  return dsr_msgs2::srv::builder::Init_ServoOff_Response_success();
}

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__SERVO_OFF__BUILDER_HPP_
